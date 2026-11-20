// Standalone benchmark of hh::Router on an exported index (no Python / BLAS threads in the process).
// Usage: hullbench <export_dir> <repeats> T,F,V[,R[,G[,H,P,K,E]]] ...
//   R = pass-2 refine count, G = gap records, H,P,K,E = graph hops, expand_from, expand_k, expand_exact
// Env: HH_PREFIX_DIMS (pass-1 sketch dims), HH_FINE_DIMS (fine centroid dims used), HH_SAME_QUERY,
//   HH_GRAPH=k:T,F,V,R build a k-NN graph by self-search (HH_GRAPH_THREADS, default 32),
//   HH_GRAPH_FILE=<int32 n×32 .bin> load an exact graph instead (also the reference for graph recall).
// Prints recall@10, µs/query (single thread, best of repeats) and per-stage µs.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../cpp/router.hpp"

namespace {

struct Table {
    const void* data;
    std::vector<size_t> shape;
};

const void* map_file(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st {};
    fstat(fd, &st);
    // Anonymous 2 MB-aligned copy: file-backed mappings do not get transparent huge pages, and the
    // scattered rerank / sketch reads would otherwise pay a page walk per access.
    constexpr size_t kHugePage = size_t{2} << 20;
    const size_t bytes = (static_cast<size_t>(st.st_size) + kHugePage - 1) / kHugePage * kHugePage;
    void* p = std::aligned_alloc(kHugePage, bytes);
    if (p == nullptr) throw std::runtime_error("alloc failed " + path);
    madvise(p, bytes, MADV_HUGEPAGE);
    for (size_t done = 0; done < static_cast<size_t>(st.st_size);) {
        const ssize_t got = pread(fd, static_cast<char*>(p) + done, st.st_size - done, done);
        if (got <= 0) throw std::runtime_error("read failed " + path);
        done += got;
    }
    close(fd);
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <export_dir> <repeats> T,F,V ...\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1];
    const int repeats = std::stoi(argv[2]);
    std::map<std::string, Table> t;
    size_t m = 0;
    float gamma = 0;
    std::ifstream meta(dir + "/meta.txt");
    for (std::string line; std::getline(meta, line);) {
        std::istringstream in(line);
        std::string kind, name, dtype;
        in >> kind;
        if (kind == "fine_per_coarse") in >> m;
        if (kind == "gamma") in >> gamma;
        if (kind != "array") continue;
        in >> name >> dtype;
        Table tab{map_file(dir + "/" + name + ".bin"), {}};
        for (size_t v; in >> v;) tab.shape.push_back(v);
        t[name] = tab;
    }
    auto f = [&](const char* n) { return static_cast<const float*>(t.at(n).data); };
    auto u8 = [&](const char* n) { return static_cast<const uint8_t*>(t.at(n).data); };
    auto i64 = [&](const char* n) { return static_cast<const int64_t*>(t.at(n).data); };
    const size_t n = t["vectors"].shape[0], dim = t["vectors"].shape[1];
    const size_t pca = t["proj"].shape[0];
    hh::IndexView ix{n, dim, pca, t["coarse_z"].shape[1], 2 * t["fine_codes"].shape[1],
                     8 * t["sketch_codes"].shape[1] / pca, t["coarse_z"].shape[0], m,
                     f("mean"), f("proj"), f("proj_offset"), f("coarse_z"), u8("fine_codes"), f("fine_lo"),
                     f("fine_step"), f("fine_bias"), u8("sketch_codes"), f("sketch_lo"), f("sketch_step"),
                     f("sketch_bias"), f("sketch_resid"), f("vectors"), i64("ids"), i64("coarse_offsets"),
                     i64("fine_offsets"), gamma};
    auto env_size = [](const char* name) { const char* v = std::getenv(name); return v ? std::stoul(v) : 0ul; };
    ix.prefix_dims = env_size("HH_PREFIX_DIMS");
    ix.fine_used_dims = env_size("HH_FINE_DIMS");
    hh::Router router(ix);
    const size_t n_threads = env_size("HH_GRAPH_THREADS") ? env_size("HH_GRAPH_THREADS") : 32;
    constexpr size_t kExactK = 32;
    std::vector<uint32_t> exact;
    if (const char* file = std::getenv("HH_GRAPH_FILE")) {
        exact.resize(n * kExactK);
        std::ifstream(file, std::ios::binary).read(reinterpret_cast<char*>(exact.data()), exact.size() * sizeof(uint32_t));
    }
    if (const char* spec = std::getenv("HH_GRAPH")) {
        size_t k = 0, T = 0, F = 0, V = 0, R = 0;
        std::sscanf(spec, "%zu:%zu,%zu,%zu,%zu", &k, &T, &F, &V, &R);
        const auto t0 = std::chrono::steady_clock::now();
        router.build_graph(k, {k, T, F, V, R}, n_threads, env_size("HH_GRAPH_REVERSE") > 0);
        auto report = [&](const char* what) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            double overlap = 0;
            if (!exact.empty()) {  // fraction of the exact k nearest neighbours found
                for (size_t r = 0; r < n; ++r) {
                    std::unordered_set<uint32_t> truth(exact.begin() + r * kExactK, exact.begin() + r * kExactK + k);
                    for (size_t j = 0; j < k; ++j) overlap += truth.count(router.graph()[r * k + j]);
                }
                overlap /= double(n * k);
            }
            std::printf("graph k=%zu via %zu,%zu,%zu,%zu %s: %.2f s on %zu threads, recall vs exact %.4f\n", k, T, F, V, R, what, s,
                        n_threads, overlap);
        };
        report("search");
        // HH_GRAPH_REBUILD=T,F,V,R,G,H,P,K,E: second self-search pass expanded by the first graph
        if (const char* rebuild = std::getenv("HH_GRAPH_REBUILD")) {
            hh::SearchParams q{.k = k, .shortlist = 0, .fine_cells = 0, .rerank = 0};
            std::sscanf(rebuild, "%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu", &q.shortlist, &q.fine_cells, &q.rerank, &q.refine, &q.gap,
                        &q.hops, &q.expand_from, &q.expand_k, &q.expand_exact);
            router.build_graph(k, q, n_threads, env_size("HH_GRAPH_REVERSE") > 0);
            report("+rebuild");
        }
        // HH_GRAPH_REFINE=fanout[,rounds]: NN-descent rounds after the search (timed cumulatively, recall check excluded)
        if (const char* refine = std::getenv("HH_GRAPH_REFINE")) {
            size_t fanout = 0, rounds = 1;
            std::sscanf(refine, "%zu,%zu", &fanout, &rounds);
            for (size_t i = 0; i < rounds; ++i) {
                const auto r0 = std::chrono::steady_clock::now();
                router.refine_graph(fanout, n_threads);
                std::printf("  refine round %zu (fanout %zu): %.2f s\n", i + 1, fanout,
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count());
                report("+refine");
            }
        }
    } else if (!exact.empty()) {
        router.set_graph(exact.data(), kExactK);
        std::printf("graph: exact %zu-NN loaded\n", kExactK);
    }
    const float* queries = f("queries");
    std::vector<float> same_query;  // HH_SAME_QUERY=1: every query = query 0 (data cache-hot → compute floor)
    if (std::getenv("HH_SAME_QUERY")) {
        for (size_t q = 0; q < t["queries"].shape[0]; ++q) same_query.insert(same_query.end(), queries, queries + dim);
        queries = same_query.data();
    }
    const int64_t* gt = i64("gt");
    const size_t nq = t["queries"].shape[0], k = 10;
    std::printf("N=%zu D=%zu Q=%zu prefix=%zu fine=%zu\n%-30s %7s %9s  %7s %7s %7s %7s %7s %7s %7s  %8s %8s\n", n, dim, nq,
                ix.prefix_dims, ix.fine_used_dims, "T,F,V,R,G,H,P,K,E", "recall", "us/query", "project", "coarse", "fine",
                "sketch", "refine", "rerank", "expand", "sketched", "expanded");
    std::vector<float> dist(nq * k);
    std::vector<int64_t> ids(nq * k);
    std::vector<hh::QueryStats> stats(nq);
    for (int a = 3; a < argc; ++a) {
        size_t T = 0, F = 0, V = 0, R = 0, G = 0, H = 0, P = 0, K = 0, E = 0;
        std::sscanf(argv[a], "%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu", &T, &F, &V, &R, &G, &H, &P, &K, &E);
        const hh::SearchParams p{k, T, F, V, R, G, H, P, K, E};
        double best = 1e30, stage[hh::kStages]{};
        for (int r = 0; r < repeats; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            router.search_batch(queries, nq, p, 1, dist.data(), ids.data(), stats.data());
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / nq;
            if (us < best) {
                best = us;
                std::fill(stage, stage + hh::kStages, 0.0);
                for (const auto& s : stats)
                    for (int st = 0; st < hh::kStages; ++st) stage[st] += s.stage_ns[st] / 1e3 / nq;
            }
        }
        size_t hits = 0;
        double sketched = 0, expanded = 0;
        for (size_t q = 0; q < nq; ++q) {
            std::unordered_set<int64_t> truth(gt + q * k, gt + q * k + k);
            for (size_t j = 0; j < k; ++j) hits += truth.count(ids[q * k + j]);
            sketched += stats[q].sketched / double(nq);
            expanded += stats[q].expanded / double(nq);
        }
        std::printf("%-30s %7.3f %9.1f  %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f  %8.0f %8.0f\n", argv[a], hits / double(nq * k), best,
                    stage[0], stage[1], stage[2], stage[3], stage[4], stage[5], stage[6], sketched, expanded);
    }
}
