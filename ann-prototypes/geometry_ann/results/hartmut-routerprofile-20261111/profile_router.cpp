// Where the two-level router's fixed cost goes (#47 follow-up, the <=0.945 tie of #52). Loads a saved TieredQ IVF
// index, builds the router at the shipping setting (expansion 64, at least 32 supers), and times on one pinned core,
// per nprobe: probe_lists alone (router), the 2-bit search with the router, and with exact routing. With loop=SECONDS
// it then repeats the routed search at the first nprobe for that long, for `perf record`.
#include <tieredq/tieredq.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {
std::vector<float> read_fvecs(const std::string& path, std::size_t dim, std::size_t rows) {
    std::ifstream in(path, std::ios::binary);
    std::vector<float> out(rows * dim);
    for (std::size_t r = 0; r < rows; ++r) {
        std::int32_t width = 0;
        in.read(reinterpret_cast<char*>(&width), sizeof(width));
        std::vector<float> row(static_cast<std::size_t>(width));
        in.read(reinterpret_cast<char*>(row.data()), std::streamsize(row.size() * sizeof(float)));
        if (!in || std::size_t(width) < dim) throw std::runtime_error("short query file");
        std::copy_n(row.begin(), dim, out.begin() + std::ptrdiff_t(r * dim));
    }
    return out;
}

template<class F> double p50_us(std::size_t count, F&& run) {
    std::vector<double> us(count);
    for (std::size_t q = 0; q < count; ++q) {
        auto t = Clock::now(); run(q);
        us[q] = std::chrono::duration<double, std::micro>(Clock::now() - t).count();
    }
    std::nth_element(us.begin(), us.begin() + std::ptrdiff_t(count / 2), us.end());
    return us[count / 2];
}
}  // namespace

int main(int argc, char** argv) try {
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i]; auto eq = a.find('='); kv[a.substr(0, eq)] = a.substr(eq + 1); }
    const std::size_t dim = 1024, count = std::stoul(kv.at("count")), top = 40;
    std::vector<std::size_t> nprobes; { std::stringstream s(kv.at("nprobes")); std::string v; while (std::getline(s, v, ',')) nprobes.push_back(std::stoul(v)); }
    const double loop_s = kv.count("loop") ? std::stod(kv.at("loop")) : 0;

    auto ivf = dv::TieredQIVFIndex::load(kv.at("index"));
    dv::TieredQRouterConfig router; router.expansion = 64; router.min_super_probes = 32;
    ivf.build_router(router);
    const auto queries = read_fvecs(kv.at("queries"), dim, count);
    cpu_set_t only; CPU_ZERO(&only); CPU_SET(std::stoi(kv.at("cpu")), &only);
    if (sched_setaffinity(0, sizeof(only), &only)) throw std::runtime_error("cannot pin");

    const dv::TieredQRouting routed{dv::TieredQRouting::Mode::router}, exact{dv::TieredQRouting::Mode::exact};
    if (!kv.count("rerank_depths")) std::printf("nprobe,probe_lists_us,search_router_us,search_exact_us\n");
    for (std::size_t nprobe : kv.count("rerank_depths") ? std::vector<std::size_t>{} : nprobes) {
        auto q = [&](std::size_t i) { return queries.data() + i * dim; };
        ivf.search(q(0), top, nprobe, {}, routed);  // warm
        const double probe = p50_us(count, [&](std::size_t i) { ivf.probe_lists(q(i), nprobe, routed); });
        const double search = p50_us(count, [&](std::size_t i) { ivf.search(q(i), top, nprobe, {}, routed); });
        const double full = p50_us(count, [&](std::size_t i) { ivf.search(q(i), top, nprobe, {}, exact); });
        std::printf("%zu,%.1f,%.1f,%.1f\n", nprobe, probe, search, full);
        std::fflush(stdout);
    }
    // Decomposition without perf (hartmut blocks perf events): probe_lists at fixed super-probe counts, with the
    // binary16 re-rank of the head (rerank 256) and without it (rerank 0). Slope over supers = member scan per super;
    // intercept = query preparation + super ranking; the rerank difference = the exact head re-rank.
    if (kv.count("decompose")) {
        std::printf("rerank,super_probes,probe_lists_us\n");
        for (std::uint32_t rerank : {256u, 0u}) {
            router.rerank = rerank;
            ivf.build_router(router);
            for (std::uint32_t supers : {1u, 4u, 16u, 32u, 64u, 128u}) {
                const dv::TieredQRouting fixed{dv::TieredQRouting::Mode::router, supers};
                ivf.probe_lists(queries.data(), 64, fixed);
                const double us = p50_us(count, [&](std::size_t i) { ivf.probe_lists(queries.data() + i * dim, 64, fixed); });
                std::printf("%u,%u,%.1f\n", rerank, supers, us);
                std::fflush(stdout);
            }
        }
    }
    // Router re-rank depth: full queries (router, 2-bit scan, exact SQ8 re-rank of the top 40) per rerank and nprobe,
    // p50 and recall@10 against the truth. codes/scales/offsets/truth/rows give the SQ8 store.
    if (kv.count("rerank_depths")) {
        const std::size_t rows = std::stoul(kv.at("rows"));
        const int fd = ::open(kv.at("codes").c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open codes");
        const auto* codes = static_cast<const std::int8_t*>(::mmap(nullptr, rows * dim, PROT_READ, MAP_SHARED, fd, 0));
        ::close(fd);
        auto read_floats = [](const std::string& path) {
            std::ifstream in(path, std::ios::binary | std::ios::ate); std::vector<float> v(std::size_t(in.tellg()) / 4);
            in.seekg(0); in.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4)); return v;
        };
        const auto scales = read_floats(kv.at("scales")), offsets = read_floats(kv.at("offsets"));
        std::vector<std::int32_t> truth(count * 10);
        {
            std::ifstream in(kv.at("truth"), std::ios::binary);
            for (std::size_t r = 0; r < count; ++r) {
                std::int32_t width = 0; in.read(reinterpret_cast<char*>(&width), 4);
                std::vector<std::int32_t> row(static_cast<std::size_t>(width));
                in.read(reinterpret_cast<char*>(row.data()), std::streamsize(row.size() * 4));
                std::copy_n(row.begin(), 10, truth.begin() + std::ptrdiff_t(r * 10));
            }
        }
        auto full = [&](const float* q, std::size_t nprobe) {
            auto found = ivf.search(q, top, nprobe, {}, routed);
            for (auto& hit : found) {
                const std::int8_t* code = codes + std::size_t(hit.id) * dim; float sum = 0;
                for (std::size_t j = 0; j < dim; ++j) { const float d = q[j] - (float(code[j]) * scales[j] + offsets[j]); sum += d * d; }
                hit.score = -sum;
            }
            const std::size_t keep = std::min<std::size_t>(10, found.size());
            std::partial_sort(found.begin(), found.begin() + std::ptrdiff_t(keep), found.end(), [](auto& a, auto& b) { return a.score > b.score; });
            found.resize(keep); return found;
        };
        std::printf("rerank,nprobe,recall10,p50_us\n");
        for (std::uint32_t rerank : {256u, 128u, 64u}) {
            router.rerank = rerank;
            ivf.build_router(router);
            for (std::size_t nprobe : nprobes) {
                full(queries.data(), nprobe);
                std::size_t hits = 0;
                const double us = p50_us(count, [&](std::size_t i) {
                    for (const auto& hit : full(queries.data() + i * dim, nprobe))
                        hits += std::count(truth.begin() + std::ptrdiff_t(i * 10), truth.begin() + std::ptrdiff_t(i * 10 + 10), std::int32_t(hit.id));
                });
                std::printf("%u,%zu,%.4f,%.1f\n", rerank, nprobe, double(hits) / double(count * 10), us);
                std::fflush(stdout);
            }
        }
    }
    if (loop_s > 0) {
        std::fprintf(stderr, "looping\n");
        const auto end = Clock::now() + std::chrono::duration<double>(loop_s);
        for (std::size_t i = 0; Clock::now() < end; i = (i + 1) % count)
            ivf.search(queries.data() + i * dim, top, nprobes.front(), {}, routed);
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what()); return 1;
}
