// TieredQ IVF with the two-level coarse router (#47) on Qwen/MS MARCO 8.84M x 1024 (SQ8 store): a 2-bit
// TieredQIVFIndex on given Lloyd lists (set_lists), or with lists trained in the library at Faiss's budget (#48:
// rows_per_list sample rows per list, train_iterations Lloyd iterations, restricted assignment with
// assignment_supers / assignment_probes; build time then counts sampling and training), measured with an exact SQ8 re-rank of its top k', once with the
// exact coarse ranking and once per router expansion (probed supers = enough for expansion member centroids per
// probed list, at least min_super_probes). The built index is saved (index=) before the router is added, so later runs
// load it instead of rebuilding. Single-query latency runs alone on one pinned core; the batch pass (#51) runs all
// queries of a set through search_batch on batch_threads threads, with the exact GEMM ranking and with the router.
// Recall@10 is against the dataset's truth; inner product on the (normalized) inputs, the re-rank uses L2 on
// SQ8-decoded rows as Faiss IVF-SQ8 scores.
// Usage: tieredq_ivf_qwen KEY=VALUE...  (see Options)
#include <tieredq/tieredq.h>
#include <fcntl.h>
#include <omp.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Routing = dv::TieredQRouting;
constexpr std::size_t top_k = 10, codec_sample_rows = 32768;
constexpr std::uint64_t seed = 42;

struct Options {
    std::string codes, scales, offsets, out, centroids, index;
    std::vector<std::string> query_sets;  // name:queries.fvecs:truth.ivecs:count
    std::size_t rows = 0, dim = 1024, lists = 65536, threads = 32, batch_threads = 0, rows_per_list = 40;
    dv::TieredQCoarseTraining training;
    int search_cpu = 2;
    unsigned code_bits = 2048, max_scan_bits = 4;
    std::vector<std::size_t> nprobes, candidates, batch_nprobes;
    bool exact = true;  // also measure the exact coarse ranking
    dv::TieredQRouterConfig router;
    std::vector<double> expansions;  // empty: no router
};
template<class T> std::vector<T> numbers(const std::string& text) {
    std::vector<T> out; std::stringstream in(text); std::string item;
    while (std::getline(in, item, ',')) if (!item.empty()) out.push_back(T(std::stod(item)));
    return out;
}
Options parse(int argc, char** argv) {
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i]; auto eq = a.find('='); if (eq == std::string::npos) throw std::invalid_argument(a); kv[a.substr(0, eq)] = a.substr(eq + 1); }
    auto take = [&](const char* key) { auto it = kv.find(key); if (it == kv.end()) throw std::invalid_argument(std::string("missing ") + key); auto v = it->second; kv.erase(it); return v; };
    auto maybe = [&](const char* key, const std::string& fallback) { auto it = kv.find(key); if (it == kv.end()) return fallback; auto v = it->second; kv.erase(it); return v; };
    Options o;
    o.codes = take("codes"); o.scales = take("scales"); o.offsets = take("offsets"); o.out = take("out");
    o.centroids = maybe("centroids", ""); o.index = maybe("index", "");
    o.rows_per_list = std::stoul(maybe("rows_per_list", "40"));
    o.training.iterations = std::uint32_t(std::stoul(maybe("train_iterations", "4")));
    o.training.assignment_supers = std::uint32_t(std::stoul(maybe("assignment_supers", "0")));
    o.training.assignment_probes = std::uint32_t(std::stoul(maybe("assignment_probes", "16")));
    o.rows = std::stoul(take("rows")); o.dim = std::stoul(maybe("dim", "1024")); o.lists = std::stoul(take("lists"));
    o.threads = std::stoul(maybe("threads", "32")); o.batch_threads = std::stoul(maybe("batch_threads", "0"));
    o.search_cpu = std::stoi(maybe("search_cpu", "2")); o.code_bits = unsigned(std::stoul(maybe("code_bits", "2048")));
    o.max_scan_bits = unsigned(std::stoul(maybe("max_scan_bits", "4")));
    o.nprobes = numbers<std::size_t>(take("nprobes")); o.candidates = numbers<std::size_t>(take("candidates"));
    o.batch_nprobes = numbers<std::size_t>(maybe("batch_nprobes", ""));
    o.exact = maybe("exact", "1") == "1";
    o.expansions = numbers<double>(maybe("expansions", ""));
    o.router.lists_per_super = std::uint32_t(std::stoul(maybe("lists_per_super", "128")));
    o.router.rerank = std::uint32_t(std::stoul(maybe("rerank", "256")));
    o.router.bits_per_dimension = std::stod(maybe("router_bits", "2"));
    o.router.min_super_probes = std::uint32_t(std::stoul(maybe("min_super_probes", "64")));
    o.router.seed = seed;
    std::stringstream sets(take("queries")); std::string set; while (std::getline(sets, set, ';')) o.query_sets.push_back(set);
    if (!kv.empty()) throw std::invalid_argument("unknown option " + kv.begin()->first);
    return o;
}

// The SQ8 store, memory-mapped: row i decodes to codes[i] * scales + offsets.
class Store {
public:
    Store(const Options& o) : rows_(o.rows), dim_(o.dim), scales_(read<float>(o.scales)), offsets_(read<float>(o.offsets)) {
        if (scales_.size() != dim_ || offsets_.size() != dim_) throw std::runtime_error("scales/offsets must have one value per dimension");
        int fd = ::open(o.codes.c_str(), O_RDONLY); if (fd < 0) throw std::runtime_error("cannot open " + o.codes);
        bytes_ = rows_ * dim_;
        void* map = ::mmap(nullptr, bytes_, PROT_READ, MAP_SHARED, fd, 0); ::close(fd);
        if (map == MAP_FAILED) throw std::runtime_error("cannot map " + o.codes);
        codes_ = static_cast<const std::int8_t*>(map);
    }
    ~Store() { ::munmap(const_cast<std::int8_t*>(codes_), bytes_); }
    std::size_t rows() const { return rows_; }
    void decode(const std::uint32_t* ids, std::size_t count, float* out) const {
        #pragma omp parallel for schedule(static) if (count >= 1024)
        for (std::size_t i = 0; i < count; ++i) decode_row(ids[i], out + i * dim_);
    }
    void decode_row(std::size_t id, float* out) const {
        const std::int8_t* code = codes_ + id * dim_;
        for (std::size_t j = 0; j < dim_; ++j) out[j] = float(code[j]) * scales_[j] + offsets_[j];
    }
    float l2(const float* query, std::size_t id) const {
        const std::int8_t* code = codes_ + id * dim_; float sum = 0;
        for (std::size_t j = 0; j < dim_; ++j) { float d = query[j] - (float(code[j]) * scales_[j] + offsets_[j]); sum += d * d; }
        return sum;
    }
    template<class T> static std::vector<T> read(const std::string& path) {
        std::ifstream in(path, std::ios::binary | std::ios::ate); if (!in) throw std::runtime_error("cannot open " + path);
        std::vector<T> v(std::size_t(in.tellg()) / sizeof(T)); in.seekg(0); in.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(T)); return v;
    }
private:
    std::size_t rows_, dim_, bytes_ = 0; std::vector<float> scales_, offsets_; const std::int8_t* codes_ = nullptr;
};

std::vector<float> sample_rows(const Store& store, std::size_t count, std::uint64_t salt, std::size_t dim) {
    std::mt19937_64 rng(seed + salt); std::vector<std::uint32_t> ids(store.rows()); std::iota(ids.begin(), ids.end(), 0u);
    for (std::size_t i = 0; i < count; ++i) std::swap(ids[i], ids[i + rng() % (ids.size() - i)]);
    ids.resize(count); std::sort(ids.begin(), ids.end());
    std::vector<float> out(count * dim); store.decode(ids.data(), count, out.data()); return out;
}
template<class T> std::vector<T> read_vecs(const std::string& path, std::size_t keep, std::size_t rows) {
    auto raw = Store::read<std::int32_t>(path); std::vector<T> out;
    for (std::size_t i = 0, r = 0; i < raw.size() && r < rows; i += std::size_t(raw[i]) + 1, ++r) {
        if (std::size_t(raw[i]) < keep) throw std::runtime_error("rows too narrow in " + path);
        const T* row = reinterpret_cast<const T*>(raw.data() + i + 1); out.insert(out.end(), row, row + keep);
    }
    return out;
}
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
std::size_t peak_rss_bytes() { rusage usage{}; getrusage(RUSAGE_SELF, &usage); return std::size_t(usage.ru_maxrss) * 1024; }
double percentile(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, std::size_t(p * (v.size() - 1) + 0.5))]; }

// The library's automatic rule with the sweep's expansion: supers holding expansion member centroids per probed list.
std::uint32_t super_probes(const Options& o, std::size_t routed_lists, std::size_t nprobe, double expansion) {
    const std::size_t supers = std::max<std::size_t>(1, routed_lists / o.router.lists_per_super);
    const double per_super = double(routed_lists) / double(supers);
    const auto wanted = std::size_t(std::ceil(double(nprobe) * expansion / per_super));
    return std::uint32_t(std::clamp<std::size_t>(std::max<std::size_t>(wanted, o.router.min_super_probes), 1, supers));
}

// Exact SQ8 re-rank of one query's candidates down to its top_k.
void rerank(const Store& store, const float* query, std::vector<dv::TieredQHit>& found) {
    for (auto& hit : found) hit.score = -store.l2(query, hit.id);
    const std::size_t keep = std::min(top_k, found.size());
    std::partial_sort(found.begin(), found.begin() + keep, found.end(), [](auto& a, auto& b) { return a.score > b.score; });
    found.resize(keep);
}
std::size_t true_hits(const std::vector<dv::TieredQHit>& found, const std::vector<std::int32_t>& truth, std::size_t q) {
    std::size_t hits = 0;
    for (std::size_t i = 0; i < std::min(top_k, found.size()); ++i)
        hits += std::count(truth.begin() + q * top_k, truth.begin() + (q + 1) * top_k, std::int32_t(found[i].id));
    return hits;
}
struct Mode { std::string name; Routing routing; double expansion; };
}  // namespace

int main(int argc, char** argv) try {
    const Options o = parse(argc, argv);
    omp_set_num_threads(int(o.threads));
    cpu_set_t all; sched_getaffinity(0, sizeof(all), &all);
    Store store(o);
    std::FILE* report = std::fopen((o.out + ".json").c_str(), "w"); std::FILE* rows_out = std::fopen((o.out + ".csv").c_str(), "w");
    std::FILE* batch_out = std::fopen((o.out + "-batch.csv").c_str(), "w");
    if (!report || !rows_out || !batch_out) throw std::runtime_error("cannot write outputs");

    double codec_s = 0, lists_s = 0, build_s = 0, load_s = 0;
    dv::TieredQBuildStats phases;
    auto start = Clock::now();
    auto build = [&] {
        auto codec_sample = sample_rows(store, std::min(codec_sample_rows, o.rows), 1, o.dim);
        auto config = dv::TieredQConfig::target_code_bits(std::uint32_t(o.dim), o.code_bits);
        config.metric = dv::TieredQMetric::inner_product; config.max_scan_bits = o.max_scan_bits;
        auto codec = dv::TieredQ::train(config, codec_sample.data(), codec_sample.size() / o.dim, seed);
        codec_s = seconds(start); start = Clock::now();
        dv::TieredQIVFIndex built(codec, std::uint32_t(o.lists), seed);
        built.set_coarse_training(o.training);
        if (!o.centroids.empty()) {
            auto given = Store::read<float>(o.centroids);
            if (given.size() != o.lists * o.dim) throw std::runtime_error("centroids must be lists x dim float32");
            built.set_lists(given.data());
        } else {
            // Faiss's recipe: rows_per_list decoded sample rows per list, trained before the build.
            auto sample = sample_rows(store, std::min(o.rows, o.lists * o.rows_per_list), 2, o.dim);
            built.train_lists(sample.data(), sample.size() / o.dim);
        }
        lists_s = seconds(start); start = Clock::now();
        built.build([&](const std::uint32_t* ids, std::size_t count, float* out) { store.decode(ids, count, out); }, o.rows);
        build_s = seconds(start);
        phases = built.build_stats();
        return built;
    };
    const bool cached = !o.index.empty() && std::filesystem::exists(o.index);
    dv::TieredQIVFIndex ivf = cached ? dv::TieredQIVFIndex::load(o.index) : build();
    if (cached) load_s = seconds(start);
    else if (!o.index.empty()) ivf.save(o.index);
    if (ivf.size() != o.rows || ivf.lists() != o.lists) throw std::runtime_error("index shape does not match the options");
    std::size_t routed = 0, largest = 0;
    for (std::size_t l = 0; l < ivf.lists(); ++l) { routed += ivf.list_size(l) > 0; largest = std::max<std::size_t>(largest, ivf.list_size(l)); }
    const std::size_t index_bytes = ivf.storage_bytes();

    double router_s = 0;
    if (!o.expansions.empty()) { start = Clock::now(); ivf.build_router(o.router); router_s = seconds(start); }
    std::fprintf(report, "{\"lists\": %zu, \"routed_lists\": %zu, \"largest_list\": %zu, \"code_bits\": %u, \"max_scan_bits\": %u, "
                 "\"cached_index\": %s, \"codec_s\": %.2f, \"lists_s\": %.2f, \"build_s\": %.2f, \"assign_s\": %.2f, "
                 "\"encode_s\": %.2f, \"total_build_s\": %.2f, \"train_iterations\": %u, \"assignment_supers\": %u, "
                 "\"assignment_probes\": %u, \"rows_per_list\": %zu, \"load_s\": %.2f, \"router_s\": %.2f, "
                 "\"index_bytes_without_router\": %zu, \"router_bytes\": %zu, \"lists_per_super\": %u, \"rerank\": %u, "
                 "\"router_bits\": %.2f, \"min_super_probes\": %u, \"build_peak_rss\": %zu}\n",
                 o.lists, routed, largest, o.code_bits, o.max_scan_bits, cached ? "true" : "false", codec_s, lists_s, build_s,
                 phases.assign_s, phases.encode_s, codec_s + lists_s + build_s + router_s, o.training.iterations,
                 o.training.assignment_supers, o.training.assignment_probes, o.centroids.empty() ? o.rows_per_list : 0, load_s,
                 router_s, index_bytes, ivf.router_bytes(), o.router.lists_per_super, o.router.rerank, o.router.bits_per_dimension,
                 o.router.min_super_probes, peak_rss_bytes());
    std::fflush(report);
    std::fprintf(stderr, "ready: build %.0fs, load %.0fs, router %.0fs; index %.2f GB, router %.1f MB\n", build_s, load_s, router_s,
                 index_bytes / 1e9, ivf.router_bytes() / 1e6);

    std::vector<Mode> modes;
    if (o.exact) modes.push_back({"exact", {Routing::Mode::exact, 0}, 0});
    for (double e : o.expansions) modes.push_back({"router", {Routing::Mode::router, 0}, e});
    auto with_probes = [&](Mode mode, std::size_t nprobe) {
        if (mode.routing.mode == Routing::Mode::router) mode.routing.super_probes = super_probes(o, routed, nprobe, mode.expansion);
        return mode.routing;
    };

    std::fprintf(rows_out, "queries,mode,expansion,super_probes,lists,nprobe,candidates,recall10,p50_us,p99_us,mean_us\n");
    std::fprintf(batch_out, "queries,mode,expansion,lists,nprobe,candidates,threads,recall10,qps\n");
    for (const auto& set : o.query_sets) {
        std::stringstream parts(set); std::string name, qpath, tpath, count;
        std::getline(parts, name, ':'); std::getline(parts, qpath, ':'); std::getline(parts, tpath, ':'); std::getline(parts, count, ':');
        const std::size_t nq = std::stoul(count);
        auto queries = read_vecs<float>(qpath, o.dim, nq); auto truth = read_vecs<std::int32_t>(tpath, top_k, nq);

        // Single queries, one pinned core.
        omp_set_num_threads(1);
        cpu_set_t only; CPU_ZERO(&only); CPU_SET(o.search_cpu, &only);
        if (sched_setaffinity(0, sizeof(only), &only)) throw std::runtime_error("cannot pin the search thread");
        for (std::size_t nprobe : o.nprobes) for (std::size_t depth : o.candidates) for (const auto& mode : modes) {
            const Routing routing = with_probes(mode, nprobe);
            auto search = [&](const float* q) { auto found = ivf.search(q, depth, nprobe, {}, routing); rerank(store, q, found); return found; };
            search(queries.data());  // warm
            std::vector<double> us(nq); std::size_t hits = 0;
            for (std::size_t q = 0; q < nq; ++q) {
                auto t = Clock::now(); auto found = search(queries.data() + q * o.dim);
                us[q] = std::chrono::duration<double, std::micro>(Clock::now() - t).count();
                hits += true_hits(found, truth, q);
            }
            std::fprintf(rows_out, "%s,%s,%.1f,%u,%zu,%zu,%zu,%.4f,%.1f,%.1f,%.1f\n", name.c_str(), mode.name.c_str(), mode.expansion,
                         routing.super_probes, o.lists, nprobe, depth, double(hits) / double(nq * top_k), percentile(us, 0.5),
                         percentile(us, 0.99), std::accumulate(us.begin(), us.end(), 0.0) / nq);
            std::fflush(rows_out);
        }

        // Batch pass (#51): every query of the set in one search_batch call plus a parallel re-rank.
        if (!o.batch_threads) continue;
        if (sched_setaffinity(0, sizeof(all), &all)) throw std::runtime_error("cannot unpin");
        omp_set_num_threads(int(o.batch_threads));
        for (std::size_t nprobe : o.batch_nprobes) for (std::size_t depth : o.candidates) for (const auto& mode : modes) {
            const Routing routing = with_probes(mode, nprobe);
            std::vector<float> scores(nq * depth); std::vector<std::uint32_t> ids(nq * depth);
            std::size_t hits = 0;
            auto pass = [&] {
                ivf.search_batch(queries.data(), nq, depth, nprobe, scores.data(), ids.data(), {}, routing);
                std::size_t found_hits = 0;
                #pragma omp parallel for schedule(dynamic) reduction(+ : found_hits)
                for (std::size_t q = 0; q < nq; ++q) {
                    std::vector<dv::TieredQHit> found;
                    for (std::size_t i = 0; i < depth && ids[q * depth + i] != UINT32_MAX; ++i) found.push_back({scores[q * depth + i], ids[q * depth + i]});
                    rerank(store, queries.data() + q * o.dim, found);
                    found_hits += true_hits(found, truth, q);
                }
                return found_hits;
            };
            pass();  // warm
            auto t = Clock::now(); hits = pass(); const double elapsed = seconds(t);
            std::fprintf(batch_out, "%s,%s,%.1f,%zu,%zu,%zu,%zu,%.4f,%.1f\n", name.c_str(), mode.name.c_str(), mode.expansion, o.lists,
                         nprobe, depth, o.batch_threads, double(hits) / double(nq * top_k), double(nq) / elapsed);
            std::fflush(batch_out);
        }
    }
    std::fclose(rows_out); std::fclose(batch_out); std::fclose(report);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what()); return 1;
}
