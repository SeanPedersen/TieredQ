// TieredQ's own IVF on Qwen/MS MARCO 8.84M x 1024 (SQ8 store), as shipped: TieredQIVFIndex (plain, or with an exact
// SQ8 re-rank of its top k') and TieredQNestedIVFIndex (prefix IVF + in-RAM refinement tier). Rows are decoded from
// the int8 store through the streaming build, so no fp32 copy of the base exists. Coarse lists are either given
// (centroids=: set_lists with the Lloyd centroids the Faiss and geometry_ann runs share) or trained with train_lists on
// a seeded sample of rows_per_list rows per list. With prefix_bits set, one build serves three modes: the prefix IVF
// alone, the prefix IVF with an exact SQ8 re-rank, and the nested index (refinement suffix to code_bits). Inner
// product on the (normalized) inputs; the exact re-rank uses L2 on SQ8-decoded rows, as Faiss IVF-SQ8 scores.
// Every query runs alone on one pinned core; recall@10 is against the dataset's truth.
// Usage: tieredq_ivf_qwen KEY=VALUE...  (see Options)
#include <tieredq/tieredq.h>
#include <tieredq/tieredq_nested.h>
#include <tieredq/tieredq_refinement.h>
#include <fcntl.h>
#include <omp.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
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
constexpr std::size_t top_k = 10, codec_sample_rows = 32768;
constexpr std::uint64_t seed = 42;

struct Options {
    std::string codes, scales, offsets, out, centroids;
    std::vector<std::string> query_sets;  // name:queries.fvecs:truth.ivecs:count
    std::size_t rows = 0, dim = 1024, lists = 65536, rows_per_list = 40, threads = 32;
    int search_cpu = 2;
    unsigned code_bits = 4096, prefix_bits = 0, max_scan_bits = 4;  // prefix_bits > 0: nested, code_bits is the total
    std::vector<std::size_t> nprobes, candidates;
};
std::vector<std::size_t> numbers(const std::string& text) {
    std::vector<std::size_t> out; std::stringstream in(text); std::string item;
    while (std::getline(in, item, ',')) out.push_back(std::stoul(item));
    return out;
}
Options parse(int argc, char** argv) {
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i]; auto eq = a.find('='); if (eq == std::string::npos) throw std::invalid_argument(a); kv[a.substr(0, eq)] = a.substr(eq + 1); }
    auto take = [&](const char* key) { auto it = kv.find(key); if (it == kv.end()) throw std::invalid_argument(std::string("missing ") + key); auto v = it->second; kv.erase(it); return v; };
    auto maybe = [&](const char* key, const std::string& fallback) { auto it = kv.find(key); if (it == kv.end()) return fallback; auto v = it->second; kv.erase(it); return v; };
    Options o;
    o.codes = take("codes"); o.scales = take("scales"); o.offsets = take("offsets"); o.out = take("out"); o.centroids = maybe("centroids", "");
    o.rows = std::stoul(take("rows")); o.dim = std::stoul(maybe("dim", "1024")); o.lists = std::stoul(take("lists"));
    o.rows_per_list = std::stoul(maybe("rows_per_list", "40")); o.threads = std::stoul(maybe("threads", "32"));
    o.search_cpu = std::stoi(maybe("search_cpu", "2")); o.code_bits = unsigned(std::stoul(take("code_bits")));
    o.prefix_bits = unsigned(std::stoul(maybe("prefix_bits", "0"))); o.max_scan_bits = unsigned(std::stoul(maybe("max_scan_bits", "4")));
    o.nprobes = numbers(take("nprobes")); o.candidates = numbers(maybe("candidates", ""));
    std::stringstream sets(take("queries")); std::string set; while (std::getline(sets, set, ';')) o.query_sets.push_back(set);
    if (!kv.empty()) throw std::invalid_argument("unknown option " + kv.begin()->first);
    if (o.prefix_bits && o.candidates.empty()) throw std::invalid_argument("nested search needs candidates");
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
std::size_t current_rss_bytes() {
    std::ifstream in("/proc/self/statm"); std::size_t pages = 0, resident = 0; in >> pages >> resident; return resident * std::size_t(sysconf(_SC_PAGESIZE));
}
double percentile(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, std::size_t(p * (v.size() - 1) + 0.5))]; }
}  // namespace

int main(int argc, char** argv) try {
    const Options o = parse(argc, argv);
    omp_set_num_threads(int(o.threads));
    Store store(o);
    std::FILE* report = std::fopen((o.out + ".json").c_str(), "w"); std::FILE* rows_out = std::fopen((o.out + ".csv").c_str(), "w");
    if (!report || !rows_out) throw std::runtime_error("cannot write outputs");
    const bool nested = o.prefix_bits > 0;
    const unsigned scan_bits = nested ? o.prefix_bits : o.code_bits;

    auto start = Clock::now();
    auto codec_sample = sample_rows(store, codec_sample_rows, 1, o.dim);
    auto config = dv::TieredQConfig::target_code_bits(std::uint32_t(o.dim), scan_bits);
    config.metric = dv::TieredQMetric::inner_product; config.max_scan_bits = o.max_scan_bits;
    auto codec = dv::TieredQ::train(config, codec_sample.data(), codec_sample_rows, seed);
    const double codec_s = seconds(start);

    start = Clock::now();
    dv::TieredQIVFIndex ivf(codec, std::uint32_t(o.lists), seed);
    if (!o.centroids.empty()) {
        auto given = Store::read<float>(o.centroids);
        if (given.size() != o.lists * o.dim) throw std::runtime_error("centroids must be lists x dim float32");
        ivf.set_lists(given.data());
    } else {
        auto list_sample = sample_rows(store, std::min(o.rows, o.lists * o.rows_per_list), 2, o.dim);
        ivf.train_lists(list_sample.data(), list_sample.size() / o.dim);
    }
    const double lists_s = seconds(start);
    start = Clock::now();
    ivf.build([&](const std::uint32_t* ids, std::size_t count, float* out) { store.decode(ids, count, out); }, o.rows);
    const double build_s = seconds(start);
    std::size_t largest = 0, empty = 0;
    for (std::size_t l = 0; l < ivf.lists(); ++l) { largest = std::max<std::size_t>(largest, ivf.list_size(l)); empty += ivf.list_size(l) == 0; }

    std::unique_ptr<dv::TieredQNestedIVFIndex> nest; double tier_s = 0, refinement_s = 0; std::size_t tier_bytes = 0;
    if (nested) {
        start = Clock::now();
        auto refinement = dv::TieredQRefinement::train(ivf.codec(), o.code_bits, codec_sample.data(), codec_sample_rows, seed);
        refinement_s = seconds(start); start = Clock::now();
        nest = std::make_unique<dv::TieredQNestedIVFIndex>(std::move(ivf), std::move(refinement), dv::TieredQTierRows::suffix);
        const std::size_t row_bytes = nest->tier_bytes_per_vector(), chunk = 65536;
        std::vector<std::uint8_t> tier(o.rows * row_bytes); std::vector<float> decoded(chunk * o.dim); std::vector<std::uint32_t> ids(chunk);
        for (std::size_t first = 0; first < o.rows; first += chunk) {
            const std::size_t n = std::min(chunk, o.rows - first);
            std::iota(ids.begin(), ids.begin() + n, std::uint32_t(first)); store.decode(ids.data(), n, decoded.data());
            nest->encode_tier(decoded.data(), first, n, tier.data() + first * row_bytes);
        }
        tier_bytes = tier.size();
        nest->attach_tier(std::make_shared<dv::TieredQMemoryRows>(std::move(tier), row_bytes));
        tier_s = seconds(start);
    }
    const dv::TieredQIVFIndex& index = nested ? nest->index() : ivf;
    codec_sample.clear(); codec_sample.shrink_to_fit();
    const std::size_t index_bytes = index.storage_bytes() + index.codec().memory_bytes() + (nested ? tier_bytes + nest->refinement().serialize().size() : 0);
    std::fprintf(report, "{\"lists\": %zu, \"rows_per_list\": %zu, \"code_bits\": %u, \"prefix_bits\": %u, \"max_scan_bits\": %u, "
                 "\"codec_s\": %.2f, \"lists_s\": %.2f, \"build_s\": %.2f, \"refinement_s\": %.2f, \"tier_s\": %.2f, "
                 "\"index_bytes\": %zu, \"ivf_storage_bytes\": %zu, \"tier_bytes\": %zu, \"largest_list\": %zu, \"empty_lists\": %zu, "
                 "\"build_peak_rss\": %zu, \"serving_rss_with_page_cache_excluded\": %zu}\n",
                 o.lists, o.rows_per_list, o.code_bits, o.prefix_bits, o.max_scan_bits, codec_s, lists_s, build_s, refinement_s, tier_s,
                 index_bytes, index.storage_bytes(), tier_bytes, largest, empty, peak_rss_bytes(), current_rss_bytes());
    std::fflush(report);
    std::fprintf(stderr, "built: codec %.0fs, lists %.0fs, build %.0fs, tier %.0fs; index %.2f GB\n", codec_s, lists_s, build_s, tier_s, index_bytes / 1e9);

    // Timed search: one pinned core, one query at a time.
    omp_set_num_threads(1);
    cpu_set_t only; CPU_ZERO(&only); CPU_SET(o.search_cpu, &only);
    if (sched_setaffinity(0, sizeof(only), &only)) throw std::runtime_error("cannot pin the search thread");
    std::fprintf(rows_out, "queries,mode,lists,code_bits,prefix_bits,nprobe,candidates,recall10,p50_us,p99_us,mean_us\n");
    std::vector<float> decoded(o.dim);
    for (const auto& set : o.query_sets) {
        std::stringstream parts(set); std::string name, qpath, tpath, count;
        std::getline(parts, name, ':'); std::getline(parts, qpath, ':'); std::getline(parts, tpath, ':'); std::getline(parts, count, ':');
        const std::size_t nq = std::stoul(count);
        auto queries = read_vecs<float>(qpath, o.dim, nq); auto truth = read_vecs<std::int32_t>(tpath, top_k, nq);
        auto measure = [&](const char* mode, std::size_t nprobe, std::size_t depth, auto&& search) {
            std::vector<double> us(nq); std::size_t hits = 0;
            search(queries.data(), std::size_t(0));  // warm
            for (std::size_t q = 0; q < nq; ++q) {
                auto t = Clock::now(); auto found = search(queries.data() + q * o.dim, q);
                us[q] = std::chrono::duration<double, std::micro>(Clock::now() - t).count();
                for (std::size_t i = 0; i < std::min(top_k, found.size()); ++i)
                    hits += std::count(truth.begin() + q * top_k, truth.begin() + (q + 1) * top_k, std::int32_t(found[i].id));
            }
            std::fprintf(rows_out, "%s,%s,%zu,%u,%u,%zu,%zu,%.4f,%.1f,%.1f,%.1f\n", name.c_str(), mode, o.lists, o.code_bits, o.prefix_bits, nprobe, depth,
                         double(hits) / double(nq * top_k), percentile(us, 0.5), percentile(us, 0.99), std::accumulate(us.begin(), us.end(), 0.0) / nq);
            std::fflush(rows_out);
        };
        for (std::size_t nprobe : o.nprobes) {
            if (nested)
                for (std::size_t depth : o.candidates)
                    measure("nested", nprobe, depth, [&](const float* q, std::size_t) { return nest->search(q, top_k, nprobe, depth); });
            measure("ivf", nprobe, 0, [&](const float* q, std::size_t) { return index.search(q, top_k, nprobe); });
            for (std::size_t depth : o.candidates)
                measure("ivf+sq8", nprobe, depth, [&](const float* q, std::size_t) {
                    auto found = index.search(q, depth, nprobe);
                    for (auto& hit : found) hit.score = -store.l2(q, hit.id);
                    std::partial_sort(found.begin(), found.begin() + std::min(top_k, found.size()), found.end(), [](auto& a, auto& b) { return a.score > b.score; });
                    found.resize(std::min(top_k, found.size())); return found;
                });
        }
    }
    std::fclose(rows_out); std::fclose(report);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what()); return 1;
}
