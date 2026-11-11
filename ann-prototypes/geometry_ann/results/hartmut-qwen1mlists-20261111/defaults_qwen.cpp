// Default-threshold and 1-bit prefix study (#52) on Qwen/MS MARCO x 1024 (SQ8 store), through TieredQ's explicit
// calls (TieredQ 9b158c6). One run trains a prefix codec (prefix_bits, 4-bit scan cap) and an 8-bit fine-tier
// refinement on the first `use_rows` rows, then per variant builds a TieredQIVFIndex with the given list count and list
// training (Lloyd iterations, restricted assignment supers x probes, sample rows per list), wraps it in a
// TieredQNestedIVFIndex with a suffix tier, and measures held-out recall@10 and latency alone on one pinned core: with
// exact coarse ranking, and through the two-level router at the library defaults when the variant asks for it.
// The tier rows do not depend on the partition (prefix codes come from the codec), so one encoding serves every
// variant; a check compares the first rows across variants. The tier sits in RAM, or in a TieredQFileRows file on SSD
// (direct reads, queue_depth in flight) that is removed at the end. Truth for a row subset is exact squared L2 over
// the decoded SQ8 rows (computed once and cached); the full set uses the dataset's truth.
// Usage: defaults_qwen KEY=VALUE...  (see parse)
#include <tieredq/tieredq.h>
#include <tieredq/tieredq_nested.h>
#include <tieredq/tieredq_refinement.h>
#include <fcntl.h>
#include <omp.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t top_k = 10, codec_sample_rows = 32768, tier_chunk_rows = 65536, truth_chunk_rows = 256,
                      tier_check_rows = 4096;
std::uint64_t seed = 42;  // the earlier studies' codec seed and samples (seed=)

struct Variant {
    std::string name;
    std::size_t lists = 0, rows_per_list = 0;
    dv::TieredQCoarseTraining training;
    bool router = false;
};
struct Options {
    std::string codes, scales, offsets, out, queries, truth, ssd_dir;
    std::size_t rows = 0, use_rows = 0, dim = 1024, threads = 32, query_count = 750, prefix_bits = 2048, total_bits = 8,
                max_scan_bits = 4, queue_depth = 32;
    int search_cpu = 2;
    bool sweep_exact = true, builder = false, check_batch = false, round_trip = false, tier_file = false, internal_lists = false;
    std::vector<std::size_t> nprobes, divisors, candidates;
    std::vector<Variant> variants;
};
template<class T> std::vector<T> numbers(const std::string& text) {
    std::vector<T> out; std::stringstream in(text); std::string item;
    while (std::getline(in, item, ',')) if (!item.empty()) out.push_back(T(std::stod(item)));
    return out;
}
// name:lists:iterations:supers:probes:rows_per_list:router, separated by ';'.
std::vector<Variant> variants(const std::string& text) {
    std::vector<Variant> out; std::stringstream in(text); std::string spec;
    while (std::getline(in, spec, ';')) {
        if (spec.empty()) continue;
        std::stringstream parts(spec); std::vector<std::string> f; std::string field;
        while (std::getline(parts, field, ':')) f.push_back(field);
        if (f.size() != 7) throw std::invalid_argument("variant needs 7 fields: " + spec);
        Variant v; v.name = f[0]; v.lists = std::stoul(f[1]);
        v.training = {std::uint32_t(std::stoul(f[2])), std::uint32_t(std::stoul(f[3])), std::uint32_t(std::stoul(f[4]))};
        v.rows_per_list = std::stoul(f[5]); v.router = f[6] == "1";
        out.push_back(v);
    }
    return out;
}
Options parse(int argc, char** argv) {
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i]; auto eq = a.find('='); if (eq == std::string::npos) throw std::invalid_argument(a); kv[a.substr(0, eq)] = a.substr(eq + 1); }
    auto take = [&](const char* key) { auto it = kv.find(key); if (it == kv.end()) throw std::invalid_argument(std::string("missing ") + key); auto v = it->second; kv.erase(it); return v; };
    auto maybe = [&](const char* key, const std::string& fallback) { auto it = kv.find(key); if (it == kv.end()) return fallback; auto v = it->second; kv.erase(it); return v; };
    Options o;
    o.codes = take("codes"); o.scales = take("scales"); o.offsets = take("offsets"); o.out = take("out");
    o.queries = take("queries"); o.truth = take("truth");
    o.rows = std::stoul(take("rows")); o.use_rows = std::stoul(maybe("use_rows", std::to_string(o.rows)));
    o.dim = std::stoul(maybe("dim", "1024")); o.threads = std::stoul(maybe("threads", "32"));
    o.search_cpu = std::stoi(maybe("search_cpu", "2")); o.query_count = std::stoul(maybe("query_count", "750"));
    o.prefix_bits = std::stoul(maybe("prefix_bits", "2048")); o.total_bits = std::stoul(maybe("total_bits", "8"));
    o.max_scan_bits = std::stoul(maybe("max_scan_bits", "4"));
    o.nprobes = numbers<std::size_t>(maybe("nprobes", "")); o.divisors = numbers<std::size_t>(maybe("nprobe_divisors", "512,256,128,64,32,16"));
    o.candidates = numbers<std::size_t>(maybe("candidates", "20"));
    o.ssd_dir = maybe("ssd_dir", ""); o.queue_depth = std::stoul(maybe("queue_depth", "32"));
    o.sweep_exact = maybe("sweep_exact", "1") == "1";
    // builder=1: each variant through TieredQNestedIVFIndex::build (its own codec, sample and lists) at the seed.
    o.builder = maybe("builder", "0") == "1"; o.check_batch = maybe("check_batch", "0") == "1";
    // With builder=1, as Python serves: round_trip=1 deserializes the built index before measuring; tier_file=1 writes
    // the tier with write_tier() (the chunked loader) to ssd_dir and serves it from that file (direct reads).
    o.round_trip = maybe("round_trip", "0") == "1"; o.tier_file = maybe("tier_file", "0") == "1";
    // internal_lists=1: the explicit path, but build() trains the lists on its own sample instead of train_lists().
    o.internal_lists = maybe("internal_lists", "0") == "1";
    seed = std::stoull(maybe("seed", "42"));
    o.variants = variants(take("variants"));
    if (!kv.empty()) throw std::invalid_argument("unknown option " + kv.begin()->first);
    if (o.use_rows > o.rows) throw std::invalid_argument("use_rows exceeds rows");
    return o;
}

// The SQ8 store, memory-mapped: row i decodes to codes[i] * scales + offsets.
class Store {
public:
    explicit Store(const Options& o) : rows_(o.use_rows), dim_(o.dim), scales_(read<float>(o.scales)), offsets_(read<float>(o.offsets)) {
        if (scales_.size() != dim_ || offsets_.size() != dim_) throw std::runtime_error("scales/offsets must have one value per dimension");
        int fd = ::open(o.codes.c_str(), O_RDONLY); if (fd < 0) throw std::runtime_error("cannot open " + o.codes);
        bytes_ = o.rows * dim_;
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
    dv::TieredQIVFIndex::RowLoader loader() const {
        return [this](const std::uint32_t* ids, std::size_t count, float* out) { decode(ids, count, out); };
    }
    template<class T> static std::vector<T> read(const std::string& path) {
        std::ifstream in(path, std::ios::binary | std::ios::ate); if (!in) throw std::runtime_error("cannot open " + path);
        std::vector<T> v(std::size_t(in.tellg()) / sizeof(T)); in.seekg(0); in.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(T)); return v;
    }
private:
    std::size_t rows_, dim_, bytes_ = 0; std::vector<float> scales_, offsets_; const std::int8_t* codes_ = nullptr;
};

// A seeded sample of the used rows, sorted (salt 1: codec and refinement; salt 2: lists), as the earlier studies draw it.
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
void write_ivecs(const std::string& path, const std::vector<std::int32_t>& ids, std::size_t width) {
    std::ofstream out(path, std::ios::binary); const std::int32_t w = std::int32_t(width);
    for (std::size_t r = 0; r < ids.size() / width; ++r) {
        out.write(reinterpret_cast<const char*>(&w), sizeof(w));
        out.write(reinterpret_cast<const char*>(ids.data() + r * width), std::streamsize(width * sizeof(std::int32_t)));
    }
    if (!out) throw std::runtime_error("cannot write " + path);
}
// Exact top-k by squared L2 over the decoded rows, row chunks across threads, each keeping every query's best k.
std::vector<std::int32_t> exact_truth(const Store& store, const std::vector<float>& queries, std::size_t count, std::size_t dim) {
    using Best = std::vector<std::pair<float, std::int32_t>>;  // max-heap on distance
    const std::size_t chunks = (store.rows() + truth_chunk_rows - 1) / truth_chunk_rows;
    std::vector<std::vector<Best>> per_thread(omp_get_max_threads(), std::vector<Best>(count));
    #pragma omp parallel
    {
        auto& best = per_thread[omp_get_thread_num()];
        std::vector<float> rows(truth_chunk_rows * dim), norms(truth_chunk_rows); std::vector<std::uint32_t> ids(truth_chunk_rows);
        #pragma omp for schedule(dynamic)
        for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
            const std::size_t first = chunk * truth_chunk_rows, n = std::min(truth_chunk_rows, store.rows() - first);
            std::iota(ids.begin(), ids.begin() + n, std::uint32_t(first));
            for (std::size_t i = 0; i < n; ++i) store.decode_row(ids[i], rows.data() + i * dim);
            for (std::size_t i = 0; i < n; ++i) { float s = 0; for (std::size_t j = 0; j < dim; ++j) s += rows[i * dim + j] * rows[i * dim + j]; norms[i] = s; }
            for (std::size_t q = 0; q < count; ++q) {
                const float* query = queries.data() + q * dim;
                for (std::size_t i = 0; i < n; ++i) {
                    float dot = 0; const float* row = rows.data() + i * dim;
                    for (std::size_t j = 0; j < dim; ++j) dot += query[j] * row[j];
                    const float distance = norms[i] - 2 * dot;  // |q|^2 is common to every row of a query
                    auto& heap = best[q];
                    if (heap.size() < top_k) { heap.emplace_back(distance, std::int32_t(first + i)); std::push_heap(heap.begin(), heap.end()); }
                    else if (distance < heap.front().first) { std::pop_heap(heap.begin(), heap.end()); heap.back() = {distance, std::int32_t(first + i)}; std::push_heap(heap.begin(), heap.end()); }
                }
            }
        }
    }
    std::vector<std::int32_t> truth(count * top_k);
    for (std::size_t q = 0; q < count; ++q) {
        Best merged; for (auto& best : per_thread) merged.insert(merged.end(), best[q].begin(), best[q].end());
        std::sort(merged.begin(), merged.end());
        for (std::size_t i = 0; i < top_k; ++i) truth[q * top_k + i] = merged[i].second;
    }
    return truth;
}
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
double percentile(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, std::size_t(p * (v.size() - 1) + 0.5))]; }
std::size_t true_hits(const std::vector<dv::TieredQHit>& found, const std::vector<std::int32_t>& truth, std::size_t q) {
    std::size_t hits = 0;
    for (std::size_t i = 0; i < std::min(top_k, found.size()); ++i)
        hits += std::count(truth.begin() + q * top_k, truth.begin() + (q + 1) * top_k, std::int32_t(found[i].id));
    return hits;
}
std::size_t anonymous_rss() {
    std::ifstream in("/proc/self/status"); std::string key, unit; std::size_t kb = 0;
    while (in >> key) if (key == "RssAnon:") { in >> kb >> unit; return kb * 1024; }
    return 0;
}
// Every base row's tier row, encoded a chunk at a time and handed to sink(first, count, rows).
template<class Sink> void encode_tier(const dv::TieredQNestedIVFIndex& nest, const Store& store, std::size_t dim, Sink&& sink) {
    std::vector<float> decoded(tier_chunk_rows * dim); std::vector<std::uint32_t> ids(tier_chunk_rows);
    std::vector<std::uint8_t> rows(tier_chunk_rows * nest.tier_bytes_per_vector());
    for (std::size_t first = 0; first < store.rows(); first += tier_chunk_rows) {
        const std::size_t n = std::min(tier_chunk_rows, store.rows() - first);
        std::iota(ids.begin(), ids.begin() + n, std::uint32_t(first)); store.decode(ids.data(), n, decoded.data());
        nest.encode_tier(decoded.data(), first, n, rows.data());
        sink(first, n, rows.data());
    }
}
std::vector<std::uint8_t> tier_head(const dv::TieredQNestedIVFIndex& nest, const Store& store, std::size_t dim) {
    const std::size_t n = std::min(tier_check_rows, store.rows());
    std::vector<std::uint32_t> ids(n); std::iota(ids.begin(), ids.end(), 0u);
    std::vector<float> decoded(n * dim); store.decode(ids.data(), n, decoded.data());
    std::vector<std::uint8_t> rows(n * nest.tier_bytes_per_vector()); nest.encode_tier(decoded.data(), 0, n, rows.data());
    return rows;
}
}  // namespace

int main(int argc, char** argv) try {
    const Options o = parse(argc, argv);
    omp_set_num_threads(int(o.threads));
    const Store store(o);
    const auto queries = read_vecs<float>(o.queries, o.dim, o.query_count);
    std::vector<std::int32_t> truth;
    double truth_s = 0;
    if (std::filesystem::exists(o.truth)) truth = read_vecs<std::int32_t>(o.truth, top_k, o.query_count);
    else {
        if (o.use_rows == o.rows) throw std::runtime_error("missing truth " + o.truth);
        const auto t = Clock::now(); truth = exact_truth(store, queries, o.query_count, o.dim); truth_s = seconds(t);
        write_ivecs(o.truth, truth, top_k);
    }
    std::FILE* report = std::fopen((o.out + ".json").c_str(), "w"); std::FILE* csv = std::fopen((o.out + ".csv").c_str(), "w");
    if (!report || !csv) throw std::runtime_error("cannot write outputs");

    auto t = Clock::now();
    const auto codec_sample = sample_rows(store, std::min(codec_sample_rows, store.rows()), 1, o.dim);
    const std::size_t sampled = codec_sample.size() / o.dim;
    auto config = dv::TieredQConfig::target_code_bits(std::uint32_t(o.dim), std::uint32_t(o.prefix_bits));
    config.metric = dv::TieredQMetric::inner_product; config.fine_tiers = false; config.max_scan_bits = std::uint32_t(o.max_scan_bits);
    // Round-tripped as the earlier studies loaded it: a persisted model drops the training-only scan cap, which 9b158c6's
    // refinement training would otherwise inherit (capping the 8-bit finer model at 4 bits per coordinate).
    const auto codec = dv::TieredQ::deserialize(dv::TieredQ::train(config, codec_sample.data(), sampled, seed).serialize());
    const double codec_s = seconds(t); t = Clock::now();
    const auto refinement = dv::TieredQRefinement::train(codec, std::uint32_t(o.total_bits * o.dim), codec_sample.data(), sampled, seed, true);
    const double refinement_s = seconds(t);
    std::fprintf(report, "{\"rows\": %zu, \"prefix_bits\": %zu, \"realized_prefix_bits\": %u, \"total_bits_per_dim\": %zu, \"truth_s\": %.1f, "
                 "\"codec_s\": %.2f, \"refinement_s\": %.2f, \"variants\": [",
                 store.rows(), o.prefix_bits, codec.config().code_bits, o.total_bits, truth_s, codec_s, refinement_s);
    std::fflush(report);
    std::fprintf(csv, "variant,lists,iterations,supers,probes,rows_per_list,routing,tier,nprobe,candidates,recall10,p50_us,p99_us,mean_us,anon_bytes,seed,builder,batch_mismatches\n");

    // One configuration's single-query sweep, alone on the pinned core.
    auto sweep = [&](const Variant& v, const char* routing, const char* tier, const dv::TieredQNestedIVFIndex& nest) {
        const auto nprobes = o.nprobes.empty() ? [&] { std::vector<std::size_t> n; for (auto d : o.divisors) n.push_back(std::max<std::size_t>(1, v.lists / d)); return n; }() : o.nprobes;
        omp_set_num_threads(1);
        cpu_set_t all; sched_getaffinity(0, sizeof(all), &all);
        cpu_set_t only; CPU_ZERO(&only); CPU_SET(o.search_cpu, &only);
        if (sched_setaffinity(0, sizeof(only), &only)) throw std::runtime_error("cannot pin the search thread");
        for (std::size_t nprobe : nprobes) for (std::size_t candidates : o.candidates) {
            nest.search(queries.data(), top_k, nprobe, candidates);  // warm
            std::vector<double> us(o.query_count); std::size_t hits = 0, mismatches = 0;
            std::vector<float> batch_scores(top_k); std::vector<std::uint32_t> batch_ids(top_k);
            for (std::size_t q = 0; q < o.query_count; ++q) {
                const auto start = Clock::now(); const auto found = nest.search(queries.data() + q * o.dim, top_k, nprobe, candidates);
                us[q] = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
                hits += true_hits(found, truth, q);
                if (!o.check_batch) continue;  // the Python path: a one-row batch
                nest.search_batch(queries.data() + q * o.dim, 1, top_k, nprobe, candidates, batch_scores.data(), batch_ids.data());
                for (std::size_t i = 0; i < found.size(); ++i) mismatches += found[i].id != batch_ids[i];
            }
            std::fprintf(csv, "%s,%zu,%u,%u,%u,%zu,%s,%s,%zu,%zu,%.4f,%.1f,%.1f,%.1f,%zu,%llu,%d,%zu\n", v.name.c_str(), v.lists, v.training.iterations,
                         v.training.assignment_supers, v.training.assignment_probes, v.rows_per_list, routing, tier, nprobe, candidates,
                         double(hits) / double(o.query_count * top_k), percentile(us, 0.5), percentile(us, 0.99),
                         std::accumulate(us.begin(), us.end(), 0.0) / double(o.query_count), anonymous_rss(),
                         (unsigned long long)seed, int(o.builder), mismatches);
            std::fflush(csv);
        }
        if (sched_setaffinity(0, sizeof(all), &all)) throw std::runtime_error("cannot unpin");
        omp_set_num_threads(int(o.threads));
    };

    std::shared_ptr<const dv::TieredQRowSource> tier;
    std::vector<std::uint8_t> head;
    std::string tier_path, tier_kind = o.ssd_dir.empty() ? "ram" : "ssd";
    double tier_s = 0;
    bool first_variant = true;
    for (const auto& v : o.variants) {
        if (o.builder) {  // the library's own build, as Python runs it; tier rows depend on its codec
            dv::TieredQNestedConfig config;
            config.dimensions = std::uint32_t(o.dim); config.prefix_bits = std::uint32_t(o.prefix_bits);
            config.total_bits = std::uint32_t(o.total_bits * o.dim); config.lists = std::uint32_t(v.lists);
            config.coarse_training = dv::TieredQCoarseTraining{v.training.iterations, v.training.assignment_supers,
                                                               v.training.assignment_probes, std::uint32_t(v.rows_per_list)};
            config.router = v.router; config.router_config.seed = seed; config.seed = seed;
            t = Clock::now();
            auto nest = dv::TieredQNestedIVFIndex::build(config, store.loader(), store.rows());
            const double build_s = seconds(t);
            const std::size_t row_bytes = nest.tier_bytes_per_vector();
            std::vector<std::uint8_t> rows(store.rows() * row_bytes);
            encode_tier(nest, store, o.dim, [&](std::size_t first, std::size_t n, const std::uint8_t* r) { std::copy_n(r, n * row_bytes, rows.data() + first * row_bytes); });
            if (o.tier_file) {
                tier_path = (std::filesystem::path(o.ssd_dir) / "tier.rows").string();
                nest.write_tier(tier_path, store.loader());
                const dv::TieredQFileRows written(tier_path);
                std::vector<std::uint32_t> ids(store.rows()); std::iota(ids.begin(), ids.end(), 0u);
                std::vector<std::uint8_t> read(rows.size()); written.read(ids.data(), ids.size(), read.data());
                std::size_t differing = 0;
                for (std::size_t i = 0; i < store.rows(); ++i)
                    differing += !std::equal(read.begin() + i * row_bytes, read.begin() + (i + 1) * row_bytes, rows.begin() + i * row_bytes);
                std::fprintf(report, "%s{\"tier_file_rows_differing\": %zu}", first_variant ? "" : ", ", differing);
                first_variant = false;
            }
            if (o.round_trip) nest = dv::TieredQNestedIVFIndex::deserialize(nest.serialize());
            if (o.tier_file) nest.attach_tier(std::make_shared<dv::TieredQFileRows>(tier_path, dv::TieredQFileReads::direct, o.queue_depth));
            else nest.attach_tier(std::make_shared<dv::TieredQMemoryRows>(std::move(rows), row_bytes));
            sweep(v, v.router ? "router" : "exact", "ram", nest);
            std::fprintf(report, "%s{\"name\": \"%s\", \"builder\": 1, \"build_s\": %.2f}", first_variant ? "" : ", ", v.name.c_str(), build_s);
            std::fflush(report); first_variant = false;
            continue;
        }
        t = Clock::now();
        dv::TieredQIVFIndex ivf(codec, std::uint32_t(v.lists), seed);
        ivf.set_coarse_training({v.training.iterations, v.training.assignment_supers, v.training.assignment_probes,
                                 std::uint32_t(v.rows_per_list)});
        if (!o.internal_lists) {
            const auto sample = sample_rows(store, std::min(store.rows(), v.lists * v.rows_per_list), 2, o.dim);
            ivf.train_lists(sample.data(), sample.size() / o.dim);
        }
        const double train_s = seconds(t); t = Clock::now();
        ivf.build(store.loader(), store.rows());
        const double build_s = seconds(t);
        const auto stats = ivf.build_stats();
        auto nested = [&](dv::TieredQIVFIndex index) { return dv::TieredQNestedIVFIndex(std::move(index), refinement, dv::TieredQTierRows::suffix); };
        if (!tier) {  // encoded once: identical for every partition (checked below)
            const auto nest = nested(ivf);
            head = tier_head(nest, store, o.dim);
            t = Clock::now();
            const std::size_t row_bytes = nest.tier_bytes_per_vector();
            if (o.ssd_dir.empty()) {
                std::vector<std::uint8_t> rows(store.rows() * row_bytes);
                encode_tier(nest, store, o.dim, [&](std::size_t first, std::size_t n, const std::uint8_t* r) { std::copy_n(r, n * row_bytes, rows.data() + first * row_bytes); });
                tier = std::make_shared<dv::TieredQMemoryRows>(std::move(rows), row_bytes);
            } else {
                tier_path = (std::filesystem::path(o.ssd_dir) / "tier.rows").string();
                dv::TieredQRowFileWriter writer(tier_path, row_bytes);
                encode_tier(nest, store, o.dim, [&](std::size_t, std::size_t n, const std::uint8_t* r) { writer.append(r, n); });
                writer.finish();
                tier = std::make_shared<dv::TieredQFileRows>(tier_path, dv::TieredQFileReads::direct, o.queue_depth);
            }
            tier_s = seconds(t);
        } else if (tier_head(nested(ivf), store, o.dim) != head) throw std::runtime_error("tier rows differ across partitions");
        if (o.sweep_exact) {
            auto nest = nested(ivf); nest.attach_tier(tier);
            sweep(v, "exact", tier_kind.c_str(), nest);
        }
        double router_s = 0; std::size_t router_bytes = 0;
        if (v.router) {
            t = Clock::now(); dv::TieredQRouterConfig router; router.seed = seed; ivf.build_router(router); router_s = seconds(t);
            router_bytes = ivf.router_bytes();
            auto nest = nested(std::move(ivf)); nest.attach_tier(tier);
            sweep(v, "router", tier_kind.c_str(), nest);
        }
        std::fprintf(report, "%s{\"name\": \"%s\", \"lists\": %zu, \"iterations\": %u, \"supers\": %u, \"probes\": %u, \"rows_per_list\": %zu, "
                     "\"train_lists_s\": %.2f, \"build_s\": %.2f, \"assign_s\": %.2f, \"encode_s\": %.2f, \"router_s\": %.2f, \"router_bytes\": %zu, \"tier_s\": %.2f}",
                     first_variant ? "" : ", ", v.name.c_str(), v.lists, v.training.iterations, v.training.assignment_supers,
                     v.training.assignment_probes, v.rows_per_list, train_s, build_s, stats.assign_s, stats.encode_s, router_s, router_bytes,
                     first_variant ? tier_s : 0.0);
        std::fflush(report); first_variant = false;
    }
    std::fprintf(report, "]}\n");
    std::fclose(csv); std::fclose(report);
    tier.reset();
    if (!tier_path.empty()) std::filesystem::remove(tier_path);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what()); return 1;
}
