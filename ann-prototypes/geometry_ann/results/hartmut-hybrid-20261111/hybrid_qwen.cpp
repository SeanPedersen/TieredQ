// Nested TieredQ refinement tiers against the SQ8 re-rank (#49) on Qwen/MS MARCO 8.84M x 1024 (SQ8 store). Loads the
// 2-bit TieredQIVFIndex saved by the router study (same codec seed and sample), adds the two-level router, and measures
// per nprobe and candidate depth: the exact SQ8 re-rank (the baseline, 1,024 B per vector) and TieredQNestedIVFIndex
// with a suffix tier refining to each total bits/dim, with and without fine suffix tiers, held in RAM. Then one tier on
// SSD (TieredQFileRows; page-cached and direct reads, queue depths). Every query runs alone on one pinned core;
// recall@10 is against the dataset's truth. Serving memory is read from /proc/self/status after each configuration's
// sweep (anonymous: the index, router and in-RAM tier; file-backed: the memory-mapped SQ8 store's touched pages, which
// only the SQ8 re-rank needs while serving), as compare_faiss reads it for Faiss.
// hybrid_qwen (#49): with sq8_rows=PATH, each in-RAM tier is also measured as the first of two re-rank stages: the tier
// rescores the top `candidates` of the 2-bit scan, then its best `finals` get their exact SQ8 rows read from SSD (a
// TieredQFileRows copy of the store, direct I/O at final_depth reads in flight) for the final top 10. The SQ8 store then
// stays on SSD; serving RAM is the IVF, router and tier.
// Usage: hybrid_qwen KEY=VALUE...  (see Options)
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
constexpr std::size_t top_k = 10, codec_sample_rows = 32768, tier_chunk_rows = 65536;
constexpr std::uint64_t seed = 42;  // the router study's codec seed and sample

struct Options {
    std::string codes, scales, offsets, out, index, ssd_dir;
    std::vector<std::string> query_sets;
    std::size_t rows = 0, dim = 1024, threads = 32;
    int search_cpu = 2;
    std::vector<std::size_t> nprobes, candidates, totals, fine, ssd_nprobes, queue_depths;
    std::size_t ssd_total = 0, ssd_fine = 0, ssd_candidates = 100;
    std::string sq8_rows; std::vector<std::size_t> finals; std::size_t final_depth = 32, hybrid_candidates = 40;
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
    o.codes = take("codes"); o.scales = take("scales"); o.offsets = take("offsets"); o.out = take("out"); o.index = take("index");
    o.rows = std::stoul(take("rows")); o.dim = std::stoul(maybe("dim", "1024")); o.threads = std::stoul(maybe("threads", "32"));
    o.search_cpu = std::stoi(maybe("search_cpu", "2"));
    o.nprobes = numbers<std::size_t>(take("nprobes")); o.candidates = numbers<std::size_t>(take("candidates"));
    o.totals = numbers<std::size_t>(take("totals")); o.fine = numbers<std::size_t>(maybe("fine", "0"));
    o.ssd_dir = maybe("ssd_dir", ""); o.ssd_total = std::stoul(maybe("ssd_total", "0")); o.ssd_fine = std::stoul(maybe("ssd_fine", "0"));
    o.ssd_candidates = std::stoul(maybe("ssd_candidates", "100"));
    o.ssd_nprobes = numbers<std::size_t>(maybe("ssd_nprobes", "")); o.queue_depths = numbers<std::size_t>(maybe("queue_depths", "1"));
    o.sq8_rows = maybe("sq8_rows", ""); o.finals = numbers<std::size_t>(maybe("finals", "12,16,20,30"));
    o.final_depth = std::stoul(maybe("final_depth", "32")); o.hybrid_candidates = std::stoul(maybe("hybrid_candidates", "40"));
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
    const std::int8_t* row(std::size_t id) const { return codes_ + id * dim_; }
    float l2(const float* query, const std::int8_t* code) const {
        float sum = 0;
        for (std::size_t j = 0; j < dim_; ++j) { float d = query[j] - (float(code[j]) * scales_[j] + offsets_[j]); sum += d * d; }
        return sum;
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

// Identical to the router study's codec sample (salt 1), so the refinement trains on the prefix's own sample.
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
double percentile(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, std::size_t(p * (v.size() - 1) + 0.5))]; }
std::size_t true_hits(const std::vector<dv::TieredQHit>& found, const std::vector<std::int32_t>& truth, std::size_t q) {
    std::size_t hits = 0;
    for (std::size_t i = 0; i < std::min(top_k, found.size()); ++i)
        hits += std::count(truth.begin() + q * top_k, truth.begin() + (q + 1) * top_k, std::int32_t(found[i].id));
    return hits;
}
struct Memory { std::size_t rss = 0, anon = 0, file = 0; };
Memory serving_memory() {
    std::ifstream in("/proc/self/status"); std::string key; std::size_t kb = 0; Memory m; std::string unit;
    while (in >> key) {
        if (key == "VmRSS:") { in >> kb >> unit; m.rss = kb * 1024; }
        else if (key == "RssAnon:") { in >> kb >> unit; m.anon = kb * 1024; }
        else if (key == "RssFile:") { in >> kb >> unit; m.file = kb * 1024; }
    }
    return m;
}
struct QuerySet { std::string name; std::size_t count; std::vector<float> queries; std::vector<std::int32_t> truth; };

// Tier rows of every base row, encoded a chunk at a time from the SQ8 store and handed to sink(first, count, rows).
template<class Sink> void encode_tier(const dv::TieredQNestedIVFIndex& nest, const Store& store, std::size_t dim, Sink&& sink) {
    const std::size_t row_bytes = nest.tier_bytes_per_vector();
    std::vector<float> decoded(tier_chunk_rows * dim); std::vector<std::uint32_t> ids(tier_chunk_rows);
    std::vector<std::uint8_t> rows(tier_chunk_rows * row_bytes);
    for (std::size_t first = 0; first < store.rows(); first += tier_chunk_rows) {
        const std::size_t n = std::min(tier_chunk_rows, store.rows() - first);
        std::iota(ids.begin(), ids.begin() + n, std::uint32_t(first)); store.decode(ids.data(), n, decoded.data());
        nest.encode_tier(decoded.data(), first, n, rows.data());
        sink(first, n, rows.data());
    }
}
}  // namespace

int main(int argc, char** argv) try {
    const Options o = parse(argc, argv);
    omp_set_num_threads(int(o.threads));
    Store store(o);
    std::shared_ptr<dv::TieredQFileRows> sq8_file;
    if (!o.sq8_rows.empty()) {
        if (!std::filesystem::exists(o.sq8_rows)) {  // the store's rows verbatim, in a page-aligned row file
            dv::TieredQRowFileWriter writer(o.sq8_rows + ".tmp", o.dim);
            std::vector<std::int8_t> chunk(tier_chunk_rows * o.dim);
            for (std::size_t first = 0; first < o.rows; first += tier_chunk_rows) {
                const std::size_t n = std::min(tier_chunk_rows, o.rows - first);
                std::copy_n(store.row(first), n * o.dim, chunk.data());
                writer.append(reinterpret_cast<const std::uint8_t*>(chunk.data()), n);
            }
            writer.finish(); std::filesystem::rename(o.sq8_rows + ".tmp", o.sq8_rows);
        }
        sq8_file = std::make_shared<dv::TieredQFileRows>(o.sq8_rows, dv::TieredQFileReads::direct, o.final_depth);
    }
    std::FILE* report = std::fopen((o.out + ".json").c_str(), "w"); std::FILE* rows_out = std::fopen((o.out + ".csv").c_str(), "w");
    if (!report || !rows_out) throw std::runtime_error("cannot write outputs");

    auto start = Clock::now();
    dv::TieredQIVFIndex ivf = dv::TieredQIVFIndex::load(o.index);
    const double load_s = seconds(start); start = Clock::now();
    // The router study's best setting: expansion 64, at least 32 probed supers.
    dv::TieredQRouterConfig router; router.expansion = 64; router.min_super_probes = 32; router.seed = seed;
    ivf.build_router(router);
    const double router_s = seconds(start);
    const auto codec_sample = sample_rows(store, std::min(codec_sample_rows, o.rows), 1, o.dim);
    std::vector<QuerySet> sets;
    for (const auto& set : o.query_sets) {
        std::stringstream parts(set); std::string name, qpath, tpath, count;
        std::getline(parts, name, ':'); std::getline(parts, qpath, ':'); std::getline(parts, tpath, ':'); std::getline(parts, count, ':');
        const std::size_t nq = std::stoul(count);
        sets.push_back({name, nq, read_vecs<float>(qpath, o.dim, nq), read_vecs<std::int32_t>(tpath, top_k, nq)});
    }
    std::fprintf(rows_out, "queries,tier,total_bits,fine,tier_bytes,reads,queue_depth,lists,nprobe,candidates,recall10,p50_us,p99_us,mean_us,"
                 "rss_bytes,anon_bytes,file_bytes\n");
    std::fprintf(report, "{\"lists\": %zu, \"load_s\": %.2f, \"router_s\": %.2f, \"ivf_storage_bytes\": %zu, \"router_bytes\": %zu, \"tiers\": [",
                 ivf.lists(), load_s, router_s, ivf.storage_bytes(), ivf.router_bytes());

    // One configuration's single-query sweep: search(query, depth) -> top hits, alone on the pinned core.
    auto sweep = [&](const char* tier, std::size_t total, std::size_t fine, std::size_t tier_bytes, const char* reads,
                     std::size_t queue_depth, const std::vector<std::size_t>& nprobes, const std::vector<std::size_t>& depths,
                     auto&& search) {
        omp_set_num_threads(1);
        cpu_set_t all; sched_getaffinity(0, sizeof(all), &all);
        cpu_set_t only; CPU_ZERO(&only); CPU_SET(o.search_cpu, &only);
        if (sched_setaffinity(0, sizeof(only), &only)) throw std::runtime_error("cannot pin the search thread");
        for (const auto& set : sets) for (std::size_t nprobe : nprobes) for (std::size_t depth : depths) {
            search(set.queries.data(), nprobe, depth);  // warm
            std::vector<double> us(set.count); std::size_t hits = 0;
            for (std::size_t q = 0; q < set.count; ++q) {
                auto t = Clock::now(); auto found = search(set.queries.data() + q * o.dim, nprobe, depth);
                us[q] = std::chrono::duration<double, std::micro>(Clock::now() - t).count();
                hits += true_hits(found, set.truth, q);
            }
            const Memory memory = serving_memory();
            std::fprintf(rows_out, "%s,%s,%zu,%zu,%zu,%s,%zu,%zu,%zu,%zu,%.4f,%.1f,%.1f,%.1f,%zu,%zu,%zu\n", set.name.c_str(), tier, total,
                         fine, tier_bytes, reads, queue_depth, ivf.lists(), nprobe, depth, double(hits) / double(set.count * top_k),
                         percentile(us, 0.5), percentile(us, 0.99), std::accumulate(us.begin(), us.end(), 0.0) / set.count,
                         memory.rss, memory.anon, memory.file);
            std::fflush(rows_out);
        }
        if (sched_setaffinity(0, sizeof(all), &all)) throw std::runtime_error("cannot unpin");
        omp_set_num_threads(int(o.threads));
    };

    sweep("sq8", 8, 0, o.dim, "ram", 1, o.nprobes, o.candidates, [&](const float* q, std::size_t nprobe, std::size_t depth) {
        auto found = ivf.search(q, depth, nprobe);
        for (auto& hit : found) hit.score = -store.l2(q, hit.id);
        const std::size_t keep = std::min(top_k, found.size());
        std::partial_sort(found.begin(), found.begin() + keep, found.end(), [](auto& a, auto& b) { return a.score > b.score; });
        found.resize(keep); return found;
    });

    auto nested = [&](std::size_t total, std::size_t fine, double& train_s) {
        auto t = Clock::now();
        auto refinement = dv::TieredQRefinement::train(ivf.codec(), std::uint32_t(total * o.dim), codec_sample.data(),
                                                       codec_sample.size() / o.dim, seed, fine != 0);
        train_s = seconds(t);
        return dv::TieredQNestedIVFIndex(ivf, std::move(refinement), dv::TieredQTierRows::suffix);
    };
    bool first_tier = true;
    for (std::size_t total : o.totals) for (std::size_t fine : o.fine) {
        double train_s = 0; auto nest = nested(total, fine, train_s);
        auto t = Clock::now();
        const std::size_t row_bytes = nest.tier_bytes_per_vector();
        std::vector<std::uint8_t> tier(o.rows * row_bytes);
        encode_tier(nest, store, o.dim, [&](std::size_t first, std::size_t n, const std::uint8_t* rows) {
            std::copy_n(rows, n * row_bytes, tier.data() + first * row_bytes);
        });
        const double encode_s = seconds(t);
        nest.attach_tier(std::make_shared<dv::TieredQMemoryRows>(std::move(tier), row_bytes));
        std::fprintf(report, "%s{\"total_bits\": %zu, \"fine\": %zu, \"tier_bytes_per_vector\": %zu, \"train_s\": %.2f, \"encode_s\": %.2f}",
                     first_tier ? "" : ", ", total, fine, row_bytes, train_s, encode_s);
        std::fflush(report); first_tier = false;
        sweep("nested", total, fine, row_bytes, "ram", 1, o.nprobes, o.candidates, [&](const float* q, std::size_t nprobe, std::size_t depth) {
            return nest.search(q, top_k, nprobe, depth);
        });
        if (sq8_file) {
            std::vector<std::uint8_t> exact(o.finals.empty() ? 0 : *std::max_element(o.finals.begin(), o.finals.end()) * o.dim);
            std::vector<std::uint32_t> ids;
            sweep("hybrid", total, fine, row_bytes, "direct", o.final_depth, o.nprobes, o.finals, [&](const float* q, std::size_t nprobe, std::size_t finals) {
                auto found = nest.search(q, finals, nprobe, std::max(finals, o.hybrid_candidates));
                ids.resize(found.size());
                for (std::size_t i = 0; i < found.size(); ++i) ids[i] = found[i].id;
                sq8_file->read(ids.data(), ids.size(), exact.data());
                for (std::size_t i = 0; i < found.size(); ++i)
                    found[i].score = -store.l2(q, reinterpret_cast<const std::int8_t*>(exact.data() + i * o.dim));
                const std::size_t keep = std::min(top_k, found.size());
                std::partial_sort(found.begin(), found.begin() + keep, found.end(), [](auto& a, auto& b) { return a.score > b.score; });
                found.resize(keep); return found;
            });
        }
    }
    std::fprintf(report, "]");

    // One tier on SSD: written once, then read through the page cache and with direct I/O at each queue depth.
    if (!o.ssd_dir.empty() && o.ssd_total) {
        double train_s = 0; auto nest = nested(o.ssd_total, o.ssd_fine, train_s);
        const auto path = (std::filesystem::path(o.ssd_dir) / "tier.rows").string();
        auto t = Clock::now();
        {
            dv::TieredQRowFileWriter writer(path, nest.tier_bytes_per_vector());
            encode_tier(nest, store, o.dim, [&](std::size_t, std::size_t n, const std::uint8_t* rows) { writer.append(rows, n); });
            writer.finish();
        }
        std::fprintf(report, ", \"ssd\": {\"total_bits\": %zu, \"fine\": %zu, \"file_bytes\": %zu, \"write_s\": %.2f, \"concurrent_reads\": %s}",
                     o.ssd_total, o.ssd_fine, std::size_t(std::filesystem::file_size(path)), seconds(t),
                     dv::TieredQFileRows::concurrent_reads_available() ? "true" : "false");
        for (auto reads : {dv::TieredQFileReads::cached, dv::TieredQFileReads::direct})
            for (std::size_t depth : o.queue_depths) {
                nest.attach_tier(std::make_shared<dv::TieredQFileRows>(path, reads, depth));
                sweep("nested", o.ssd_total, o.ssd_fine, nest.tier_bytes_per_vector(), reads == dv::TieredQFileReads::cached ? "cached" : "direct",
                      depth, o.ssd_nprobes, {o.ssd_candidates}, [&](const float* q, std::size_t nprobe, std::size_t candidates) {
                          return nest.search(q, top_k, nprobe, candidates);
                      });
            }
        std::filesystem::remove(path);
    }
    std::fprintf(report, "}\n");
    std::fclose(rows_out); std::fclose(report);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what()); return 1;
}
