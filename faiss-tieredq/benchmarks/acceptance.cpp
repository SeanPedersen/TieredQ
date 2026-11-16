// #55 acceptance benchmark for IndexIVFTieredQ on one shared partition.
//   build DATA SEED OUT: Faiss k-means lists (seeded sample, 32 threads), then TieredQIVFIndex with those lists
//     (set_lists), IndexIVFTieredQ on TieredQ's codec and TieredQ's row assignment, and Faiss's IndexIVFRaBitQ
//     (1, 2, 4 bits) and IndexIVFPQFastScan (1 and 2 bits per dimension) on the same lists and assignment. Encoders
//     train on the same 32,768 rows. Writes every index, the centroids and build.json.
//   serve DATA OUT KIND NAME RESULT: one index in a fresh process: serving RAM (RssAnon growth across load and coarse
//     setup), then alone on QUERY_CORE with one thread, per nprobe single-query QPS (best of passes), p50, recall@k
//     and every hit.
//     64K+ lists rank through an IndexHNSWFlat over the centroids (Faiss and plugin) or TieredQ's own router.
//   parity DATA OUT RESULT: TieredQIVFIndex (exact coarse ranking) against IndexIVFTieredQ (flat quantizer) on the
//     first parity_queries queries: per nprobe, queries with identical (id, score) sets and with identical scores.
// DATA holds meta.txt ("rows dimensions k f32|sq8"), queries.f32, truth.i32 (k ids per query) and base.f32, or the
// SQ8 store base.i8 with scales.f32 and offsets.f32 (vector = code * scale + offset).
#include "faiss_tieredq/IndexIVFTieredQ.h"
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/index_io.h>
#include <tieredq/tieredq.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <malloc.h>
#include <memory>
#include <numeric>
#include <omp.h>
#include <random>
#include <sched.h>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
using faiss::idx_t;
using Clock = std::chrono::steady_clock;
constexpr size_t hnsw_lists = 65536, hnsw_links = 32, min_ef_search = 64, ef_per_probe = 2, passes = 3;
constexpr size_t chunk_rows = size_t(1) << 20, parity_queries = 50;

double seconds_since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

template<class T> std::vector<T> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::vector<T> values(size_t(in.tellg()) / sizeof(T));
  in.seekg(0); in.read(reinterpret_cast<char*>(values.data()), std::streamsize(values.size() * sizeof(T)));
  return values;
}
template<class T> void write_file(const std::string& path, const T* values, size_t count) {
  std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(values), std::streamsize(count * sizeof(T)));
}

const void* map_file(const std::string& path) {
  const int fd = open(path.c_str(), O_RDONLY);
  struct stat info{};
  if (fd < 0 || fstat(fd, &info)) throw std::runtime_error("cannot open " + path);
  void* data = mmap(nullptr, size_t(info.st_size), PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (data == MAP_FAILED) throw std::runtime_error("cannot map " + path);
  return data;
}

// Base rows decoded on demand (float32 file or SQ8 store, memory-mapped), queries and their true neighbors.
struct Dataset {
  size_t rows = 0, d = 0, k = 0;
  std::vector<float> queries;
  std::vector<int32_t> truth;
  const float* floats = nullptr;
  const int8_t* codes = nullptr;
  std::vector<float> scales, offsets;

  explicit Dataset(const std::string& dir) {
    std::string kind;
    std::ifstream(dir + "/meta.txt") >> rows >> d >> k >> kind;
    queries = read_file<float>(dir + "/queries.f32");
    truth = read_file<int32_t>(dir + "/truth.i32");
    if (kind == "f32") { floats = static_cast<const float*>(map_file(dir + "/base.f32")); return; }
    codes = static_cast<const int8_t*>(map_file(dir + "/base.i8"));
    scales = read_file<float>(dir + "/scales.f32"); offsets = read_file<float>(dir + "/offsets.f32");
  }
  size_t query_count() const { return queries.size() / d; }
  void gather(const uint32_t* ids, size_t n, float* out) const {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) decode(ids[i], out + i * d);
  }
  void range(size_t first, size_t n, float* out) const {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) decode(first + i, out + i * d);
  }
  void decode(size_t row, float* out) const {
    if (floats) { std::copy_n(floats + row * d, d, out); return; }
    for (size_t j = 0; j < d; ++j) out[j] = float(codes[row * d + j]) * scales[j] + offsets[j];
  }
};

std::vector<uint32_t> seeded_sample(size_t rows, size_t count, uint64_t seed) {
  std::vector<uint32_t> ids(rows);
  std::iota(ids.begin(), ids.end(), 0u);
  std::shuffle(ids.begin(), ids.end(), std::mt19937_64(seed));
  ids.resize(std::min(rows, count));
  std::sort(ids.begin(), ids.end());
  return ids;
}

// Hands out precomputed lists in call order, so an index that assigns rows itself gets the shared partition; residual
// encoders reconstruct the real centroids through it.
struct FixedAssignment : faiss::Index {
  FixedAssignment(const faiss::Index& centroids, const std::vector<idx_t>& lists)
      : Index(centroids.d), centroids_(centroids), lists_(lists) { is_trained = true; ntotal = centroids.ntotal; }
  void reconstruct(idx_t key, float* recons) const override { centroids_.reconstruct(key, recons); }
  void add(idx_t, const float*) override { throw std::logic_error("FixedAssignment::add"); }
  void reset() override { cursor_ = 0; }
  void assign(idx_t n, const float*, idx_t* labels, idx_t k = 1) const override {
    if (k != 1) throw std::logic_error("FixedAssignment::assign k");
    std::copy_n(lists_.begin() + idx_t(cursor_), n, labels);
    cursor_ += size_t(n);
  }
  void search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels, const faiss::SearchParameters*) const override {
    assign(n, x, labels, k);
    std::fill_n(distances, n, 0.0f);
  }
  const faiss::Index& centroids_;
  const std::vector<idx_t>& lists_;
  mutable size_t cursor_ = 0;
};

struct FaissConfig { std::string name; std::function<faiss::IndexIVF*(faiss::Index*, size_t, size_t)> make; };
std::vector<FaissConfig> faiss_configs(size_t d) {
  std::vector<FaissConfig> configs;
  for (uint8_t bits : {1, 2, 4})
    configs.push_back({"rabitq" + std::to_string(bits), [bits](faiss::Index* q, size_t dims, size_t lists) {
      return new faiss::IndexIVFRaBitQ(q, dims, lists, faiss::METRIC_INNER_PRODUCT, true, bits);
    }});
  for (size_t m : {d / 4, d / 2})  // 4-bit codes: 1 and 2 bits per dimension
    configs.push_back({"pq4fs-m" + std::to_string(m), [m](faiss::Index* q, size_t dims, size_t lists) {
      return new faiss::IndexIVFPQFastScan(q, dims, lists, m, 4, faiss::METRIC_INNER_PRODUCT);
    }});
  return configs;
}

std::unique_ptr<faiss::IndexFlatL2> flat_quantizer(size_t d, const std::vector<float>& centroids) {
  auto quantizer = std::make_unique<faiss::IndexFlatL2>(idx_t(d));
  quantizer->add(idx_t(centroids.size() / d), centroids.data());
  return quantizer;
}

void build(const Dataset& data, uint64_t seed, const std::string& out) {
  std::ostringstream report;
  report << "{\"seed\": " << seed;
  const size_t d = data.d;
  const bool large = data.rows > 4000000;
  const size_t lists = large ? hnsw_lists : 2048, points_per_list = large ? 40 : 64;
  // Lists: Faiss k-means (L2, as TieredQ's coarse lists are for inner product) on a seeded sample.
  auto started = Clock::now();
  const auto sample_ids = seeded_sample(data.rows, lists * points_per_list, seed);
  std::vector<float> sample(sample_ids.size() * d);
  data.gather(sample_ids.data(), sample_ids.size(), sample.data());
  faiss::ClusteringParameters params;
  params.niter = large ? 4 : 10; params.seed = int(seed); params.max_points_per_centroid = int(points_per_list);
  faiss::Clustering clustering(int(d), int(lists), params);
  faiss::IndexFlatL2 assigner{idx_t(d)};
  clustering.train(idx_t(sample_ids.size()), sample.data(), assigner);
  const std::vector<float> centroids = clustering.centroids;
  write_file(out + "/centroids.f32", centroids.data(), centroids.size());
  report << ", \"lists\": " << lists << ", \"kmeans_s\": " << seconds_since(started);
  // The codec's training rows, as the plugin's train_encoder (and TieredQ's Python build) draws them.
  const auto codec_ids = faiss_tieredq::codec_training_sample(data.rows);
  std::vector<float> codec_rows(codec_ids.size() * d);
  data.gather(codec_ids.data(), codec_ids.size(), codec_rows.data());
  const uint32_t code_bits = uint32_t(large ? d : d * 7 / 4);
  auto plugin_quantizer = flat_quantizer(d, centroids);
  faiss_tieredq::IndexIVFTieredQ plugin(plugin_quantizer.release(), d, lists, code_bits, faiss::METRIC_INNER_PRODUCT);
  plugin.own_fields = true;
  started = Clock::now();
  plugin.train_encoder(idx_t(codec_ids.size()), codec_rows.data(), nullptr);
  plugin.is_trained = true;
  report << ", \"codec_train_s\": " << seconds_since(started) << ", \"code_bits\": " << code_bits;
  // TieredQIVFIndex on the same codec and lists.
  started = Clock::now();
  tq::TieredQIVFIndex tieredq(plugin.codec(), uint32_t(lists));
  tieredq.set_lists(centroids.data());
  tieredq.build([&](const uint32_t* ids, size_t n, float* rows) { data.gather(ids, n, rows); }, data.rows);
  report << ", \"tieredq_build_s\": " << seconds_since(started) << ", \"tieredq_storage_bytes\": " << tieredq.storage_bytes();
  tieredq.save(out + "/tieredq.tq");
  const auto assignment32 = tieredq.list_assignment();
  const std::vector<idx_t> assignment(assignment32.begin(), assignment32.end());
  // The plugin on TieredQ's assignment: lists laid out once for their final sizes, rows added a chunk at a time.
  started = Clock::now();
  std::vector<size_t> sizes(lists, 0);
  for (idx_t list : assignment) ++sizes[size_t(list)];
  for (size_t list = 0; list < lists; ++list) plugin.lists().set_capacity(list, sizes[list]);
  std::vector<float> chunk(chunk_rows * d);
  for (size_t first = 0; first < data.rows; first += chunk_rows) {
    const size_t n = std::min(chunk_rows, data.rows - first);
    data.range(first, n, chunk.data());
    plugin.add_core(idx_t(n), chunk.data(), nullptr, assignment.data() + first);
  }
  report << ", \"plugin_add_s\": " << seconds_since(started) << ", \"plugin_storage_bytes\": " << plugin.storage_bytes();
  faiss_tieredq::write_index_tieredq(plugin, out + "/plugin.tqf");
  // Faiss's codecs on the same lists, assignment and training rows.
  for (const auto& config : faiss_configs(d)) {
    std::unique_ptr<faiss::IndexIVF> index(config.make(flat_quantizer(d, centroids).release(), d, lists));
    index->own_fields = true;
    started = Clock::now();
    index->train(idx_t(codec_ids.size()), codec_rows.data());
    const double train_s = seconds_since(started);
    started = Clock::now();
    faiss::Index* own = index->quantizer;
    FixedAssignment fixed(*own, assignment);
    index->quantizer = &fixed;
    for (size_t first = 0; first < data.rows; first += chunk_rows) {
      const size_t n = std::min(chunk_rows, data.rows - first);
      data.range(first, n, chunk.data());
      index->add(idx_t(n), chunk.data());
    }
    index->quantizer = own;
    report << ", \"" << config.name << "_train_s\": " << train_s << ", \"" << config.name << "_add_s\": " << seconds_since(started);
    faiss::write_index(index.get(), (out + "/" + config.name + ".faiss").c_str());
  }
  report << "}\n";
  std::ofstream(out + "/build.json") << report.str();
}

size_t rss_anon_bytes() {
  std::ifstream status("/proc/self/status");
  for (std::string line; std::getline(status, line);)
    if (line.rfind("RssAnon:", 0) == 0) return std::stoul(line.substr(8)) * 1024;
  return 0;
}

std::vector<size_t> nprobes(const Dataset& data) {
  return data.rows > 4000000 ? std::vector<size_t>{64, 128, 256, 512, 1024} : std::vector<size_t>{8, 16, 32, 48, 64};
}

// One query's top k, ids as int64 and scores as Faiss returns them.
using Searcher = std::function<void(const float* query, size_t nprobe, float* scores, idx_t* ids)>;

void serve(const Dataset& data, const std::string& out, const std::string& kind, const std::string& name, const std::string& result) {
  const size_t k = data.k, d = data.d, queries = data.query_count();
  const auto centroids = read_file<float>(out + "/centroids.f32");
  const bool hnsw = centroids.size() / d >= hnsw_lists;
  malloc_trim(0);
  const size_t before = rss_anon_bytes();
  std::unique_ptr<tq::TieredQIVFIndex> tieredq;
  std::unique_ptr<faiss::IndexIVF> ivf;
  faiss::SearchParametersHNSW coarse;
  faiss::SearchParametersIVF params;
  if (kind == "tieredq") {
    tieredq = std::make_unique<tq::TieredQIVFIndex>(tq::TieredQIVFIndex::load(out + "/tieredq.tq"));
  } else {
    ivf.reset(kind == "plugin" ? faiss_tieredq::read_index_tieredq(out + "/" + name)
                               : dynamic_cast<faiss::IndexIVF*>(faiss::read_index((out + "/" + name).c_str())));
    if (hnsw) {
      auto graph = std::make_unique<faiss::IndexHNSWFlat>(int(d), int(hnsw_links), faiss::METRIC_L2);
      graph->add(idx_t(centroids.size() / d), centroids.data());
      if (ivf->own_fields) delete ivf->quantizer;
      ivf->quantizer = graph.release();
      ivf->own_fields = true;
      params.quantizer_params = &coarse;
    }
  }
  malloc_trim(0);
  const size_t ram = rss_anon_bytes() - before;
  cpu_set_t query_core; CPU_ZERO(&query_core); CPU_SET(std::atoi(std::getenv("QUERY_CORE")), &query_core);
  if (sched_setaffinity(0, sizeof(query_core), &query_core)) throw std::runtime_error("cannot pin to QUERY_CORE");
  omp_set_num_threads(1);
  const Searcher search = tieredq
      ? Searcher([&](const float* q, size_t nprobe, float* scores, idx_t* ids) {
          const auto hits = tieredq->search(q, k, nprobe);
          for (size_t i = 0; i < k; ++i) { ids[i] = i < hits.size() ? idx_t(hits[i].id) : -1; scores[i] = i < hits.size() ? hits[i].score : 0; }
        })
      : Searcher([&](const float* q, size_t nprobe, float* scores, idx_t* ids) {
          params.nprobe = nprobe;
          coarse.efSearch = int(std::max(min_ef_search, ef_per_probe * nprobe));
          ivf->search(1, q, idx_t(k), scores, ids, &params);
        });
  std::ostringstream report;
  report << "{\"kind\": \"" << kind << "\", \"name\": \"" << name << "\", \"ram_bytes\": " << ram << ", \"points\": [";
  std::vector<float> scores(queries * k);
  std::vector<idx_t> ids(queries * k);
  for (size_t p = 0; p < nprobes(data).size(); ++p) {
    const size_t nprobe = nprobes(data)[p];
    search(data.queries.data(), nprobe, scores.data(), ids.data());  // warm
    double best = 1e300;
    std::vector<double> latencies(queries), fastest;
    for (size_t pass = 0; pass < passes; ++pass) {
      for (size_t q = 0; q < queries; ++q) {
        const auto started = Clock::now();
        search(data.queries.data() + q * d, nprobe, scores.data() + q * k, ids.data() + q * k);
        latencies[q] = seconds_since(started);
      }
      const double total = std::accumulate(latencies.begin(), latencies.end(), 0.0);
      if (total < best) { best = total; fastest = latencies; }
    }
    std::sort(fastest.begin(), fastest.end());
    size_t found = 0;
    for (size_t q = 0; q < queries; ++q)
      for (size_t i = 0; i < k; ++i)
        found += std::count(data.truth.begin() + q * k, data.truth.begin() + (q + 1) * k, int32_t(ids[q * k + i]));
    report << (p ? ", " : "") << "{\"nprobe\": " << nprobe << ", \"qps\": " << double(queries) / best << ", \"p50_us\": "
           << fastest[queries / 2] * 1e6 << ", \"recall\": " << double(found) / double(queries * k) << "}";
    write_file(result + ".np" + std::to_string(nprobe) + ".ids", ids.data(), ids.size());
    write_file(result + ".np" + std::to_string(nprobe) + ".scores", scores.data(), scores.size());
  }
  report << "]}\n";
  std::ofstream(result) << report.str();
}

void parity(const Dataset& data, const std::string& out, const std::string& result) {
  const size_t k = data.k, d = data.d;
  const auto tieredq = tq::TieredQIVFIndex::load(out + "/tieredq.tq");
  std::unique_ptr<faiss_tieredq::IndexIVFTieredQ> plugin(faiss_tieredq::read_index_tieredq(out + "/plugin.tqf"));
  const tq::TieredQRouting exact{tq::TieredQRouting::Mode::exact, 0};
  std::ostringstream report;
  report << "{\"queries\": " << parity_queries << ", \"points\": [";
  auto probes = nprobes(data);
  probes.push_back(plugin->nlist);
  for (size_t p = 0; p < probes.size(); ++p) {
    size_t same_hits = 0, same_scores = 0;
    faiss::SearchParametersIVF params; params.nprobe = probes[p];
    std::vector<float> scores(parity_queries * k);
    std::vector<idx_t> ids(parity_queries * k);
    plugin->search(idx_t(parity_queries), data.queries.data(), idx_t(k), scores.data(), ids.data(), &params);
    for (size_t q = 0; q < parity_queries; ++q) {
      const auto hits = tieredq.search(data.queries.data() + q * d, k, probes[p], {}, exact);
      std::vector<std::pair<idx_t, float>> left, right;
      std::vector<float> left_scores, right_scores;
      for (size_t i = 0; i < k; ++i) {
        left.emplace_back(i < hits.size() ? idx_t(hits[i].id) : -1, i < hits.size() ? hits[i].score : 0);
        right.emplace_back(ids[q * k + i], ids[q * k + i] < 0 ? 0 : scores[q * k + i]);
        left_scores.push_back(left.back().second); right_scores.push_back(right.back().second);
      }
      same_scores += left_scores == right_scores;
      std::sort(left.begin(), left.end()); std::sort(right.begin(), right.end());
      same_hits += left == right;
    }
    report << (p ? ", " : "") << "{\"nprobe\": " << probes[p] << ", \"same_hits\": " << same_hits << ", \"same_scores\": " << same_scores << "}";
  }
  report << "]}\n";
  std::ofstream(result) << report.str();
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "build" && argc == 5) build(Dataset(argv[2]), std::stoull(argv[3]), argv[4]);
    else if (mode == "serve" && argc == 7) serve(Dataset(argv[2]), argv[3], argv[4], argv[5], argv[6]);
    else if (mode == "parity" && argc == 5) parity(Dataset(argv[2]), argv[3], argv[4]);
    else { std::fprintf(stderr, "usage: see the header of benchmarks/acceptance.cpp\n"); return 2; }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
