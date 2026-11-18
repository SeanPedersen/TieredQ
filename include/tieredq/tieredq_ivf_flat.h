// Flat inverted-file index (#56): rows grouped by k-means list, scored exactly (no codes ranked approximately). Its batch engine is
// list-major: queries are grouped by their nearest list, and each group is scored against every list its members
// probe in one sgemm, instead of one matrix-vector scan per query and list. Recall equals per-query IVF at the same
// nprobe; the arithmetic runs at GEMM throughput.
//
// knn_graph() is the self-join that dimensionality reduction (UMAP, t-SNE, NCVis) needs: every stored row's k nearest
// other rows. On DBpedia 100K x 512 (8 threads, build plus all rows) it reached recall@15 0.940 in 2.4 s and 0.982 in
// 4.4 s, against hnswlib's 0.942 in 5.9 s and 0.980 in 12.8 s. NN-descent (pynndescent) is still faster: 0.946 in
// 2.0 s and 0.979 in 2.5 s at 100K, and about 7x faster at 1M, as IVF scores every row of the probed lists
// (ann-prototypes/knn_graph/README.md).
//
// Lists are scored in L2-sized tiles of rows (1.7x faster joins than whole-list sgemms at 1M x 1024). Storage sq8
// keeps the rows as int8 codes with per-dimension scales and offsets (a quarter of FP32's bytes) and decodes each tile
// into FP32 before its sgemm, so the arithmetic stays FP32 and queries stay FP32 (asymmetric, like Faiss IVF-SQ8). On
// MS MARCO 1M x 1024 it joined 8-17% faster than FP32 (a quarter of the bytes to stream) for 0.002-0.004 recall@15.
#pragma once
#include "tieredq/tieredq.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tq {
// Row storage: FP32, or SQ8 codes spanning each dimension's [min, max] (or the codes given to build).
enum class TieredQIVFFlatStorage { fp32, sq8 };
// List training: Lloyd k-means (iterations), or e-means (epochs; mini-batch EMA k-means, about four times fewer passes
// for Lloyd's partition quality, so it can afford a larger sample at the same cost).
enum class TieredQIVFFlatTrainer { lloyd, emeans };

struct TieredQIVFFlatConfig {
  uint32_t lists = 0;  // 0: automatic_lists(rows) at build
  TieredQMetric metric = TieredQMetric::l2;
  TieredQIVFFlatStorage storage = TieredQIVFFlatStorage::fp32;
  // k-means on a seeded sample of sample_rows_per_list rows per list, exact (BLAS) assignment. E-means (#45) streams
  // its sample: on MS MARCO 1M x 1024 (1,024 lists) 256 rows per list x 2.5 epochs trained in 1.66 s against 2.06 s
  // for Lloyd's former default (64 rows x 10 iterations), with higher knn_graph and query recall.
  TieredQIVFFlatTrainer trainer = TieredQIVFFlatTrainer::emeans;
  uint32_t iterations = 10;  // lloyd
  double epochs = 2.5;       // emeans: passes over the sample, may be fractional
  uint32_t sample_rows_per_list = 256;
  uint64_t seed = 42;
  void validate() const;
};

// Build phases of the last build(), in wall seconds.
struct TieredQIVFFlatBuildStats {
  double train_s = 0, assign_s = 0, store_s = 0;
};

class TieredQIVFFlatIndex {
 public:
  TieredQIVFFlatIndex(uint32_t dimensions, TieredQIVFFlatConfig config = {});
  ~TieredQIVFFlatIndex();
  TieredQIVFFlatIndex(TieredQIVFFlatIndex&&) noexcept;
  TieredQIVFFlatIndex& operator=(TieredQIVFFlatIndex&&) noexcept;

  // Trains the lists on rows (count x dimensions) and stores them; replaces earlier content. Ids are row numbers.
  void build(const float* rows, size_t count);
  // The same from an SQ8 store (count x dimensions int8 codes, row = codes * scales + offsets, dimensions values each),
  // decoded a chunk at a time, so no FP32 copy of the base is made. sq8 storage keeps the codes as given (cosine
  // re-encodes the normalized rows); fp32 storage keeps the decoded rows.
  void build(const int8_t* codes, size_t count, const float* scales, const float* offsets);
  // Lists for the next builds from given centroids (lists x dimensions, row-major) instead of k-means, e.g. from
  // another clustering method; train_s then stays 0. Cleared by clear_centroids().
  void set_centroids(const float* centroids, size_t lists);
  void clear_centroids();

  // Results are rows x k, best first. Scores: inner product, cosine similarity, or -|q - x|^2; larger is better.
  // Slots past the probed rows hold id UINT32_MAX and score -inf. Work runs on OpenMP workers.
  void search(const float* queries, size_t rows, size_t k, size_t nprobe, float* scores, uint32_t* ids) const;
  // Every stored row's k nearest other rows (size() x k, row r answers stored row r), probing nprobe lists per row.
  void knn_graph(size_t k, size_t nprobe, float* scores, uint32_t* ids) const;

  size_t size() const;
  uint32_t dimensions() const;
  size_t lists() const;
  const TieredQIVFFlatConfig& config() const;
  const TieredQIVFFlatBuildStats& build_stats() const;
  std::vector<uint32_t> list_assignment() const;  // each row's list, in row order
  // The lists' centroids as routing ranks them (binary16-rounded), lists() x dimensions, row-major.
  const std::vector<float>& centroids() const;
  size_t storage_bytes() const;                   // rows (or codes and their scales), per-row norms and ids, centroids

  // About sqrt(rows) lists, a power of two (256 at 100K rows, 1,024 at 1M).
  static uint32_t automatic_lists(size_t rows);

 private:
  struct State;
  std::unique_ptr<State> state_;
};
} // namespace tq
