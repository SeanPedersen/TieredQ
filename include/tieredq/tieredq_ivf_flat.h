// Unquantized inverted-file index (#56): FP32 rows grouped by k-means list, no codec to train. Its batch engine is
// list-major: queries are grouped by their nearest list, and each group is scored against every list its members
// probe in one sgemm, instead of one matrix-vector scan per query and list. Recall equals per-query IVF at the same
// nprobe; the arithmetic runs at GEMM throughput.
//
// knn_graph() is the self-join that dimensionality reduction (UMAP, t-SNE, NCVis) needs: every stored row's k nearest
// other rows. On DBpedia 100K x 512 (8 threads, build plus all rows) it reached recall@15 0.940 in 2.4 s and 0.982 in
// 4.4 s, against hnswlib's 0.942 in 5.9 s and 0.980 in 12.8 s. NN-descent (pynndescent) is still faster: 0.946 in
// 2.0 s and 0.979 in 2.5 s at 100K, and about 7x faster at 1M, as IVF scores every row of the probed lists
// (ann-prototypes/knn_graph/README.md).
#pragma once
#include "tieredq/tieredq.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tq {
struct TieredQIVFFlatConfig {
  uint32_t lists = 0;  // 0: automatic_lists(rows) at build
  TieredQMetric metric = TieredQMetric::l2;
  // k-means on a seeded sample of sample_rows_per_list rows per list, exact (BLAS) assignment.
  uint32_t iterations = 10;
  uint32_t sample_rows_per_list = 64;
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
  size_t storage_bytes() const;                   // rows, per-row norms and ids, centroids

  // About sqrt(rows) lists, a power of two (256 at 100K rows, 1,024 at 1M).
  static uint32_t automatic_lists(size_t rows);

 private:
  struct State;
  std::unique_ptr<State> state_;
};
} // namespace tq
