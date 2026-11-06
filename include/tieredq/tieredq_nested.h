// Nested TieredQ serving (#23): the prefix codes live in an in-RAM IVF; each query's top candidates are rescored with
// their TieredQRefinement suffix rows, read by id from a row source in RAM or on SSD (TieredQTierRows: with or
// without a copy of the prefix row).
#pragma once
#include "tieredq/tieredq.h"
#include "tieredq/tieredq_refinement.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dv {
// Fixed-size rows addressed by id. read() is called concurrently from search workers.
class TieredQRowSource {
 public:
  virtual ~TieredQRowSource() = default;
  virtual size_t row_bytes() const = 0;
  virtual size_t rows() const = 0;
  // Copies rows ids[0..count) to out (count * row_bytes()).
  virtual void read(const uint32_t* ids, size_t count, uint8_t* out) const = 0;
};

class TieredQMemoryRows final : public TieredQRowSource {
 public:
  TieredQMemoryRows(std::vector<uint8_t> data, size_t row_bytes);
  size_t row_bytes() const override { return row_bytes_; }
  size_t rows() const override { return rows_; }
  void read(const uint32_t* ids, size_t count, uint8_t* out) const override;
 private:
  std::vector<uint8_t> data_;
  size_t row_bytes_, rows_;
};

// How TieredQFileRows reads: through the OS page cache, or with direct I/O that bypasses it, so every candidate is a
// device read of the aligned pages covering its row (one, or two when a row crosses a page boundary).
enum class TieredQFileReads { cached, direct };

// A row file: a header, then rows from a page-aligned offset, so the tier can sit on SSD. Each candidate is one
// positioned read. With queue_depth > 1 a read() keeps up to that many in flight on a per-thread io_uring, which
// SSDs need for throughput (one read at a time waits about 100 us each); builds without liburing read one at a time.
class TieredQFileRows final : public TieredQRowSource {
 public:
  explicit TieredQFileRows(const std::string& path, TieredQFileReads reads = TieredQFileReads::cached,
                           size_t queue_depth = 1);
  ~TieredQFileRows() override;
  TieredQFileRows(const TieredQFileRows&) = delete;
  TieredQFileRows& operator=(const TieredQFileRows&) = delete;
  size_t row_bytes() const override { return row_bytes_; }
  size_t rows() const override { return rows_; }
  void read(const uint32_t* ids, size_t count, uint8_t* out) const override;
  size_t queue_depth() const { return queue_depth_; }
  // Whether this build issues concurrent reads (io_uring); otherwise any queue depth reads one row at a time.
  static bool concurrent_reads_available();
  static constexpr uint32_t format = 1;
  // Data offset and direct-read granularity; covers the logical block size of common devices.
  static constexpr size_t data_alignment = 4096;
 private:
  int fd_ = -1;
  TieredQFileReads reads_;
  size_t queue_depth_;
  size_t row_bytes_ = 0, rows_ = 0, file_bytes_ = 0;
};

// Writes a TieredQFileRows file in id order, appended in chunks; finish() completes it.
class TieredQRowFileWriter {
 public:
  TieredQRowFileWriter(const std::string& path, size_t row_bytes);
  ~TieredQRowFileWriter();
  TieredQRowFileWriter(const TieredQRowFileWriter&) = delete;
  TieredQRowFileWriter& operator=(const TieredQRowFileWriter&) = delete;
  void append(const uint8_t* rows, size_t count);
  void finish();
 private:
  int fd_ = -1;
  size_t row_bytes_, rows_ = 0;
};

// What a tier row holds. prefix_and_suffix stores each vector's prefix row (codec().bytes_per_vector()) before its
// suffix, so a candidate costs one contiguous read. suffix keeps only the suffix and decodes the prefix from the IVF
// lists, a smaller tier at a few microseconds of decoding per candidate (about 3.4 us at a 1024-bit prefix, d = 1536).
// Neither adds RAM beyond the IVF.
enum class TieredQTierRows { prefix_and_suffix, suffix };

class TieredQNestedIVFIndex {
 public:
  // index is built; refinement was trained on index.codec() (checked by model id).
  TieredQNestedIVFIndex(TieredQIVFIndex index, TieredQRefinement refinement,
                        TieredQTierRows tier_rows = TieredQTierRows::prefix_and_suffix);
  const TieredQIVFIndex& index() const { return index_; }
  const TieredQRefinement& refinement() const { return refinement_; }
  TieredQTierRows tier_rows() const { return tier_rows_; }
  size_t size() const { return index_.size(); }
  size_t tier_bytes_per_vector() const;
  // Tier rows of the input rows with ids [first, first + count) (rows: count x dimensions, as given to build), in id
  // order, for a row source.
  void encode_tier(const float* rows, size_t first, size_t count, uint8_t* out) const;
  // One tier row per id; rows() must equal size().
  void attach_tier(std::shared_ptr<const TieredQRowSource> tier);
  bool has_tier() const { return tier_ != nullptr; }
  // The IVF's top `candidates` (k' >= k) by prefix score, rescored by prefix plus suffix; scores are the metric's
  // (as TieredQIVFIndex::search returns them).
  std::vector<TieredQHit> search(const float* query, size_t k, size_t nprobe, size_t candidates,
                                 TieredQPruning pruning = {}) const;
  // Results are rows x min(k, size()) in row-major order; slots past the probed rows hold id UINT32_MAX and
  // score -inf. Queries are split across OpenMP workers.
  void search_batch(const float* queries, size_t rows, size_t k, size_t nprobe, size_t candidates, float* scores,
                    uint32_t* ids, TieredQPruning pruning = {}) const;
  // Second step of a search over a tier stored elsewhere (e.g. a database); the first is index().search_batch with
  // depth candidates. Rescores each query's candidate ids (rows x depth, ended early by UINT32_MAX) using their tier
  // rows (rows x depth x tier_bytes_per_vector(), as encode_tier() writes them; slots past a query's last id are
  // ignored). Results as search_batch; no tier needs to be attached.
  void rescore_batch(const float* queries, size_t rows, const uint32_t* candidates, size_t depth,
                     const uint8_t* tier_rows, size_t k, float* scores, uint32_t* ids) const;
  // Index, refinement and tier layout; the tier rows are stored by the row source and attached after loading.
  std::vector<uint8_t> serialize() const;
  static TieredQNestedIVFIndex deserialize(const std::vector<uint8_t>& bytes);

 private:
  // The attached tier's rows of the candidates, in a per-thread buffer valid until the next call.
  const uint8_t* read_tier(const uint32_t* candidates, size_t found) const;
  // Writes the candidates' row-major prefix rows (from the lists for suffix-only tiers) and returns their suffix
  // rows: the tier rows themselves for suffix-only tiers, else split into suffix_rows.
  const uint8_t* candidate_rows(const uint32_t* candidates, size_t found, const uint8_t* tier_rows,
                                uint8_t* prefix_rows, uint8_t* suffix_rows) const;
  // Rescores one query's found candidates (best first, with their tier rows) to its top k; returns how many it kept.
  // found_scores, the scan's metric scores of the candidates when known, let additive refinements over suffix-only
  // tiers skip reading the candidates' prefix codes.
  size_t rescore(const float* query, const TieredQRefinementQuery& refined_query, const uint32_t* candidates,
                 const float* found_scores, size_t found, const uint8_t* tier_rows, size_t k, float* scores,
                 uint32_t* ids) const;
  // The candidates' refined model-space scores, from the scan's scores or else from their prefix rows.
  void refined_scores(const float* query, const TieredQRefinementQuery& refined_query, const uint32_t* candidates,
                      const float* found_scores, size_t found, const uint8_t* tier_rows, float* out) const;
  // Rescores row-major candidate lists (depth per query, UINT32_MAX-terminated; found_scores alongside, or null)
  // into rows x min(k, size()) results.
  void rescore_rows(const float* queries, size_t rows, std::vector<TieredQQuery> weights, const uint32_t* candidates,
                    const float* found_scores, size_t depth, const uint8_t* tier_rows, size_t k, float* scores,
                    uint32_t* ids) const;
  TieredQIVFIndex index_;
  TieredQRefinement refinement_;
  TieredQTierRows tier_rows_;
  std::vector<uint32_t> documents_;  // input id -> stored IVF document (suffix-only tiers)
  std::shared_ptr<const TieredQRowSource> tier_;
};
} // namespace dv
