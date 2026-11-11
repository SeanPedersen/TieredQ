// Nested TieredQ serving (#23): the prefix codes live in an in-RAM IVF; each query's top candidates are rescored with
// their TieredQRefinement suffix rows, read by id from a row source in RAM or on SSD (TieredQTierRows: with or
// without a copy of the prefix row). TieredQNestedIVFIndex::build trains and builds the low-RAM configuration (#49, #52)
// by default, from float rows or an SQ8 store (sq8_row_loader).
#pragma once
#include "tieredq/tieredq.h"
#include "tieredq/tieredq_refinement.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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

// Exact input vectors (dimensions() floats, as given to build) for an optional final rescoring stage, decoded by id.
// decode() is called concurrently from search workers.
class TieredQExactRows {
 public:
  virtual ~TieredQExactRows() = default;
  virtual size_t dimensions() const = 0;
  virtual size_t rows() const = 0;
  // Writes rows ids[0..count) to out (count x dimensions()).
  virtual void decode(const uint32_t* ids, size_t count, float* out) const = 0;
};

// Per-dimension int8 scalar quantization (an SQ8 store): row i is codes[i][j] * scales[j] + offsets[j], its int8
// codes read from a row source of dimensions bytes per row (in RAM or a TieredQFileRows file on SSD).
class TieredQSQ8Rows final : public TieredQExactRows {
 public:
  TieredQSQ8Rows(std::shared_ptr<const TieredQRowSource> codes, std::vector<float> scales, std::vector<float> offsets);
  size_t dimensions() const override { return scales_.size(); }
  size_t rows() const override { return codes_->rows(); }
  void decode(const uint32_t* ids, size_t count, float* out) const override;
 private:
  std::shared_ptr<const TieredQRowSource> codes_;
  std::vector<float> scales_, offsets_;
};

// Rows of an SQ8 store in memory (e.g. a memory-mapped file of rows x scales.size() int8 codes, row-major), decoded as
// TieredQSQ8Rows decodes them, for builds and tier writes that need no float copy of the base. codes must outlive the
// loader; ids must be below rows.
TieredQIVFIndex::RowLoader sq8_row_loader(const int8_t* codes, size_t rows, std::vector<float> scales,
                                          std::vector<float> offsets);

// What a tier row holds. prefix_and_suffix stores each vector's prefix row (codec().bytes_per_vector()) before its
// suffix, so a candidate costs one contiguous read. suffix keeps only the suffix and decodes the prefix from the IVF
// lists, a smaller tier at a few microseconds of decoding per candidate (about 3.4 us at a 1024-bit prefix, d = 1536).
// Neither adds RAM beyond the IVF.
enum class TieredQTierRows { prefix_and_suffix, suffix };

// TieredQNestedIVFIndex::build settings; 0 or unset picks the default. The defaults are the low-RAM configuration on
// Qwen/MS MARCO 8.84M x 1024 (#49, #52): a 1-bit prefix (4-bit scan cap: nibble kernels) in RAM over
// automatic_lists() lists (64K) with the IVF's automatic list training and router, and an 8-bit fine suffix tier,
// stored suffix-only for an SSD. With the tier on NVMe (direct reads, 32 in flight) and 40 candidates, serving RAM was
// 2.03 GB for recall@10 0.944-0.984 at 2.2-8.4 ms per query on one core (nprobe 96-1024), 0.991 at nprobe 4096; the
// 2-bit prefix at 20 candidates reached the same recall with 3.35 GB at 2.4-11.6 ms.
struct TieredQNestedConfig {
  static constexpr uint32_t automatic_lists = UINT32_MAX;
  uint32_t dimensions = 0;              // required
  uint32_t prefix_bits = 0;             // 0: 1 bit per dimension
  uint32_t total_bits = 0;              // prefix included; 0: 8 bits per dimension
  uint32_t lists = automatic_lists;     // TieredQIVFIndex::automatic_lists(rows); 0: the codec's centroids
  TieredQMetric metric = TieredQMetric::inner_product;
  uint32_t max_scan_bits = 0;           // prefix scan cap; 0: TieredQIVFIndex::automatic_scan_bits
  bool fine_tiers = true;               // suffix widths up to max_tier_bits (inner product and cosine)
  uint32_t max_tier_bits = 12;
  TieredQTierRows tier_rows = TieredQTierRows::suffix;
  std::optional<TieredQCoarseTraining> coarse_training;  // unset: TieredQCoarseTraining::automatic(lists)
  // Unset: the router (with router_config) from TieredQRouterConfig::automatic_lists lists on; true or false forces it.
  std::optional<bool> router;
  TieredQRouterConfig router_config;
  uint64_t seed = 0;
};

class TieredQNestedIVFIndex {
 public:
  // index is built; refinement was trained on index.codec() (checked by model id).
  TieredQNestedIVFIndex(TieredQIVFIndex index, TieredQRefinement refinement,
                        TieredQTierRows tier_rows = TieredQTierRows::suffix);
  // Trains the prefix codec and the refinement on one seeded sample of at most 32,768 rows (of sample when given,
  // else of rows), then builds the lists over rows (the given sample also trains the lists, on all its rows, so a
  // partition learned on earlier data stays fixed). No tier is attached yet: write_tier() or encode_tier() it.
  static TieredQNestedIVFIndex build(const TieredQNestedConfig& config, const TieredQIVFIndex::RowLoader& rows,
                                     size_t count, const float* sample = nullptr, size_t sample_rows = 0);
  // Candidates a search rescores when none are given: 40 up to k = 20, 2k beyond. With the 1-bit prefix and the 8-bit
  // tier on SSD, 20 candidates lost about 0.005 recall@10 against 30-40, and 40 kept 0.0008 over 30 from nprobe 1024;
  // each 10 more reads cost about 0.12 ms (the 2-bit prefix was flat from 20). On VIBE arxiv at k = 100, 200
  // candidates came within 0.001 of 400 at 1.5-2.6x the QPS, and 100 capped recall@100 near 0.905.
  static constexpr size_t default_candidates = 40;
  static size_t candidates_for(size_t k) { return std::max(default_candidates, 2 * k); }
  const TieredQIVFIndex& index() const { return index_; }
  const TieredQRefinement& refinement() const { return refinement_; }
  TieredQTierRows tier_rows() const { return tier_rows_; }
  size_t size() const { return index_.size(); }
  size_t tier_bytes_per_vector() const;
  // Tier rows of the input rows with ids [first, first + count) (rows: count x dimensions, as given to build), in id
  // order, for a row source.
  void encode_tier(const float* rows, size_t first, size_t count, uint8_t* out) const;
  // Encodes every indexed vector (rows: the build()'s input, by id) into a TieredQFileRows file, a chunk at a time.
  void write_tier(const std::string& path, const TieredQIVFIndex::RowLoader& rows) const;
  // One tier row per id; rows() must equal size().
  void attach_tier(std::shared_ptr<const TieredQRowSource> tier);
  bool has_tier() const { return tier_ != nullptr; }
  // Optional final stage: the tier's best max(k, finals) candidates are rescored with their exact rows (one per
  // indexed vector, in id order), so the ranking matches the exact vectors' while only a few rows per query are read,
  // e.g. an SQ8 store left on SSD. Null detaches it.
  void attach_exact(std::shared_ptr<const TieredQExactRows> exact, size_t finals);
  bool has_exact() const { return exact_ != nullptr; }
  size_t finals() const { return finals_; }
  // The IVF's top `candidates` (k' >= k) by prefix score, rescored by prefix plus suffix (then exactly, with
  // attach_exact); scores are the metric's (as TieredQIVFIndex::search returns them).
  std::vector<TieredQHit> search(const float* query, size_t k, size_t nprobe, size_t candidates,
                                 TieredQPruning pruning = {}) const;
  std::vector<TieredQHit> search(const float* query, size_t k, size_t nprobe) const {
    return search(query, k, nprobe, candidates_for(k));
  }
  // Results are rows x min(k, size()) in row-major order; slots past the probed rows hold id UINT32_MAX and
  // score -inf. Queries are split across OpenMP workers.
  void search_batch(const float* queries, size_t rows, size_t k, size_t nprobe, size_t candidates, float* scores,
                    uint32_t* ids, TieredQPruning pruning = {}) const;
  void search_batch(const float* queries, size_t rows, size_t k, size_t nprobe, float* scores, uint32_t* ids) const {
    search_batch(queries, rows, k, nprobe, candidates_for(k), scores, ids);
  }
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
  size_t rescore(const float* query, const TieredQRefinementQuery& refined_query, const uint32_t* candidates, size_t found,
                 const uint8_t* tier_rows, size_t k, float* scores, uint32_t* ids) const;
  // The exact stage of rescore: candidates[order[0..staged)] (best tier score first) by their exact rows, to the top
  // kept; returns kept.
  size_t rescore_exact(const float* query, const uint32_t* candidates, const uint32_t* order, size_t staged, size_t kept,
                       float* scores, uint32_t* ids) const;
  // Rescores row-major candidate lists (depth per query, UINT32_MAX-terminated) into rows x min(k, size()) results.
  void rescore_rows(const float* queries, size_t rows, std::vector<TieredQQuery> weights, const uint32_t* candidates,
                    size_t depth, const uint8_t* tier_rows, size_t k, float* scores, uint32_t* ids) const;
  TieredQIVFIndex index_;
  TieredQRefinement refinement_;
  TieredQTierRows tier_rows_;
  std::vector<uint32_t> documents_;  // input id -> stored IVF document (suffix-only tiers)
  std::shared_ptr<const TieredQRowSource> tier_;
  std::shared_ptr<const TieredQExactRows> exact_;
  size_t finals_ = 0;
};
} // namespace dv
