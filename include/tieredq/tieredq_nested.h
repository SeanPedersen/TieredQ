// Nested TieredQ serving (#23): the prefix codes live in an in-RAM IVF; each query's top candidates are rescored with
// their TieredQRefinement suffix rows, read by id from a row source in RAM or on SSD. Candidates' prefix rows come
// from the IVF lists themselves (TieredQIVFIndex::document_rows), so the only per-vector data outside the index is
// the suffix.
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

// A row file: a header, then rows from a page-aligned offset. Each candidate is one positioned read (pread), so the
// tier can sit on SSD; the OS page cache serves repeated reads.
class TieredQFileRows final : public TieredQRowSource {
 public:
  explicit TieredQFileRows(const std::string& path);
  ~TieredQFileRows() override;
  TieredQFileRows(const TieredQFileRows&) = delete;
  TieredQFileRows& operator=(const TieredQFileRows&) = delete;
  size_t row_bytes() const override { return row_bytes_; }
  size_t rows() const override { return rows_; }
  void read(const uint32_t* ids, size_t count, uint8_t* out) const override;
  static constexpr uint32_t format = 1;
  static constexpr size_t data_alignment = 4096;
 private:
  int fd_ = -1;
  size_t row_bytes_ = 0, rows_ = 0;
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

class TieredQNestedIVFIndex {
 public:
  // index is built; refinement was trained on index.codec() (checked by model id).
  TieredQNestedIVFIndex(TieredQIVFIndex index, TieredQRefinement refinement);
  const TieredQIVFIndex& index() const { return index_; }
  const TieredQRefinement& refinement() const { return refinement_; }
  size_t size() const { return index_.size(); }
  size_t suffix_bytes_per_vector() const { return refinement_.bytes_per_vector(); }
  // Suffix rows of the input rows with ids [first, first + count) (rows: count x dimensions, as given to build), in
  // id order, for a row source.
  void encode_suffix(const float* rows, size_t first, size_t count, uint8_t* out) const;
  // One suffix row per id; rows() must equal size().
  void attach_suffix(std::shared_ptr<const TieredQRowSource> suffix);
  bool has_suffix() const { return suffix_ != nullptr; }
  // The IVF's top `candidates` (k' >= k) by prefix score, rescored by prefix plus suffix; scores are the metric's
  // (as TieredQIVFIndex::search returns them).
  std::vector<TieredQHit> search(const float* query, size_t k, size_t nprobe, size_t candidates,
                                 TieredQPruning pruning = {}) const;
  // Results are rows x min(k, size()) in row-major order; slots past the probed rows hold id UINT32_MAX and
  // score -inf. Queries are split across OpenMP workers.
  void search_batch(const float* queries, size_t rows, size_t k, size_t nprobe, size_t candidates, float* scores,
                    uint32_t* ids, TieredQPruning pruning = {}) const;
  // Index and refinement bytes; the suffix rows are stored by the row source and attached after loading.
  std::vector<uint8_t> serialize() const;
  static TieredQNestedIVFIndex deserialize(const std::vector<uint8_t>& bytes);

 private:
  // Rescores one query's found candidates (best first) in place to its top k; returns how many it kept.
  size_t rescore(const float* query, TieredQQuery weights, const uint32_t* candidates, size_t found, size_t k,
                 float* scores, uint32_t* ids) const;
  TieredQIVFIndex index_;
  TieredQRefinement refinement_;
  std::vector<uint32_t> documents_;  // input id -> stored IVF document
  std::shared_ptr<const TieredQRowSource> suffix_;
};
} // namespace dv
