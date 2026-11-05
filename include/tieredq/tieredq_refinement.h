// Nested TieredQ codes (#23): a suffix that refines a trained prefix model towards a finer one. The prefix keeps
// serving alone (e.g. an in-RAM IVF scan); suffix rows, read by id, rescore its candidates as the finer code would.
// - Each prefix tier keeps its rotation block and gains whole bits per coordinate: 2^gain sub-levels inside every
//   prefix cell, fitted by 1-D Lloyd on the training rows of that cell (in units of the row's prefix factor).
// - PCA ranks the prefix leaves uncoded take a finer model's basis (same sample and seed), rotated in runs of equal
//   bits, with their own Lloyd levels and a per-row least-squares scale.
// - The suffix stores its own centroid correction for the refined reconstruction.
// A suffix row is suffix_code_bytes() of packed codes, then binary16 scale and correction.
#pragma once
#include "tieredq/tieredq.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace dv {
namespace tieredq_detail { struct RefinementEncoder; }
struct TieredQRefinementQuery {
  // Prefix projection terms (TieredQ::model_query_weights): centroid scores and correction weights.
  TieredQQuery prefix;
  // Weight (projection * sigma) per refined position: prefix ranks, then the ranks only the suffix codes.
  std::vector<float> weights;
  float extra_mean = 0;  // projected mean of the positions the prefix leaves uncoded
};
class TieredQRefinement {
 public:
  // total_code_bits sizes the finer model (prefix plus suffix). sample, rows and seed must be the prefix's training
  // inputs: the finer model has to share its centroids and PCA basis, which training checks.
  // fine_tiers enables wider IP/cosine allocation for the suffix (max_tier_bits in [8, 12]);
  // the prefix must have fine tiers disabled.
  static TieredQRefinement train(const TieredQ& prefix, uint32_t total_code_bits, const float* sample, size_t rows,
                                 uint64_t seed = 0, bool fine_tiers = false, uint32_t max_tier_bits = 12);
  const TieredQ& prefix() const { return prefix_; }
  // Widest combined prefix/suffix code; fine_tiers allows up to max_tier_bits (at most 12).
  uint32_t max_code_bits() const;
  size_t suffix_bits() const { return suffix_bits_; }
  size_t suffix_code_bytes() const { return (suffix_bits_ + 7) / 8; }
  size_t bytes_per_vector() const { return suffix_code_bytes() + metadata_bytes; }
  // Positions scored per row: the prefix's coded ranks plus the ranks only the suffix codes.
  size_t positions() const;
  // Writes count * bytes_per_vector() bytes; prefix_rows are the same rows' prefix.encode() output.
  void encode(const float* rows, size_t count, const uint8_t* prefix_rows, uint8_t* out) const;
  TieredQRefinementQuery query(const float* vector) const;
  // Reuses the prefix's query_weights() for the same vector (e.g. from the scan that found the candidates).
  TieredQRefinementQuery query(const float* vector, TieredQQuery prefix_query) const;
  // Batched query() over contiguous rows and their prefix queries (e.g. prefix().query_weights(vectors, rows)); results
  // equal per-row calls exactly. Up to 8 queries share each pass over the suffix basis, which the per-row projection
  // streams from memory once per query; blocks of queries and basis panels run on OpenMP workers.
  std::vector<TieredQRefinementQuery> query(const float* vectors, size_t rows,
                                            std::vector<TieredQQuery> prefix_queries) const;
  // Row-major prefix and suffix rows (their bytes_per_vector() strides) of the same count vectors; scores estimate
  // model-space inner products, as TieredQ::score does.
  // kernel: automatic, scalar or avx512 (AVX-512F/BW); others are rejected.
  void score(const uint8_t* prefix_rows, const uint8_t* suffix_rows, size_t count, const TieredQRefinementQuery& query,
             float* scores, TieredQKernel kernel = TieredQKernel::automatic) const;
  // The prefix is persisted separately; deserialize checks it is the model the suffix was trained on.
  std::vector<uint8_t> serialize() const;
  static TieredQRefinement deserialize(const std::vector<uint8_t>& data, TieredQ prefix);
  static constexpr size_t metadata_bytes = 4;

 private:
  // One prefix tier: its prefix bits stay where the prefix stores them; gain bits per coordinate go to the suffix.
  struct Block { uint32_t start, dimensions, bits, prefix_offset, gain, suffix_offset; std::vector<float> levels; };
  // A rotated run of ranks only the suffix codes, with its Lloyd levels.
  struct Run { uint32_t start, dimensions, bits, suffix_offset; std::vector<float> levels; };
  TieredQRefinement() = default;
  TieredQ prefix_;
  std::vector<Block> blocks_;  // levels: 2^bits cells x 2^gain sub-levels, cell major (unit: factor * sigma)
  std::vector<Run> runs_;
  // Suffix-only ranks: binary16 model-space basis (row-major d x extra), mean and sigma per rank.
  std::vector<uint16_t> extra_basis_;
  std::vector<float> extra_mean_, extra_sigma_;
  size_t suffix_bits_ = 0;
  // Derived serving state (not serialized): the extra basis times sigma as int8 rank panels with a scale per rank,
  // and mean / sigma per rank.
  std::vector<int8_t> extra_panels_;
  std::vector<float> extra_panel_scale_;
  std::vector<float> extra_mean_ratio_;
  // Up to 16 consecutive positions of one block or run: fixed-width prefix and suffix fields (0 bits where a side
  // holds none) and the offset of its levels in level_table_, indexed by (prefix code << shift) | suffix code.
  struct Segment { uint32_t position, count, prefix_bit, prefix_bits, suffix_bit, suffix_bits, shift, table, table_size; };
  std::vector<Segment> segments_;
  size_t extra_segment_ = 0;  // first segment of the ranks only the suffix codes
  std::vector<float> level_table_;
  // Encoder-only state (every position's basis as FP32 panels, centroid projections), built on first use.
  std::shared_ptr<tieredq_detail::RefinementEncoder> encoder_;
  void initialize_layout(const TieredQ& finer, uint64_t seed, uint32_t bit_cap);
  void prepare();
  void check_prefix_query(const TieredQQuery& prefix_query) const;
  TieredQRefinementQuery assemble_query(TieredQQuery prefix_query, const float* extra_weights) const;
  const tieredq_detail::RefinementEncoder& encoder() const;
  void fit_levels(const float* model_rows, size_t rows, const std::vector<uint8_t>& prefix_rows);
  // Centered projections of rows' centroid residuals: prefix ranks then extra ranks, positions() per row.
  void project(const float* model_rows, size_t count, const uint8_t* prefix_rows, std::vector<float>& out) const;
};
} // namespace dv
