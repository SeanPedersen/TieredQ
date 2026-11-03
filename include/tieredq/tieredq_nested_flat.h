// Flat nested TieredQ search (#13): a narrow prefix scanned for every document, then the top `candidates` rescored with
// their TieredQRefinement suffix from RAM. On OpenAI-1536 a 512-bit prefix with a 3 bits/dim total tier reached
// RaBitQ's 4-bit recall at 2.2-2.9x its single-query QPS and 0.76-0.79x its stored bits (100K / 1M). It is a
// TieredQNestedIVFIndex with one list, scanned whole.
#pragma once
#include "tieredq/tieredq_nested.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dv {
class TieredQNestedIndex {
 public:
  // Encodes rows (count x dimensions) as the prefix scan list and an in-RAM tier. refinement was trained on prefix
  // (TieredQRefinement::train on the prefix's sample and seed).
  TieredQNestedIndex(TieredQ prefix, TieredQRefinement refinement, const float* rows, size_t count,
                     TieredQTierRows tier_rows = TieredQTierRows::suffix);
  size_t size() const { return nested_.size(); }
  // Prefix scan plus tier, in bytes.
  size_t storage_bytes() const;
  // The top `candidates` (>= k) by prefix score, rescored by prefix plus suffix; scores are the metric's.
  std::vector<TieredQHit> search(const float* query, size_t k, size_t candidates) const;
  // Results are rows x min(k, size()) in row-major order. Queries are split across OpenMP workers.
  void search_batch(const float* queries, size_t rows, size_t k, size_t candidates, float* scores,
                    uint32_t* ids) const;
  // The underlying one-list nested index, for serialization.
  const TieredQNestedIVFIndex& nested() const { return nested_; }

 private:
  TieredQNestedIVFIndex nested_;
};
} // namespace dv
