// Exact/fast execution over immutable index state; single queries are serial and native batches own workers.
#pragma once
#include <tieredq/tieredq.h>
#include <algorithm>

namespace tieredq_python {
enum class SearchMode { configured, fast };
inline void search_rows(const tq::TieredQIndex& index, const float* queries,
                        size_t rows, size_t width, size_t k, float* scores,
                        uint32_t* ids, SearchMode mode) {
  const size_t found = std::min(k, index.size());
  if (!rows || !found) return;
  if (rows == 1) {
    const auto hits = mode == SearchMode::fast ? index.search_fast(queries, k) : index.search(queries, k);
    for (size_t i = 0; i < hits.size(); ++i) { scores[i] = hits[i].score; ids[i] = hits[i].id; }
    return;
  }
  if (mode == SearchMode::fast) index.search_batch_fast(queries, rows, k, scores, ids);
  else index.search_batch(queries, rows, k, scores, ids);
}
}  // namespace tieredq_python
