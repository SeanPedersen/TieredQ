// Batch execution over immutable index state; each worker owns one output row.
#pragma once
#include <tieredq/tieredq.h>
#include <algorithm>
#include <exception>
#include <mutex>
#include <omp.h>

namespace tieredq_python {
inline void search_rows(const dv::TieredQIndex& index, const float* queries,
                        size_t rows, size_t width, size_t k, float* scores,
                        uint32_t* ids) {
  const size_t found = std::min(k, index.size());
  if (!rows || !found) return;
  std::exception_ptr failure;
  std::mutex failure_mutex;
  const size_t workers = std::min(rows, static_cast<size_t>(omp_get_max_threads()));
#pragma omp parallel for schedule(static) num_threads(workers) if(workers > 1 && !omp_in_parallel())
  for (size_t row = 0; row < rows; ++row) {
    try {
      const auto hits = index.search(queries + row * width, k);
      for (size_t hit = 0; hit < hits.size(); ++hit) {
        scores[row * found + hit] = hits[hit].score;
        ids[row * found + hit] = hits[hit].id;
      }
    } catch (...) {
      // Exceptions must return to the calling thread before crossing the Python boundary.
      std::lock_guard lock(failure_mutex);
      if (!failure) failure = std::current_exception();
    }
  }
  if (failure) std::rethrow_exception(failure);
}
}  // namespace tieredq_python
