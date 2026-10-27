// Incremental deterministic top-k selection and exception-safe parallel query preparation.
#pragma once
#include "tieredq/tieredq.h"
#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <mutex>
#include <numeric>
#include <omp.h>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif
namespace dv::tieredq_detail {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
// Positions i < n with scores[i] > threshold, in order; returns how many were written to out.
__attribute__((target("avx512f"))) inline size_t above_threshold(const float* scores,size_t n,float threshold,uint32_t* out) {
  constexpr size_t selection_lanes=16;
  const __m512 limit=_mm512_set1_ps(threshold);
  const __m512i step=_mm512_set1_epi32(int(selection_lanes));
  __m512i positions=_mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
  size_t found=0,i=0;
  for (;i+selection_lanes<=n;i+=selection_lanes,positions=_mm512_add_epi32(positions,step)) {
    const __mmask16 above=_mm512_cmp_ps_mask(_mm512_loadu_ps(scores+i),limit,_CMP_GT_OQ);
    if (!above) continue;
    _mm512_mask_compressstoreu_epi32(out+found,above,positions);
    found+=size_t(__builtin_popcount(above));
  }
  for (;i<n;++i) if (scores[i]>threshold) out[found++]=uint32_t(i);
  return found;
}
#endif
inline size_t select_above(const float* scores,size_t n,float threshold,uint32_t* out) {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
  if (__builtin_cpu_supports("avx512f")) return tieredq_detail::above_threshold(scores,n,threshold,out);
#endif
  size_t found=0;
  for (size_t i=0;i<n;++i) if (scores[i]>threshold) out[found++]=uint32_t(i);
  return found;
}
constexpr size_t selection_block = 256;
inline bool better_hit(const TieredQHit& a,const TieredQHit& b) {
  return a.score>b.score || (a.score==b.score && a.id<b.id);
}
// Incremental top-k over fixed-size score blocks; ids grow with the block base so ties keep
// the lowest id exactly like the original search loop.
class TopK {
 public:
  explicit TopK(size_t k) : k_(k) { hits_.reserve(k); }
  void add(const float* scores,size_t n,size_t base) {
    const size_t found=hits_.size()<k_ ? (std::iota(candidates_.begin(),candidates_.begin()+n,0u),n)
                                       : select_above(scores,n,hits_.front().score,candidates_.data());
    for (size_t c=0;c<found;++c) {
      TieredQHit hit{scores[candidates_[c]],uint32_t(base+candidates_[c])};
      if (hits_.size()<k_) { hits_.push_back(hit); std::push_heap(hits_.begin(),hits_.end(),better_hit); }
      else if (better_hit(hit,hits_.front())) {
        std::pop_heap(hits_.begin(),hits_.end(),better_hit); hits_.back()=hit; std::push_heap(hits_.begin(),hits_.end(),better_hit);
      }
    }
  }
  // Until the heap is full, every document remains eligible.
  float threshold() const {
    return hits_.size()<k_ ? -std::numeric_limits<float>::infinity() : hits_.front().score;
  }
  std::vector<TieredQHit> finish() { std::sort(hits_.begin(),hits_.end(),better_hit); return std::move(hits_); }
 private:
  size_t k_;
  std::vector<TieredQHit> hits_;
  std::array<uint32_t,selection_block> candidates_;
};
// Query preparation is shared by exact and fast batches; exceptions must leave the worker region safely.
template<class Prepare>
auto prepare_search_queries(const float* queries,size_t rows,size_t dimensions,Prepare prepare) {
  using Prepared=decltype(prepare(queries));
  std::vector<Prepared> prepared(rows);
  if (!rows) return prepared;
  std::exception_ptr failure; std::mutex failure_mutex;
#pragma omp parallel for schedule(static) num_threads(std::min(rows,size_t(omp_get_max_threads()))) if(rows>1 && !omp_in_parallel())
  for (size_t row=0;row<rows;++row) {
    try { prepared[row]=prepare(queries+row*dimensions); }
    catch (...) {
      std::lock_guard<std::mutex> lock(failure_mutex);
      if (!failure) failure=std::current_exception();
    }
  }
  if (failure) std::rethrow_exception(failure);
  return prepared;
}
} // namespace dv::tieredq_detail
