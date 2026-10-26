// x86 fast-scan dispatch: narrow models use compact query-specific byte lookup tables.
// Wider models retain the AVX-512BW level-dot fallback with a bias removed per query;
// vpmaddubsw/vpmaddwd support CPUs without VNNI.
#pragma once
#include "tieredq_scan_int8.h"
#include "tieredq_scan_query_x86.h"
#include "tieredq_scan_lookup_avx512.h"
#include "tieredq_scan_codes_x86.h"
#include <immintrin.h>
#include <omp.h>

namespace dv::tieredq_detail {
__attribute__((target("avx512f,avx512bw")))
inline __mmask16 fast_mask(size_t n) {
  return n >= 16 ? __mmask16(0xFFFF) : __mmask16((1u<<n)-1);
}

// Tile boundaries are multiples of lookup_block_docs; only the final tile may be short.
constexpr size_t fast_scan_query_tile_x86 = 16;
template<size_t query_tile>
__attribute__((target("avx512f,avx512bw")))
inline void scan_fast_avx512_tile_impl(const FastScanCache& cache, const FastScanQueryX86* queries,
    size_t rows, size_t first, size_t count, float* scores, size_t stride) {
  const size_t query_count=query_tile==1 ? 1 : rows;
  if (cache.code_lookup()) {
    scan_codes_avx512_tile(cache,query_count,first,count,scores,stride,
        [&](size_t row)->const TieredQQuery& {return queries[row].codes;});
    return;
  }
  if (cache.lookup) {
    for (size_t row=0; row<query_count; ++row)
      scan_fast_lookup_avx512_range(cache,queries[row].lookup,first,count,scores+row*stride);
    return;
  }
  const __m512i ones = _mm512_set1_epi16(1);
  const __m512i bias = _mm512_set1_epi8(char(cache.level_max + 1));
  for (size_t block=first/fast_scan_block_docs; block<(first+count+fast_scan_block_docs-1)/fast_scan_block_docs; ++block) {
    __m512i sums[query_tile];
    for (size_t t=0; t<query_count; ++t) sums[t]=_mm512_setzero_si512();
    const int8_t* base = cache.levels.data() + block*cache.groups*64;
    for (size_t g=0; g<cache.groups; ++g) {
      const __m512i levels = _mm512_add_epi8(_mm512_loadu_si512(base + g*64), bias);
      for (size_t t=0; t<query_count; ++t)
        sums[t] = _mm512_add_epi32(sums[t],
            _mm512_madd_epi16(_mm512_maddubs_epi16(levels, _mm512_set1_epi32(queries[t].qwords[g])), ones));
    }
    const __m512i cid = _mm512_cvtepu8_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(cache.centroid.data()+block*fast_scan_block_docs)));
    const __m512 correction = _mm512_loadu_ps(cache.correction.data()+block*fast_scan_block_docs);
    const __m512 factor = _mm512_loadu_ps(cache.factor.data()+block*fast_scan_block_docs);
    const size_t n = std::min(fast_scan_block_docs,first+count-block*fast_scan_block_docs);
    const __mmask16 mask = fast_mask(n);
    for (size_t t=0; t<query_count; ++t) {
      const auto& q=queries[t];
      const __m512 dots = _mm512_mul_ps(
          _mm512_sub_ps(_mm512_cvtepi32_ps(sums[t]), _mm512_set1_ps(q.offset)), _mm512_set1_ps(q.scale));
      __m512 score = _mm512_i32gather_ps(cid,q.centroid_scores.data(),sizeof(float));
      score = _mm512_add_ps(score,_mm512_mul_ps(
          _mm512_i32gather_ps(cid,q.correction_weights.data(),sizeof(float)),correction));
      score = _mm512_add_ps(score,_mm512_mul_ps(factor,dots));
      _mm512_mask_storeu_ps(scores+t*stride+block*fast_scan_block_docs-first,mask,score);
    }
  }
}
inline void scan_fast_avx512_tile(const FastScanCache& cache,const FastScanQueryX86* queries,
    size_t query_count,size_t first,size_t count,float* scores,size_t stride) {
  if (query_count==1) scan_fast_avx512_tile_impl<1>(cache,queries,1,first,count,scores,stride);
  else scan_fast_avx512_tile_impl<fast_scan_query_tile_x86>(cache,queries,query_count,first,count,scores,stride);
}
inline void scan_fast_avx512(const FastScanCache& cache, const FastScanQueryX86& query, float* scores) {
  scan_fast_avx512_tile(cache,&query,1,0,cache.count,scores,cache.count);
}
inline void scan_fast_avx512_batch(const FastScanCache& cache,
    const std::vector<FastScanQueryX86>& queries,float* scores) {
  const size_t tile=cache.lookup ? 1 : cache.code_lookup() ? fast_code_query_group : fast_scan_query_tile_x86;
#pragma omp parallel for schedule(static) if(queries.size()>tile && !omp_in_parallel())
  for (size_t first=0;first<queries.size();first+=tile)
    scan_fast_avx512_tile(cache,queries.data()+first,std::min(tile,queries.size()-first),
        0,cache.count,scores+first*cache.count,cache.count);
}
} // namespace dv::tieredq_detail
