// Range-only rejection policy for the existing AVX-512 single-query decoder and score pipeline.
#pragma once
#include "tieredq_range_checkpoint.h"
#include "tieredq_scan_codes_x86.h"

namespace dv::tieredq_detail {
constexpr int range_rounding_up=_MM_FROUND_TO_POS_INF|_MM_FROUND_NO_EXC;
struct FastRangeCheckpoint {
  size_t rank;
  FastRangeBounds bounds;
  float threshold;
  __attribute__((target("avx512f,avx512bw"),always_inline))
  bool check(const FastScanCache& cache,const __m512* dots,const float* centers,
      const float* corrections,size_t doc,size_t valid,uint16_t* keep,__m512* terms) const {
    code_base_terms_avx512(cache,centers,corrections,doc,terms);
    const auto factor=_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(cache.factor.data()+doc)));
    const auto sign_mask=_mm512_set1_epi32(INT32_MAX);
    const auto abs_factor=_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(factor),sign_mask));
    const auto abs_base=_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(terms[0]),sign_mask));
    const auto partial=_mm512_add_round_ps(terms[0],_mm512_mul_round_ps(factor,dots[0],range_rounding_up),range_rounding_up);
    // Negative factors require the outward-rounded lower endpoint of the suffix range.
    const auto range=_mm512_mask_blend_ps(_mm512_cmp_ps_mask(factor,_mm512_setzero_ps(),_CMP_GE_OQ),
        _mm512_set1_ps(bounds.low),_mm512_set1_ps(bounds.high));
    auto upper=_mm512_add_round_ps(partial,_mm512_mul_round_ps(factor,range,range_rounding_up),range_rounding_up);
    const auto allowance=_mm512_mul_round_ps(_mm512_set1_ps(bounds.allowance_scale),
        _mm512_add_round_ps(abs_base,_mm512_mul_round_ps(abs_factor,_mm512_set1_ps(bounds.magnitude),range_rounding_up),
            range_rounding_up),range_rounding_up);
    upper=_mm512_add_round_ps(upper,allowance,range_rounding_up);
    const uint16_t valid_mask=uint16_t((uint32_t(1)<<valid)-1);
    const uint16_t rejected=uint16_t(_mm512_cmp_ps_mask(upper,_mm512_set1_ps(threshold),_CMP_LT_OQ))&valid_mask;
    keep[0]=valid_mask&~rejected;
    return keep[0]!=0;
  }
};

__attribute__((target("avx512f,avx512bw")))
inline void scan_range_code_blocks_avx512(const FastScanCache& cache,
    const std::vector<CodeTableSegment>& segments,const float* weights,const float* centers,
    const float* corrections,size_t first,size_t count,float* scores,const FastRangeCheckpoint& checkpoint) {
  for (size_t doc=first;doc<first+count;doc+=fast_scan_block_docs) {
    const size_t valid=std::min(fast_scan_block_docs,first+count-doc);
    const auto* packed=cache.levels.data()+doc/fast_scan_block_docs*cache.level_block_bytes();
    AccumulateLevelsAvx512 sink{weights,_mm512_setzero_ps()};
    size_t tier=0;
    for (;tier<segments.size() && segments[tier].end<=checkpoint.rank;++tier)
      walk_code_segment_avx512(packed,segments[tier],sink);
    uint16_t keep[1];
    __m512 terms[fast_code_query_tile_avx512];
    if (checkpoint.check(cache,&sink.dot,centers,corrections,doc,valid,keep,terms))
      for (;tier<segments.size();++tier) walk_code_segment_avx512(packed,segments[tier],sink);
    store_code_scores_avx512<1,true>(cache,&sink.dot,centers,corrections,doc,valid,scores+doc-first,count,keep,terms);
  }
}

inline void scan_fast_range_avx512(const FastScanCache& cache,const TieredQQuery& query,
    const FastRangeBounds& bounds,size_t prefix_rank,size_t first,size_t count,
    float* scores,float threshold) {
  const auto segments=code_table_segments(cache);
  const CentroidTiles centroids(1,[&](size_t)->const TieredQQuery& { return query; });
  scan_range_code_blocks_avx512(cache,segments,query.weights.data(),centroids.scores(0),centroids.corrections(0),
      first,count,scores,FastRangeCheckpoint{prefix_rank,bounds,threshold});
}
} // namespace dv::tieredq_detail
