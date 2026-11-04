// AVX2 nibble scan with the prefix-section range checkpoint (the AVX-512 kernel's counterpart): 32-document chunks
// whose documents all fail the bound skip their suffix sections and score -inf; every other score equals the unpruned
// AVX2 scan bit for bit. AVX2 has no per-instruction rounding mode and MXCSR changes would also hit the exact scores,
// so the bound adds range_nearest_slack instead.
#pragma once
#include "tieredq_lookup_range.h"
#include "tieredq_scan_int8_avx2.h"

namespace dv::tieredq_detail {
__attribute__((target("avx2,f16c"),always_inline))
inline __m256 abs_ps_avx2(__m256 x) { return _mm256_and_ps(x,_mm256_castsi256_ps(_mm256_set1_epi32(INT32_MAX))); }
// Whether any of the first valid documents of the 8 from doc can still reach the threshold after the prefix dot.
__attribute__((target("avx2,f16c"),always_inline))
inline bool lookup_vector_survives_avx2(const FastScanCache& c,const FastScanLookupQuery& q,
    const LookupRangeCheckpoint& checkpoint,size_t doc,__m256 dot,size_t valid) {
  const auto& b=checkpoint.bounds.bounds;
  const auto cid=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(c.centroid.data()+doc)));
  const auto centers=_mm256_i32gather_ps(q.centroid_scores.data(),cid,sizeof(float));
  const auto weighted=_mm256_mul_ps(_mm256_i32gather_ps(q.correction_weights.data(),cid,sizeof(float)),
      _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c.correction.data()+doc))));
  const auto factor=_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c.factor.data()+doc)));
  const auto base=_mm256_add_ps(centers,weighted);
  const auto prefix=_mm256_mul_ps(factor,dot);
  // Negative factors take the lower endpoint of the suffix range.
  const auto suffix=_mm256_mul_ps(factor,_mm256_blendv_ps(_mm256_set1_ps(b.low),_mm256_set1_ps(b.high),
      _mm256_cmp_ps(factor,_mm256_setzero_ps(),_CMP_GE_OQ)));
  const auto allowance=_mm256_mul_ps(_mm256_set1_ps(b.allowance_scale),
      _mm256_add_ps(abs_ps_avx2(base),_mm256_mul_ps(abs_ps_avx2(factor),_mm256_set1_ps(b.magnitude))));
  const auto upper=_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(base,prefix),suffix),allowance);
  const auto absolute=_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(abs_ps_avx2(centers),abs_ps_avx2(weighted)),
      _mm256_add_ps(abs_ps_avx2(prefix),abs_ps_avx2(suffix))),allowance);
  const auto reach=_mm256_cmp_ps(_mm256_add_ps(upper,_mm256_mul_ps(_mm256_set1_ps(range_nearest_slack),absolute)),
      _mm256_set1_ps(checkpoint.threshold),_CMP_GE_OQ);
  const int lanes=valid>=fast_scan_avx2_lanes ? (1<<fast_scan_avx2_lanes)-1 : (1<<valid)-1;
  return (_mm256_movemask_ps(reach)&lanes)!=0;
}
// The pruned counterpart of scan_fast_lookup_avx2_range, with the same range alignment.
__attribute__((target("avx2,f16c")))
inline void scan_range_lookup_avx2(const FastScanCache& cache,const FastScanLookupQuery& q,size_t first,size_t count,
    float* scores,const LookupRangeCheckpoint& checkpoint) {
  const auto& lookup=*cache.lookup;
  const size_t prefix=checkpoint.bounds.prefix_sections,sections=lookup.section_ends.size(),end=first+count;
  for (size_t doc=first;doc<end;doc+=fast_lookup_avx2_docs) {
    __m256 dots[fast_lookup_avx2_vectors];
    for (auto& dot:dots) dot=_mm256_setzero_ps();
    const uint8_t* base=lookup_codes_avx2(lookup,doc);
    lookup_sections_avx2(lookup,q,base,0,prefix,dots);
    bool live=false;
    for (size_t v=0;v<fast_lookup_avx2_vectors && !live && doc+v*fast_scan_avx2_lanes<end;++v) {
      const size_t at=doc+v*fast_scan_avx2_lanes;
      live=lookup_vector_survives_avx2(cache,q,checkpoint,at,dots[v],end-at);
    }
    if (!live) {
      std::fill(scores+(doc-first),scores+(std::min(end,doc+fast_lookup_avx2_docs)-first),-std::numeric_limits<float>::infinity());
      continue;
    }
    lookup_sections_avx2(lookup,q,base,prefix,sections,dots);
    lookup_chunk_epilogue_avx2(cache,q,doc,first,end,dots,scores);
  }
}
} // namespace dv::tieredq_detail
