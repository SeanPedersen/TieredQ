// NEON nibble scan with the prefix-section range checkpoint (the AVX-512 kernel's counterpart): 16-document vectors
// whose documents all fail the bound skip their suffix sections and score -inf; every other score equals the unpruned
// NEON scan bit for bit. NEON has no per-instruction rounding mode, so the bound adds range_nearest_slack.
#pragma once
#include "tieredq_lookup_range.h"
#include "tieredq_scan_lookup_neon.h"

namespace dv::tieredq_detail {
// Whether any of the first valid documents of the vector at doc can still reach the threshold after the prefix dot.
inline bool lookup_vector_survives_neon(const FastScanCache& c,const FastScanLookupQuery& q,
    const LookupRangeCheckpoint& checkpoint,size_t doc,const float32x4_t (&dot)[lookup_neon_quarters],size_t valid) {
  const auto& b=checkpoint.bounds.bounds;
  const auto high=vdupq_n_f32(b.high),low=vdupq_n_f32(b.low),magnitude=vdupq_n_f32(b.magnitude);
  const auto allowance_scale=vdupq_n_f32(b.allowance_scale),slack=vdupq_n_f32(range_nearest_slack);
  const auto threshold=vdupq_n_f32(checkpoint.threshold);
  const uint32_t lane_order[lookup_neon_lanes]={0,1,2,3};
  const auto lanes=vld1q_u32(lane_order);
  auto survivors=vdupq_n_u32(0);
  for (size_t quarter=0;quarter<lookup_neon_quarters && quarter*lookup_neon_lanes<valid;++quarter) {
    const size_t at=doc+quarter*lookup_neon_lanes;
    const auto terms=lookup_centroid_terms_neon(c,q,at);
    const auto factor=half_quarter_neon(c.factor.data()+at);
    const auto weighted=vmulq_f32(terms.weights,half_quarter_neon(c.correction.data()+at));
    const auto base=vaddq_f32(terms.centers,weighted);
    const auto prefix=vmulq_f32(factor,dot[quarter]);
    // Negative factors take the lower endpoint of the suffix range.
    const auto suffix=vmulq_f32(factor,vbslq_f32(vcgeq_f32(factor,vdupq_n_f32(0)),high,low));
    const auto allowance=vmulq_f32(allowance_scale,vaddq_f32(vabsq_f32(base),vmulq_f32(vabsq_f32(factor),magnitude)));
    const auto upper=vaddq_f32(vaddq_f32(vaddq_f32(base,prefix),suffix),allowance);
    const auto absolute=vaddq_f32(vaddq_f32(vaddq_f32(vabsq_f32(terms.centers),vabsq_f32(weighted)),
        vaddq_f32(vabsq_f32(prefix),vabsq_f32(suffix))),allowance);
    const auto reach=vcgeq_f32(vaddq_f32(upper,vmulq_f32(slack,absolute)),threshold);
    const auto in_range=vcltq_u32(lanes,vdupq_n_u32(uint32_t(valid-quarter*lookup_neon_lanes)));
    survivors=vorrq_u32(survivors,vandq_u32(reach,in_range));
  }
  return vmaxvq_u32(survivors)!=0;
}
template<size_t Vectors>
inline void scan_range_lookup_vectors_neon(const FastScanCache& cache,const FastScanLookupQuery& q,const uint8_t* base,
    size_t doc,size_t end,float* out,const LookupRangeCheckpoint& checkpoint) {
  const auto& lookup=*cache.lookup;
  const size_t prefix=checkpoint.bounds.prefix_sections,sections=lookup.section_ends.size();
  float32x4_t dots[Vectors][lookup_neon_quarters];
  for (auto& vector:dots) for (auto& dot:vector) dot=vdupq_n_f32(0);
  lookup_sections_neon<Vectors>(lookup,q,base,0,prefix,dots);
  auto valid=[&](size_t v) { return std::min(lookup_vector_docs,end-(doc+v*lookup_vector_docs)); };
  bool live[Vectors];
  size_t survivors=0;
  for (size_t v=0;v<Vectors;++v) {
    live[v]=lookup_vector_survives_neon(cache,q,checkpoint,doc+v*lookup_vector_docs,dots[v],valid(v));
    survivors+=live[v];
  }
  // All survivors finish their suffix together; otherwise each survivor finishes alone.
  if (survivors==Vectors) lookup_sections_neon<Vectors>(lookup,q,base,prefix,sections,dots);
  else for (size_t v=0;v<Vectors;++v) if (live[v]) {
    float32x4_t single[1][lookup_neon_quarters];
    std::copy_n(dots[v],lookup_neon_quarters,single[0]);
    lookup_sections_neon<1>(lookup,q,base+v*lookup_vector_docs,prefix,sections,single);
    std::copy_n(single[0],lookup_neon_quarters,dots[v]);
  }
  for (size_t v=0;v<Vectors;++v) {
    float* scores=out+v*lookup_vector_docs;
    if (live[v]) lookup_epilogue_neon(cache,q,doc+v*lookup_vector_docs,dots[v],scores,valid(v));
    else std::fill_n(scores,valid(v),-std::numeric_limits<float>::infinity());
  }
}
// The pruned counterpart of scan_fast_lookup_neon_range.
inline void scan_range_lookup_neon(const FastScanCache& cache,const FastScanLookupQuery& q,size_t first,size_t count,
    float* scores,const LookupRangeCheckpoint& checkpoint) {
  for_lookup_vector_runs_neon(cache,first,count,scores,[&](auto vectors,const uint8_t* base,size_t doc,size_t end,float* out) {
    scan_range_lookup_vectors_neon<decltype(vectors)::value>(cache,q,base,doc,end,out,checkpoint);
  });
}
} // namespace dv::tieredq_detail
