// NEON byte-table full scan over the nibble-lookup cache (layout in tieredq_scan_lookup_x86.h): vqtbl1q_s8 maps the
// 16 nibble codes of one group to its table entries, the AArch64 counterpart of the x86 byte shuffles. Sections sum in
// int16 and convert once per section, in the scalar reference's order, so scores match it bit for bit.
#pragma once
#include "tieredq_scan_lookup_x86.h"
#include <arm_neon.h>
#include <type_traits>

namespace tq::tieredq_detail {
constexpr size_t lookup_neon_lanes=4,lookup_neon_quarters=lookup_vector_docs/lookup_neon_lanes;
constexpr size_t lookup_neon_vectors=lookup_chunk_docs/lookup_vector_docs;
static_assert(lookup_vector_docs==16 && lookup_table_entries==16 && lookup_pair_groups==2 && lookup_neon_vectors==4);

inline float32x4_t half_quarter_neon(const uint16_t* halves) { return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(halves))); }
struct LookupCentroidTermsNeon { float32x4_t centers,weights; };
// Query centroid scores and correction weights of the 4 documents from doc.
inline LookupCentroidTermsNeon lookup_centroid_terms_neon(const FastScanCache& cache,const FastScanLookupQuery& q,size_t doc) {
  float centers[lookup_neon_lanes],weights[lookup_neon_lanes];
  for (size_t lane=0;lane<lookup_neon_lanes;++lane) {
    const uint8_t cid=cache.centroid[doc+lane];
    centers[lane]=q.centroid_scores[cid]; weights[lane]=q.correction_weights[cid];
  }
  return {vld1q_f32(centers),vld1q_f32(weights)};
}
// Centroid score, plus correction weight times correction, plus factor times dot, for the 16 documents from doc.
inline void lookup_epilogue_neon(const FastScanCache& cache,const FastScanLookupQuery& q,size_t doc,
    const float32x4_t (&dot)[lookup_neon_quarters],float* out,size_t valid) {
  alignas(16) float scores[lookup_vector_docs];
  for (size_t quarter=0;quarter<lookup_neon_quarters;++quarter) {
    const size_t at=doc+quarter*lookup_neon_lanes;
    const auto terms=lookup_centroid_terms_neon(cache,q,at);
    auto score=vaddq_f32(terms.centers,vmulq_f32(terms.weights,half_quarter_neon(cache.correction.data()+at)));
    score=vaddq_f32(score,vmulq_f32(half_quarter_neon(cache.factor.data()+at),dot[quarter]));
    vst1q_f32(scores+quarter*lookup_neon_lanes,score);
  }
  std::copy_n(scores,valid,out);
}
// Adds sections [section_begin, section_end) to the dots of Vectors consecutive 16-document vectors of one block;
// base points at the first vector's codes of pair 0.
template<size_t Vectors>
__attribute__((always_inline))
inline void lookup_sections_neon(const FastScanLookupCache& lookup,const FastScanLookupQuery& q,const uint8_t* base,
    size_t section_begin,size_t section_end,float32x4_t (&dots)[Vectors][lookup_neon_quarters],
    size_t stride=lookup_block_docs) {
  const auto mask=vdupq_n_u8(lookup_table_entries-1);
  for (size_t section=section_begin,start=section ? lookup.section_ends[section-1] : 0;section<section_end;
       start=lookup.section_ends[section++]) {
    int16x8_t sums[Vectors][2];
    for (auto& vector:sums) for (auto& sum:vector) sum=vdupq_n_s16(0);
    const size_t end=lookup.section_ends[section];
    for (size_t g=start;g<end;g+=lookup_pair_groups) {
      const auto low_table=vld1q_s8(q.tables.data()+g*lookup_table_entries);
      const auto high_table=vld1q_s8(q.tables.data()+(g+1)*lookup_table_entries);
      const uint8_t* pair=base+g/lookup_pair_groups*stride;
      for (size_t v=0;v<Vectors;++v) {
        const auto packed=vld1q_u8(pair+v*lookup_vector_docs);
        const auto low=vqtbl1q_s8(low_table,vandq_u8(packed,mask));
        const auto high=vqtbl1q_s8(high_table,vshrq_n_u8(packed,lookup_code_bits));
        sums[v][0]=vaddq_s16(sums[v][0],vaddl_s8(vget_low_s8(low),vget_low_s8(high)));
        sums[v][1]=vaddq_s16(sums[v][1],vaddl_high_s8(low,high));
      }
    }
    const auto scale=vdupq_n_f32(q.scales[section]);
    auto add=[&](float32x4_t& dot,int32x4_t sum) { dot=vaddq_f32(dot,vmulq_f32(vcvtq_f32_s32(sum),scale)); };
    for (size_t v=0;v<Vectors;++v) {
      add(dots[v][0],vmovl_s16(vget_low_s16(sums[v][0]))); add(dots[v][1],vmovl_high_s16(sums[v][0]));
      add(dots[v][2],vmovl_s16(vget_low_s16(sums[v][1]))); add(dots[v][3],vmovl_high_s16(sums[v][1]));
    }
  }
}
template<size_t Vectors>
inline void scan_lookup_vectors_neon(const FastScanCache& cache,const FastScanLookupQuery& q,const uint8_t* base,
    size_t doc,size_t end,float* out) {
  float32x4_t dots[Vectors][lookup_neon_quarters];
  for (auto& vector:dots) for (auto& dot:vector) dot=vdupq_n_f32(0);
  lookup_sections_neon<Vectors>(*cache.lookup,q,base,0,cache.lookup->section_ends.size(),dots,
                                cache.lookup->stride(doc/lookup_block_docs));
  for (size_t v=0;v<Vectors && doc+v*lookup_vector_docs<end;++v) {
    const size_t at=doc+v*lookup_vector_docs;
    lookup_epilogue_neon(cache,q,at,dots[v],out+v*lookup_vector_docs,std::min(lookup_vector_docs,end-at));
  }
}
// Splits [first, first + count) into runs of up to lookup_neon_vectors vectors inside one block and calls
// scan(std::integral_constant<size_t, Vectors>, base, doc, end, out) for each; first is aligned to lookup_vector_docs,
// and codes and metadata are padded to whole blocks, so every vector reads complete rows.
template<class Scan>
inline void for_lookup_vector_runs_neon(const FastScanCache& cache,size_t first,size_t count,float* scores,Scan scan) {
  if (first%lookup_vector_docs) throw std::invalid_argument("TieredQ: NEON lookup range start is not vector aligned");
  const auto& lookup=*cache.lookup;
  const size_t end=first+count;
  for (size_t doc=first;doc<end;) {
    const size_t in_block=doc%lookup_block_docs;
    const size_t vectors=std::min({lookup_neon_vectors,(lookup_block_docs-in_block)/lookup_vector_docs,
        (end-doc+lookup_vector_docs-1)/lookup_vector_docs});
    const uint8_t* base=lookup.codes.data()+doc/lookup_block_docs*lookup.pairs()*lookup_block_docs+in_block;
    float* out=scores+(doc-first);
    switch (vectors) {
      case 4: scan(std::integral_constant<size_t,4>{},base,doc,end,out); break;
      case 3: scan(std::integral_constant<size_t,3>{},base,doc,end,out); break;
      case 2: scan(std::integral_constant<size_t,2>{},base,doc,end,out); break;
      default: scan(std::integral_constant<size_t,1>{},base,doc,end,out); break;
    }
    doc+=vectors*lookup_vector_docs;
  }
}
// Scores [first, first + count) into scores.
inline void scan_fast_lookup_neon_range(const FastScanCache& cache,const FastScanLookupQuery& q,
    size_t first,size_t count,float* scores) {
  for_lookup_vector_runs_neon(cache,first,count,scores,[&](auto vectors,const uint8_t* base,size_t doc,size_t end,float* out) {
    scan_lookup_vectors_neon<decltype(vectors)::value>(cache,q,base,doc,end,out);
  });
}
} // namespace tq::tieredq_detail
