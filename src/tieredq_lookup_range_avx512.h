// AVX-512 nibble scan with the prefix-section range checkpoint: chunks whose 64 documents all fail the bound skip
// their suffix sections and score -inf; every other score equals the unpruned nibble scan bit for bit.
#pragma once
#include "tieredq_lookup_range.h"
#include "tieredq_range_checkpoint_avx512.h"
#include "tieredq_scan_lookup_avx512.h"

namespace dv::tieredq_detail {
// Whether any of the vector's valid documents can still reach the threshold after the prefix dot product.
__attribute__((target("avx512f,avx512bw"),always_inline))
inline bool lookup_vector_survives(const FastScanCache& c,const FastScanLookupQuery& q,const LookupRangeCheckpoint& checkpoint,
    size_t doc,__m512 dot,size_t valid) {
  const auto& b=checkpoint.bounds.bounds;
  const auto cid=_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c.centroid.data()+doc)));
  const auto correction=_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(c.correction.data()+doc)));
  const auto factor=_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(c.factor.data()+doc)));
  const auto base=_mm512_add_round_ps(_mm512_i32gather_ps(cid,q.centroid_scores.data(),sizeof(float)),
      _mm512_mul_round_ps(_mm512_i32gather_ps(cid,q.correction_weights.data(),sizeof(float)),correction,range_rounding_up),range_rounding_up);
  const auto sign_mask=_mm512_set1_epi32(INT32_MAX);
  const auto abs_factor=_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(factor),sign_mask));
  const auto abs_base=_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(base),sign_mask));
  const auto partial=_mm512_add_round_ps(base,_mm512_mul_round_ps(factor,dot,range_rounding_up),range_rounding_up);
  // Negative factors require the outward-rounded lower endpoint of the suffix range.
  const auto range=_mm512_mask_blend_ps(_mm512_cmp_ps_mask(factor,_mm512_setzero_ps(),_CMP_GE_OQ),
      _mm512_set1_ps(b.low),_mm512_set1_ps(b.high));
  auto upper=_mm512_add_round_ps(partial,_mm512_mul_round_ps(factor,range,range_rounding_up),range_rounding_up);
  const auto allowance=_mm512_mul_round_ps(_mm512_set1_ps(b.allowance_scale),
      _mm512_add_round_ps(abs_base,_mm512_mul_round_ps(abs_factor,_mm512_set1_ps(b.magnitude),range_rounding_up),
          range_rounding_up),range_rounding_up);
  upper=_mm512_add_round_ps(upper,allowance,range_rounding_up);
  const __mmask16 lanes=valid>=lookup_vector_docs ? __mmask16(0xffff) : __mmask16((1u<<valid)-1);
  return _mm512_mask_cmp_ps_mask(lanes,upper,_mm512_set1_ps(checkpoint.threshold),_CMP_GE_OQ)!=0;
}
template<size_t Chunks>
__attribute__((target("avx512f,avx512bw")))
inline void scan_range_lookup_chunks_avx512(const FastScanCache& c,const FastScanLookupQuery& q,size_t block,
    size_t first_chunk,size_t first,size_t count,float* out,const LookupRangeCheckpoint& checkpoint) {
  const auto& lookup=*c.lookup;
  const auto* base=lookup.codes.data()+block*lookup.pairs()*lookup_block_docs;
  const size_t prefix=checkpoint.bounds.prefix_sections,sections=lookup.section_ends.size();
  __m512 dots[Chunks][4];
  for(size_t chunk=0;chunk<Chunks;++chunk)for(auto& dot:dots[chunk])dot=_mm512_setzero_ps();
  lookup_sections_avx512<Chunks>(lookup,q,base,first_chunk,0,prefix,dots);
  auto document=[&](size_t chunk,size_t t) { return block*lookup_block_docs+(first_chunk+chunk)*lookup_chunk_docs+t*lookup_vector_docs; };
  auto in_range=[&](size_t doc) { return doc>=first && doc<first+count; };
  bool live[Chunks]; size_t survivors=0;
  for(size_t chunk=0;chunk<Chunks;++chunk){
    live[chunk]=false;
    for(size_t t=0;t<4 && !live[chunk];++t){
      const size_t doc=document(chunk,t);
      if(in_range(doc))live[chunk]=lookup_vector_survives(c,q,checkpoint,doc,dots[chunk][t],first+count-doc);
    }
    survivors+=live[chunk];
  }
  if(survivors==Chunks)lookup_sections_avx512<Chunks>(lookup,q,base,first_chunk,prefix,sections,dots);
  else for(size_t chunk=0;chunk<Chunks;++chunk)if(live[chunk]){
    __m512 single[1][4]={{dots[chunk][0],dots[chunk][1],dots[chunk][2],dots[chunk][3]}};
    lookup_sections_avx512<1>(lookup,q,base,first_chunk+chunk,prefix,sections,single);
    for(size_t t=0;t<4;++t)dots[chunk][t]=single[0][t];
  }
  const auto rejected=_mm512_set1_ps(-std::numeric_limits<float>::infinity());
  for(size_t chunk=0;chunk<Chunks;++chunk)for(size_t t=0;t<4;++t){
    const size_t doc=document(chunk,t);
    if(!in_range(doc))continue;
    if(live[chunk])lookup_epilogue(c,q,doc,dots[chunk][t],out,first,first+count);
    else{
      const size_t n=std::min(lookup_vector_docs,first+count-doc);
      _mm512_mask_storeu_ps(out+doc-first,n==lookup_vector_docs ? 0xffff : (1u<<n)-1,rejected);
    }
  }
}
// The pruned counterpart of scan_fast_lookup_avx512_range; first is aligned to lookup_vector_docs.
__attribute__((target("avx512f,avx512bw")))
inline void scan_range_lookup_avx512(const FastScanCache& c,const FastScanLookupQuery& q,size_t first,size_t count,
    float* out,const LookupRangeCheckpoint& checkpoint) {
  constexpr size_t chunks=lookup_block_docs/lookup_chunk_docs;
  for (size_t b=first/lookup_block_docs;b<(first+count+lookup_block_docs-1)/lookup_block_docs;++b) {
    const size_t block_first=b*lookup_block_docs;
    const size_t chunk_begin=std::max(first,block_first)%lookup_block_docs/lookup_chunk_docs;
    const size_t chunk_end=(std::min(first+count,block_first+lookup_block_docs)-block_first+lookup_chunk_docs-1)/lookup_chunk_docs;
    switch (chunk_end-chunk_begin) {
      case chunks: scan_range_lookup_chunks_avx512<chunks>(c,q,b,0,first,count,out,checkpoint); break;
      case 3: scan_range_lookup_chunks_avx512<3>(c,q,b,chunk_begin,first,count,out,checkpoint); break;
      case 2: scan_range_lookup_chunks_avx512<2>(c,q,b,chunk_begin,first,count,out,checkpoint); break;
      default: scan_range_lookup_chunks_avx512<1>(c,q,b,chunk_begin,first,count,out,checkpoint); break;
    }
  }
}
} // namespace dv::tieredq_detail
