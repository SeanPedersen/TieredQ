// AVX-512BW byte-table full scan: two shuffles score a pair of groups for 64 documents.
// Pair sums and scaled sections fit signed int16; conversion to float occurs once per section.
#pragma once
#include "tieredq_scan_lookup_x86.h"

namespace dv::tieredq_detail {
__attribute__((target("avx512f,avx512bw")))
inline void lookup_epilogue(const FastScanCache& c, const FastScanLookupQuery& q, size_t first, __m512 dot, float* out) {
  const auto cid = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c.centroid.data() + first)));
  auto score = _mm512_i32gather_ps(cid, q.centroid_scores.data(), sizeof(float));
  score = _mm512_add_ps(score, _mm512_mul_ps(_mm512_i32gather_ps(cid, q.correction_weights.data(), sizeof(float)), _mm512_loadu_ps(c.correction.data() + first)));
  score = _mm512_add_ps(score, _mm512_mul_ps(_mm512_loadu_ps(c.factor.data() + first), dot));
  const size_t n = std::min(lookup_vector_docs, c.count - first);
  _mm512_mask_storeu_ps(out + first, n == lookup_vector_docs ? 0xffff : (1u << n) - 1, score);
}
__attribute__((target("avx512f,avx512bw")))
inline void scan_fast_lookup_avx512(const FastScanCache& c, const FastScanLookupQuery& q, float* out) {
  const auto& lookup=*c.lookup;
  constexpr size_t chunks = lookup_block_docs / lookup_chunk_docs;
  const size_t pairs = lookup.pairs();
  const auto mask = _mm512_set1_epi8(lookup_table_entries-1), ones = _mm512_set1_epi8(1);
  const auto order0 = _mm512_setr_epi64(0,1,8,9,2,3,10,11);
  const auto order1 = _mm512_setr_epi64(4,5,12,13,6,7,14,15);
  for (size_t b = 0; b < c.blocks; ++b) {
    __m512 dots[chunks][4];
    for(size_t chunk=0;chunk<chunks;++chunk)for(auto& dot:dots[chunk])dot=_mm512_setzero_ps();
    const uint8_t* base = lookup.codes.data()+b*pairs*lookup_block_docs;
    for (size_t start = 0; start < lookup.groups.size(); start += lookup_scale_groups) {
      __m512i sum[chunks][2];
      for(size_t chunk=0;chunk<chunks;++chunk)for(auto& x:sum[chunk])x=_mm512_setzero_si512();
      const size_t end = std::min(start+lookup_scale_groups,lookup.groups.size());
      for (size_t g = start; g < end; g += lookup_pair_groups) {
        const auto t0 = _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q.tables.data()+g*lookup_table_entries)));
        const auto t1 = _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q.tables.data()+(g+1)*lookup_table_entries)));
#pragma GCC unroll 4
        for(size_t chunk=0;chunk<chunks;++chunk){
          const auto packed = _mm512_loadu_si512(base+g/lookup_pair_groups*lookup_block_docs+chunk*lookup_chunk_docs);
          const auto v0 = _mm512_shuffle_epi8(t0,_mm512_and_si512(packed,mask));
          const auto v1 = _mm512_shuffle_epi8(t1,_mm512_and_si512(_mm512_srli_epi16(packed,lookup_code_bits),mask));
          sum[chunk][0] = _mm512_add_epi16(sum[chunk][0],_mm512_maddubs_epi16(ones,_mm512_unpacklo_epi8(v0,v1)));
          sum[chunk][1] = _mm512_add_epi16(sum[chunk][1],_mm512_maddubs_epi16(ones,_mm512_unpackhi_epi8(v0,v1)));
        }
      }
      const auto scale = _mm512_set1_ps(q.scales[start/lookup_scale_groups]);
      // Byte unpacking is lane-local: restore document order before widening the section sums.
#pragma GCC unroll 4
      for(size_t chunk=0;chunk<chunks;++chunk){
        const auto a = _mm512_permutex2var_epi64(sum[chunk][0],order0,sum[chunk][1]);
        const auto z = _mm512_permutex2var_epi64(sum[chunk][0],order1,sum[chunk][1]);
        dots[chunk][0] = _mm512_add_ps(dots[chunk][0],_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(a))),scale));
        dots[chunk][1] = _mm512_add_ps(dots[chunk][1],_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(a,1))),scale));
        dots[chunk][2] = _mm512_add_ps(dots[chunk][2],_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(z))),scale));
        dots[chunk][3] = _mm512_add_ps(dots[chunk][3],_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(z,1))),scale));
      }
    }
    for(size_t chunk=0;chunk<chunks;++chunk)for(size_t t=0;t<4;++t){
      const size_t first=b*lookup_block_docs+chunk*lookup_chunk_docs+t*lookup_vector_docs;
      if(first<c.count)lookup_epilogue(c,q,first,dots[chunk][t],out);
    }
  }
}

} // namespace dv::tieredq_detail
