// AVX-512 FP32 grouped scan over packed byte planes (16 documents per block, plane b at block + 16 b).
// Each 32-bit code window of 16 documents is one 64-byte load transposed into doc-major dwords, so a group costs
// one shift and one register permute instead of a byte widen per coordinate (tieredq_scan_groups.h).
#pragma once
#include "tieredq_layout.h"
#include "tieredq_scan_groups.h"
#include <immintrin.h>
#include <stdexcept>

namespace dv::tieredq_detail {
constexpr uint32_t window_bits = 32;
constexpr size_t block_docs = 16, planes_per_window = 4;
constexpr uint32_t register_book_bits = 6;  // codebooks up to 64 levels live in four registers
// A batch spans several blocks, more than L1 holds: pull each block's upcoming windows in ahead of use.
constexpr uint32_t prefetch_windows = 4;

// Decode one fp16 metadata field for 16 documents; plane bytes are interleaved into half words.
template<bool transposed>
__attribute__((target("avx512f,avx2"))) inline void half16(const uint8_t* base,size_t stride,size_t offset,float* out) {
  __m256i packed;
  if constexpr (transposed) {
    const auto lo=_mm_loadu_si128(reinterpret_cast<const __m128i*>(base+offset*16));
    const auto hi=_mm_loadu_si128(reinterpret_cast<const __m128i*>(base+(offset+1)*16));
    packed=_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(lo,hi)),_mm_unpackhi_epi8(lo,hi),1);
  } else {
    alignas(32) uint16_t values[16];
    for (size_t i=0;i<16;++i) values[i]=uint16_t(base[i*stride+offset]) | uint16_t(base[i*stride+offset+1])<<8;
    packed=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }
  _mm512_storeu_ps(out,_mm512_cvtph_ps(packed));
}

// Code bytes [4 window, 4 window + 4) of 16 documents as one little-endian dword per document.
__attribute__((target("avx512f,avx512bw"),always_inline)) inline __m512i code_window(const uint8_t* block,uint32_t window) {
  const __m512i planes=_mm512_loadu_si512(block+size_t(window)*planes_per_window*block_docs);
  const __m512i lanes=_mm512_permutexvar_epi32(_mm512_setr_epi32(0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15),planes);
  return _mm512_shuffle_epi8(lanes,_mm512_broadcast_i32x4(_mm_setr_epi8(0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15)));
}

// Exact codebook level for 16 codes: vpermt2ps picks within 32-level slices (index bits 0-4), then a blend tree
// on bits 5.. selects the slice. Higher garbage bits are ignored, so callers never mask.
template<uint32_t bits>
__attribute__((target("avx512f"),always_inline)) inline __m512 level_tree(const float* levels,__m512i ids) {
  constexpr uint32_t slices=(1u<<bits)/long_table_entries;
  __m512 level[slices];
  for (uint32_t s=0;s<slices;++s)
    level[s]=_mm512_permutex2var_ps(_mm512_loadu_ps(levels+s*long_table_entries),ids,
                                    _mm512_loadu_ps(levels+s*long_table_entries+short_table_entries));
  for (uint32_t width=1,bit=long_table_entries;width<slices;width*=2,bit*=2) {
    const __mmask16 high=_mm512_test_epi32_mask(ids,_mm512_set1_epi32(int(bit)));
    for (uint32_t s=0;s<slices;s+=2*width) level[s]=_mm512_mask_blend_ps(high,level[s],level[s+width]);
  }
  return level[0];
}

// only. Windows stay inside each document's stride because metadata follows the code. Returns rows scored.
template<int batches>
__attribute__((target("avx512f,avx512bw,avx2,fma"))) size_t scan_grouped(const uint8_t* payload,size_t count,size_t stride,
    size_t code_bytes,const TieredQQuery& q,const Layout& layout,uint32_t centroids,
    const std::vector<std::vector<float>>& levels,const std::vector<ScanGroupPlan>& plan,float* scores) {
  constexpr uint32_t pf=prefetch_windows;
  const size_t block_bytes=block_docs*stride;
  const float* book6=levels[register_book_bits].data();
  const __m512 book[4]={_mm512_loadu_ps(book6),_mm512_loadu_ps(book6+16),_mm512_loadu_ps(book6+32),_mm512_loadu_ps(book6+48)};
  const __m512i high_half=_mm512_set1_epi32(int(long_table_entries));
  const __m512i centroid_limit=_mm512_set1_epi32(int(centroids));
  __m512i word[batches],next[batches];
  for (int b=0;b<batches;++b) word[b]=next[b]=_mm512_setzero_si512();
  size_t begin=0;
  for (;count-begin>=block_docs*batches;begin+=block_docs*batches) {
    const uint8_t* base=payload+begin*stride;
    __m512 sums[batches];
    for (int b=0;b<batches;++b) sums[b]=_mm512_setzero_ps();
    uint32_t loaded=UINT32_MAX; bool has_next=false;
    for (const auto& g:plan) {
      const uint32_t window=g.window, shift=g.shift;
      if (window!=loaded) {
        const bool advance=has_next && window==loaded+1;
        for (int b=0;b<batches;++b) {
          word[b]=advance ? next[b] : code_window(base+b*block_bytes,window);
          _mm_prefetch(reinterpret_cast<const char*>(base+b*block_bytes+size_t(window+pf)*planes_per_window*block_docs),_MM_HINT_T0);
        }
        loaded=window; has_next=false;
      }
      const bool straddles=g.straddles;
      if (straddles && !has_next) {
        for (int b=0;b<batches;++b) next[b]=code_window(base+b*block_bytes,window+1);
        has_next=true;
      }
      const __m512i right=_mm512_set1_epi32(int(shift)), left=_mm512_set1_epi32(int(g.left));
      __m512i ids[batches];
      for (int b=0;b<batches;++b) {
        ids[b]=_mm512_srlv_epi32(word[b],right);
        if (straddles) ids[b]=_mm512_or_si512(ids[b],_mm512_sllv_epi32(next[b],left));
      }
      if (g.table!=TieredQScanGroup::wide) {
        const float* table=q.scan_tables.data()+g.table;
        const __m512 lo=_mm512_loadu_ps(table);
        if (g.bits<=short_table_bits) {
          for (int b=0;b<batches;++b) sums[b]=_mm512_add_ps(sums[b],_mm512_permutexvar_ps(ids[b],lo));
        } else {
          const __m512 hi=_mm512_loadu_ps(table+short_table_entries);
          for (int b=0;b<batches;++b) sums[b]=_mm512_add_ps(sums[b],_mm512_permutex2var_ps(lo,ids[b],hi));
        }
        continue;
      }
      const __m512 weight=_mm512_set1_ps(q.weights[g.first]);
      if (g.bits==register_book_bits) {
        for (int b=0;b<batches;++b) {
          const __m512 lo=_mm512_permutex2var_ps(book[0],ids[b],book[1]),hi=_mm512_permutex2var_ps(book[2],ids[b],book[3]);
          const __m512 level=_mm512_mask_blend_ps(_mm512_test_epi32_mask(ids[b],high_half),lo,hi);
          sums[b]=_mm512_fmadd_ps(level,weight,sums[b]);
        }
      } else if (g.bits==register_book_bits+1) {
        for (int b=0;b<batches;++b)
          sums[b]=_mm512_fmadd_ps(level_tree<register_book_bits+1>(levels[g.bits].data(),ids[b]),weight,sums[b]);
      } else {
        // 256 levels would take eight port-5 permutes, which hyperthread pairs share; gathers use the load ports.
        const __m512i mask=_mm512_set1_epi32(int((1u<<g.bits)-1));
        const float* levels_g=levels[g.bits].data();
        for (int b=0;b<batches;++b)
          sums[b]=_mm512_fmadd_ps(_mm512_i32gather_ps(_mm512_and_si512(ids[b],mask),levels_g,4),weight,sums[b]);
      }
    }
    // Same operation order as the scalar scorer: centroid score, + weighted correction, + factor * sum.
    for (int b=0;b<batches;++b) {
      const uint8_t* docs=base+b*block_bytes;
      const __m512i cid=_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(docs+code_bytes*block_docs)));
      if (_mm512_cmpge_epu32_mask(cid,centroid_limit)) throw std::runtime_error("TieredQ: invalid centroid id");
      alignas(64) float factors[block_docs],corrections[block_docs];
      half16<true>(docs,stride,layout.factor_offset(),factors);
      half16<true>(docs,stride,layout.correction_offset(),corrections);
      __m512 score=_mm512_i32gather_ps(cid,q.centroid_scores.data(),4);
      score=_mm512_add_ps(score,_mm512_mul_ps(_mm512_i32gather_ps(cid,q.correction_weights.data(),4),_mm512_load_ps(corrections)));
      score=_mm512_add_ps(score,_mm512_mul_ps(_mm512_load_ps(factors),sums[b]));
      _mm512_storeu_ps(scores+begin+b*block_docs,score);
    }
  }
  return begin;
}
} // namespace dv::tieredq_detail
