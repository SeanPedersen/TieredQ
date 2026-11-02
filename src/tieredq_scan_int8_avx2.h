// AVX2 fast scans over the persisted lookup and int8-level layouts, with bounded tail stores.
#pragma once
#include "tieredq_scan_query_x86.h"
#include "tieredq_scan_codes_x86.h"
#include <immintrin.h>

namespace dv::tieredq_detail {
constexpr size_t fast_scan_avx2_lanes=8,fast_lookup_avx2_docs=32;
constexpr size_t fast_scan_avx2_group_bytes=fast_scan_block_docs*sizeof(int32_t);

__attribute__((target("avx2,f16c")))
inline void fast_epilogue_avx2(const FastScanCache& cache,const std::vector<float>& centers,
    const std::vector<float>& corrections,size_t doc,__m256 dot,float* out,size_t count) {
  const auto cid=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(cache.centroid.data()+doc)));
  auto score=_mm256_i32gather_ps(centers.data(),cid,sizeof(float));
  score=_mm256_add_ps(score,_mm256_mul_ps(_mm256_i32gather_ps(corrections.data(),cid,sizeof(float)),
      _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cache.correction.data()+doc)))));
  score=_mm256_add_ps(score,_mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cache.factor.data()+doc))),dot));
  if (count>=fast_scan_avx2_lanes) { _mm256_storeu_ps(out,score); return; }
  alignas(32) float tail[fast_scan_avx2_lanes];
  _mm256_store_ps(tail,score);
  std::copy_n(tail,count,out);
}

// Range starts are aligned to fast_lookup_avx2_docs (a chunk never crosses a lookup block); the final chunk may be partial.
__attribute__((target("avx2,f16c")))
inline void scan_fast_lookup_avx2_range(const FastScanCache& cache,const FastScanLookupQuery& q,
    size_t first,size_t count,float* scores) {
  const auto& lookup=*cache.lookup;
  const auto mask=_mm256_set1_epi8(lookup_table_entries-1),ones=_mm256_set1_epi8(1);
  constexpr size_t vectors=fast_lookup_avx2_docs/fast_scan_avx2_lanes;
  for (size_t doc=first;doc<first+count;doc+=fast_lookup_avx2_docs) {
    __m256 dots[vectors];
    for (auto& dot:dots) dot=_mm256_setzero_ps();
    const auto* base=lookup.codes.data()+doc/lookup_block_docs*lookup.pairs()*lookup_block_docs+doc%lookup_block_docs;
    for (size_t section=0,start=0;section<lookup.section_ends.size();start=lookup.section_ends[section++]) {
      auto low=_mm256_setzero_si256(),high=_mm256_setzero_si256();
      const size_t end=lookup.section_ends[section];
      for (size_t g=start;g<end;g+=lookup_pair_groups) {
        const auto t0=_mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q.tables.data()+g*lookup_table_entries)));
        const auto t1=_mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q.tables.data()+(g+1)*lookup_table_entries)));
        const auto packed=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(base+g/lookup_pair_groups*lookup_block_docs));
        const auto v0=_mm256_shuffle_epi8(t0,_mm256_and_si256(packed,mask));
        const auto v1=_mm256_shuffle_epi8(t1,_mm256_and_si256(_mm256_srli_epi16(packed,lookup_code_bits),mask));
        low=_mm256_add_epi16(low,_mm256_maddubs_epi16(ones,_mm256_unpacklo_epi8(v0,v1)));
        high=_mm256_add_epi16(high,_mm256_maddubs_epi16(ones,_mm256_unpackhi_epi8(v0,v1)));
      }
      const auto scale=_mm256_set1_ps(q.scales[section]);
      // Unpack is lane-local: low contains documents 0..7 and 16..23, high the other two vectors.
      const __m128i sums[vectors]={_mm256_castsi256_si128(low),_mm256_castsi256_si128(high),
          _mm256_extracti128_si256(low,1),_mm256_extracti128_si256(high,1)};
      for (size_t v=0;v<vectors;++v)
        dots[v]=_mm256_add_ps(dots[v],_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sums[v])),scale));
    }
    for (size_t v=0;v<vectors && doc+v*fast_scan_avx2_lanes<first+count;++v) {
      const size_t begin=doc+v*fast_scan_avx2_lanes;
      fast_epilogue_avx2(cache,q.centroid_scores,q.correction_weights,begin,dots[v],scores+begin-first,
          std::min(fast_scan_avx2_lanes,first+count-begin));
    }
  }
}

constexpr size_t fast_scan_query_tile_avx2=8;
template<size_t query_tile>
__attribute__((target("avx2,f16c")))
inline void scan_fast_avx2_tile_impl(const FastScanCache& cache,const FastScanQueryX86* queries,
    size_t rows,size_t first,size_t count,float* scores,size_t stride) {
  if (cache.code_lookup()) {
    scan_codes_avx2_tile(cache,rows,first,count,scores,stride,
        [&](size_t row)->const TieredQQuery& {return queries[row].codes;});
    return;
  }
  if (cache.lookup) {
    scan_lookup_query_group(rows,first,count,scores,stride,[&](size_t row,size_t begin,size_t documents,float* out) {
      scan_fast_lookup_avx2_range(cache,queries[row].lookup,begin,documents,out);
    });
    return;
  }
  const auto ones=_mm256_set1_epi16(1),bias=_mm256_set1_epi8(char(cache.level_max+1));
  for (size_t doc=first;doc<first+count;doc+=fast_scan_avx2_lanes) {
    __m256i sums[query_tile];
    for (size_t row=0;row<rows;++row) sums[row]=_mm256_setzero_si256();
    const auto* base=cache.levels.data()+doc/fast_scan_block_docs*cache.groups*fast_scan_avx2_group_bytes+doc%fast_scan_block_docs*sizeof(int32_t);
    for (size_t g=0;g<cache.groups;++g) {
      const auto levels=_mm256_add_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(base+g*fast_scan_avx2_group_bytes)),bias);
      for (size_t row=0;row<rows;++row)
        sums[row]=_mm256_add_epi32(sums[row],_mm256_madd_epi16(
            _mm256_maddubs_epi16(levels,_mm256_set1_epi32(queries[row].qwords[g])),ones));
    }
    for (size_t row=0;row<rows;++row) {
      const auto& q=queries[row];
      const auto dot=_mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(sums[row]),_mm256_set1_ps(q.offset)),_mm256_set1_ps(q.scale));
      fast_epilogue_avx2(cache,q.centroid_scores,q.correction_weights,doc,dot,scores+row*stride+doc-first,
          std::min(fast_scan_avx2_lanes,first+count-doc));
    }
  }
}
inline void scan_fast_avx2_tile(const FastScanCache& cache,const FastScanQueryX86* queries,
    size_t rows,size_t first,size_t count,float* scores,size_t stride) {
  if (rows==1) scan_fast_avx2_tile_impl<1>(cache,queries,rows,first,count,scores,stride);
  else scan_fast_avx2_tile_impl<fast_scan_query_tile_avx2>(cache,queries,rows,first,count,scores,stride);
}
} // namespace dv::tieredq_detail
