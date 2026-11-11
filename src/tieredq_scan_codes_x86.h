// x86 packed-code scans without requantizing codebooks or query weights.
// AVX-512 unpacks each coordinate's codes from the interleaved 32-bit words with shifts and decodes them from
// register-resident codebooks: vpermt2ps resolves the low five code bits within a 32-entry register pair
// and mask blends select pairs. This avoids vgatherdps, which is microcoded under Gather Data Sampling
// mitigations. AVX2 decodes with scalar loads.
#pragma once
#include "tieredq_scan_codes.h"
#include <immintrin.h>
#include <numeric>
#include <limits>
#include <utility>

namespace tq::tieredq_detail {
constexpr size_t code_register_lanes=16,code_pair_entries=2*code_register_lanes,code_pair_bits=5;
constexpr size_t fast_code_query_tile_avx512=8,fast_code_query_tile_avx2=4;

constexpr size_t code_table_pairs(size_t width) {
  return width>code_pair_bits ? size_t(1)<<(width-code_pair_bits) : 1;
}
// Contiguous coordinates sharing one tier. Narrow codebooks repeat across the register pair, so code
// bits above the tier width (neighbouring codes) select the same level; wider tables are exact.
struct CodeTableSegment { size_t begin,end,width,offset; std::vector<float> table; };
inline std::vector<CodeTableSegment> code_table_segments(const FastScanCache& cache) {
  std::vector<CodeTableSegment> segments;
  for (size_t rank=0;rank<cache.dimensions;) {
    size_t end=rank;
    while (end<cache.dimensions && cache.tier_of[end]==cache.tier_of[rank]) ++end;
    const auto& book=cache.codebooks[cache.tier_of[rank]];
    const size_t width=fast_code_width(cache,rank);
    if (width>fast_scan_code_bits) throw std::runtime_error("TieredQ: code table exceeds byte codes");
    std::vector<float> table(code_table_pairs(width)*code_pair_entries);
    for (size_t entry=0;entry<table.size();++entry) table[entry]=book[entry%book.size()];
    segments.push_back({rank,end,width,cache.code_offset[rank],std::move(table)});
    rank=end;
  }
  return segments;
}

template<size_t Pairs>
__attribute__((target("avx512f,avx512bw"),always_inline))
inline __m512 table_lookup_avx512(const __m512* table,__m512i code) {
  __m512 level[Pairs];
  for (size_t pair=0;pair<Pairs;++pair) level[pair]=_mm512_permutex2var_ps(table[2*pair],code,table[2*pair+1]);
  size_t live=Pairs;
  for (size_t bit=code_pair_bits;live>1;++bit,live/=2) {
    const __mmask16 high=_mm512_test_epi32_mask(code,_mm512_set1_epi32(int(1u<<bit)));
    for (size_t pair=0;pair<live/2;++pair) level[pair]=_mm512_mask_blend_ps(high,level[2*pair],level[2*pair+1]);
  }
  return level[0];
}
// Codes at a runtime bit offset: low word shifted down, plus the next word when the code straddles.
template<size_t Width>
__attribute__((target("avx512f,avx512bw"),always_inline))
inline __m512i load_codes_avx512(const int8_t* base,size_t offset) {
  const size_t shift=offset%fast_code_word_bits;
  const int8_t* low=base+offset/fast_code_word_bits*fast_scan_group_bytes;
  const __m512i code=_mm512_srlv_epi32(_mm512_loadu_si512(low),_mm512_set1_epi32(int(shift)));
  if (shift+Width<=fast_code_word_bits) return code;
  return _mm512_or_si512(code,_mm512_sllv_epi32(_mm512_loadu_si512(low+fast_scan_group_bytes),
      _mm512_set1_epi32(int(fast_code_word_bits-shift))));
}
// Code `Index` of a period that starts at bit `Phase` of words[0]; all shifts are immediates.
template<size_t Width,size_t Phase,size_t Index>
__attribute__((target("avx512f,avx512bw"),always_inline))
inline __m512i period_codes_avx512(const __m512i* words) {
  constexpr size_t bit=Phase+Index*Width,word=bit/fast_code_word_bits,shift=bit%fast_code_word_bits;
  if constexpr (!shift) return words[word];
  else if constexpr (shift+Width<=fast_code_word_bits) return _mm512_srli_epi32(words[word],shift);
  else return _mm512_or_si512(_mm512_srli_epi32(words[word],shift),_mm512_slli_epi32(words[word+1],fast_code_word_bits-shift));
}
// Blocks consecutive 16-document blocks (block_bytes apart) walk together: each rank's lookups for all blocks are
// independent, so their accumulation chains overlap; every block sees the same per-rank sequence as alone.
template<size_t Width,size_t Phase,size_t Pairs,size_t Index,size_t Blocks,size_t Words,class Sink>
__attribute__((target("avx512f,avx512bw"),always_inline))
inline void walk_period_rank_avx512(const __m512* table,const __m512i (&words)[Blocks][Words],size_t rank,Sink& sink) {
  for (size_t block=0;block<Blocks;++block)
    sink(rank+Index,block,table_lookup_avx512<Pairs>(table,period_codes_avx512<Width,Phase,Index>(words[block])));
}
template<size_t Width,size_t Phase,size_t Pairs,size_t Blocks,size_t Words,class Sink,size_t... Index>
__attribute__((target("avx512f,avx512bw"),always_inline))
inline void walk_period_blocks_avx512(const __m512* table,const __m512i (&words)[Blocks][Words],size_t rank,Sink& sink,std::index_sequence<Index...>) {
  (walk_period_rank_avx512<Width,Phase,Pairs,Index>(table,words,rank,sink),...);
}
template<size_t Width,size_t Pairs,size_t Blocks,class Sink>
__attribute__((target("avx512f,avx512bw"),always_inline))
inline void walk_rank_blocks_avx512(const __m512* table,const int8_t* base,size_t block_bytes,size_t rank,size_t offset,Sink& sink) {
  for (size_t block=0;block<Blocks;++block)
    sink(rank,block,table_lookup_avx512<Pairs>(table,load_codes_avx512<Width>(base+block*block_bytes,offset)));
}
// Walks one tier segment of Blocks 16-document blocks, handing each coordinate's FP32 levels to `sink` in rank
// order. Ranks between bit offsets congruent to Phase mod 32 form periods whose code positions are compile-time.
template<size_t Width,size_t Phase,size_t Blocks,class Sink>
__attribute__((target("avx512f,avx512bw"),always_inline))
inline void walk_segment_avx512(const int8_t* base,size_t block_bytes,const CodeTableSegment& segment,Sink& sink) {
  constexpr size_t pairs=code_table_pairs(Width),period_bits=std::lcm(Width,fast_code_word_bits);
  constexpr size_t period_ranks=period_bits/Width,period_words=(Phase+period_bits+fast_code_word_bits-1)/fast_code_word_bits;
  __m512 table[2*pairs];
  for (size_t i=0;i<2*pairs;++i) table[i]=_mm512_loadu_ps(segment.table.data()+i*code_register_lanes);
  size_t rank=segment.begin,offset=segment.offset;
  for (;rank<segment.end && offset%fast_code_word_bits!=Phase;++rank,offset+=Width)
    walk_rank_blocks_avx512<Width,pairs,Blocks>(table,base,block_bytes,rank,offset,sink);
  for (;rank+period_ranks<=segment.end;rank+=period_ranks,offset+=period_bits) {
    const int8_t* low=base+offset/fast_code_word_bits*fast_scan_group_bytes;
    __m512i words[Blocks][period_words];
    for (size_t block=0;block<Blocks;++block) for (size_t word=0;word<period_words;++word)
      words[block][word]=_mm512_loadu_si512(low+block*block_bytes+word*fast_scan_group_bytes);
    walk_period_blocks_avx512<Width,Phase,pairs,Blocks>(table,words,rank,sink,std::make_index_sequence<period_ranks>{});
  }
  for (;rank<segment.end;++rank,offset+=Width) walk_rank_blocks_avx512<Width,pairs,Blocks>(table,base,block_bytes,rank,offset,sink);
}
// Bit offsets of one segment share a residue mod gcd(Width, 32); that residue selects the period phase.
template<size_t Width,size_t Blocks,size_t Phase=0,class Sink>
__attribute__((target("avx512f,avx512bw"),always_inline))
inline void walk_segment_phase_avx512(const int8_t* base,size_t block_bytes,const CodeTableSegment& segment,Sink& sink) {
  constexpr size_t phases=std::gcd(Width,fast_code_word_bits);
  if constexpr (Phase+1<phases) if (segment.offset%phases!=Phase) return walk_segment_phase_avx512<Width,Blocks,Phase+1>(base,block_bytes,segment,sink);
  walk_segment_avx512<Width,Phase,Blocks>(base,block_bytes,segment,sink);
}
template<size_t Blocks=1,class Sink>
__attribute__((target("avx512f,avx512bw"),always_inline))
inline void walk_code_segment_avx512(const int8_t* base,size_t block_bytes,const CodeTableSegment& segment,Sink& sink) {
  switch (segment.width) {
    case 1: walk_segment_phase_avx512<1,Blocks>(base,block_bytes,segment,sink); break;
    case 2: walk_segment_phase_avx512<2,Blocks>(base,block_bytes,segment,sink); break;
    case 3: walk_segment_phase_avx512<3,Blocks>(base,block_bytes,segment,sink); break;
    case 4: walk_segment_phase_avx512<4,Blocks>(base,block_bytes,segment,sink); break;
    case 5: walk_segment_phase_avx512<5,Blocks>(base,block_bytes,segment,sink); break;
    case 6: walk_segment_phase_avx512<6,Blocks>(base,block_bytes,segment,sink); break;
    case 7: walk_segment_phase_avx512<7,Blocks>(base,block_bytes,segment,sink); break;
    default: walk_segment_phase_avx512<fast_scan_code_bits,Blocks>(base,block_bytes,segment,sink); break;
  }
}
template<size_t Blocks=1,class Sink>
__attribute__((target("avx512f,avx512bw"),always_inline))
inline void walk_code_block_avx512(const FastScanCache& cache,const std::vector<CodeTableSegment>& segments,
    size_t block,Sink& sink) {
  const int8_t* base=cache.levels.data()+block*cache.level_block_bytes();
  for (const auto& segment:segments) walk_code_segment_avx512<Blocks>(base,cache.level_block_bytes(),segment,sink);
}
struct DecodeLevelsAvx512 {
  float* decoded;
  __attribute__((target("avx512f,avx512bw"),always_inline)) void operator()(size_t rank,size_t,__m512 levels) {
    _mm512_storeu_ps(decoded+rank*fast_scan_block_docs,levels);
  }
};
// A single query consumes levels directly instead of storing and reloading a decoded block; one dot per block.
template<size_t Blocks=1>
struct AccumulateLevelsAvx512 {
  const float* weights;
  __m512 dot[Blocks];
  __attribute__((target("avx512f,avx512bw"),always_inline)) void operator()(size_t rank,size_t block,__m512 levels) {
    dot[block]=_mm512_add_ps(dot[block],_mm512_mul_ps(levels,_mm512_set1_ps(weights[rank])));
  }
};
// Single-query blocks walked together (independent accumulation chains).
constexpr size_t code_interleaved_blocks=4;
__attribute__((target("avx512f,avx512bw")))
inline void decode_code_block_avx512(const FastScanCache& cache,const std::vector<CodeTableSegment>& segments,
    size_t block,float* decoded) {
  DecodeLevelsAvx512 sink{decoded};
  walk_code_block_avx512(cache,segments,block,sink);
}

// Centroid query terms interleaved per register tile: [tile][centroid][fast_code_query_tile_avx512].
// One vector load then serves every tile row of a document; missing rows stay zero.
class CentroidTiles {
 public:
  template<class QueryAt>
  CentroidTiles(size_t rows,QueryAt query) : centroids_(query(0).centroid_scores.size()) {
    const size_t tiles=(rows+fast_code_query_tile_avx512-1)/fast_code_query_tile_avx512;
    scores_.assign(tiles*centroids_*fast_code_query_tile_avx512,0);corrections_.assign(scores_.size(),0);
    for (size_t row=0;row<rows;++row) {
      const auto& q=query(row);
      if (q.centroid_scores.size()!=centroids_ || q.correction_weights.size()!=centroids_)
        throw std::runtime_error("TieredQ: invalid centroid shape");
      const size_t tile=row/fast_code_query_tile_avx512,lane=row%fast_code_query_tile_avx512;
      for (size_t centroid=0;centroid<centroids_;++centroid) {
        const size_t at=(tile*centroids_+centroid)*fast_code_query_tile_avx512+lane;
        scores_[at]=q.centroid_scores[centroid];corrections_[at]=q.correction_weights[centroid];
      }
    }
  }
  const float* scores(size_t tile) const { return scores_.data()+tile*centroids_*fast_code_query_tile_avx512; }
  const float* corrections(size_t tile) const { return corrections_.data()+tile*centroids_*fast_code_query_tile_avx512; }
 private:
  size_t centroids_;
  std::vector<float> scores_,corrections_;
};

// Transposes one 8-row vector per document of a block into one 16-document vector per row.
__attribute__((target("avx512f,avx512bw"),always_inline))
inline void transpose_tile_terms_avx512(const __m256* by_document,__m512* by_row) {
  constexpr size_t half=fast_scan_block_docs/2;
  __m512 joined[half],pairs[half],rows[half];
  for (size_t doc=0;doc<half;++doc)
    joined[doc]=_mm512_castpd_ps(_mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_castps_pd(by_document[doc])),
        _mm256_castps_pd(by_document[doc+half]),1));
  for (size_t doc=0;doc<half;doc+=2) {
    pairs[doc]=_mm512_unpacklo_ps(joined[doc],joined[doc+1]);
    pairs[doc+1]=_mm512_unpackhi_ps(joined[doc],joined[doc+1]);
  }
  // rows[r] and rows[r+4]: row r (128-bit lanes 0/2) and row r+4 (lanes 1/3) for documents 0-3 or 4-7.
  for (size_t quad=0;quad<half;quad+=4) {
    rows[quad]=_mm512_shuffle_ps(pairs[quad],pairs[quad+2],0x44);
    rows[quad+1]=_mm512_shuffle_ps(pairs[quad],pairs[quad+2],0xEE);
    rows[quad+2]=_mm512_shuffle_ps(pairs[quad+1],pairs[quad+3],0x44);
    rows[quad+3]=_mm512_shuffle_ps(pairs[quad+1],pairs[quad+3],0xEE);
  }
  const __m512i low=_mm512_setr_epi32(0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27);
  const __m512i high=_mm512_setr_epi32(4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31);
  for (size_t row=0;row<half/2;++row) {
    by_row[row]=_mm512_permutex2var_ps(rows[row],low,rows[row+4]);
    by_row[row+4]=_mm512_permutex2var_ps(rows[row],high,rows[row+4]);
  }
}

__attribute__((target("avx512f,avx512bw"),always_inline))
inline void code_base_terms_avx512(const FastScanCache& cache,const float* centers,
    const float* corrections,size_t doc,__m512* terms) {
  __m256 by_document[fast_scan_block_docs];
  // One conversion widens the block's 16 binary16 corrections; per-lane scalar decoding is far slower.
  alignas(64) float correction[fast_scan_block_docs];
  _mm512_store_ps(correction,_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(cache.correction.data()+doc))));
  for (size_t lane=0;lane<fast_scan_block_docs;++lane) {
    const size_t at=size_t(cache.centroid[doc+lane])*fast_code_query_tile_avx512;
    by_document[lane]=_mm256_add_ps(_mm256_loadu_ps(centers+at),
        _mm256_mul_ps(_mm256_loadu_ps(corrections+at),_mm256_set1_ps(correction[lane])));
  }
  transpose_tile_terms_avx512(by_document,terms);
}
// Same rounding sequence as the scalar codec: (center + weight*correction) + factor*dot.
template<size_t Rows,bool Filter=false>
__attribute__((target("avx512f,avx512bw"),always_inline))
inline void store_code_scores_avx512(const FastScanCache& cache,const __m512* dots,const float* centers,
    const float* corrections,size_t doc,size_t valid,float* scores,size_t stride,const uint16_t* keep=nullptr,const __m512* cached_terms=nullptr) {
  __m512 generated[fast_code_query_tile_avx512];
  if constexpr (!Filter) code_base_terms_avx512(cache,centers,corrections,doc,generated);
  const __m512* terms=Filter ? cached_terms : generated;
  const __m512 factor=_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(cache.factor.data()+doc)));
  const auto mask=__mmask16((1u<<valid)-1);
  for (size_t row=0;row<Rows;++row) {
    auto score=_mm512_add_ps(terms[row],_mm512_mul_ps(factor,dots[row]));
    if constexpr (Filter) score=_mm512_mask_blend_ps(keep[row],_mm512_set1_ps(-std::numeric_limits<float>::infinity()),score);
    _mm512_mask_storeu_ps(scores+row*stride,mask,score);
  }
}
template<size_t Rows>
__attribute__((target("avx512f,avx512bw")))
inline void accumulate_codes_avx512(const FastScanCache& cache,const float* decoded,const float* weights,
    const float* centers,const float* corrections,size_t doc,size_t valid,float* scores,size_t stride) {
  __m512 dots[Rows];
  for (auto& dot:dots) dot=_mm512_setzero_ps();
  for (size_t rank=0;rank<cache.dimensions;++rank) {
    const __m512 levels=_mm512_loadu_ps(decoded+rank*fast_scan_block_docs);
    for (size_t row=0;row<Rows;++row)
      dots[row]=_mm512_add_ps(dots[row],_mm512_mul_ps(levels,_mm512_set1_ps(weights[rank*Rows+row])));
  }
  store_code_scores_avx512<Rows>(cache,dots,centers,corrections,doc,valid,scores,stride);
}
template<size_t Blocks>
__attribute__((target("avx512f,avx512bw"),always_inline))
inline void scan_single_code_blocks_avx512(const FastScanCache& cache,const std::vector<CodeTableSegment>& segments,
    const float* weights,const float* centers,const float* corrections,size_t doc,size_t first,size_t count,float* scores) {
  AccumulateLevelsAvx512<Blocks> sink{weights,{}};
  for (auto& dot:sink.dot) dot=_mm512_setzero_ps();
  walk_code_block_avx512<Blocks>(cache,segments,doc/fast_scan_block_docs,sink);
  for (size_t block=0;block<Blocks;++block) {
    const size_t at=doc+block*fast_scan_block_docs;
    store_code_scores_avx512<1>(cache,&sink.dot[block],centers,corrections,at,std::min(fast_scan_block_docs,first+count-at),
        scores+at-first,count);
  }
}
__attribute__((target("avx512f,avx512bw")))
inline void scan_single_code_avx512(const FastScanCache& cache,const std::vector<CodeTableSegment>& segments,
    const float* weights,const float* centers,const float* corrections,size_t first,size_t count,float* scores) {
  constexpr size_t span=code_interleaved_blocks*fast_scan_block_docs;
  size_t doc=first;
  for (;doc+span<=first+count;doc+=span)
    scan_single_code_blocks_avx512<code_interleaved_blocks>(cache,segments,weights,centers,corrections,doc,first,count,scores);
  for (;doc<first+count;doc+=fast_scan_block_docs)
    scan_single_code_blocks_avx512<1>(cache,segments,weights,centers,corrections,doc,first,count,scores);
}

template<size_t Rows,class QueryAt>
__attribute__((target("avx2,f16c")))
inline void accumulate_codes_avx2(const FastScanCache& cache,const float* decoded,const float* weights,
    size_t doc,size_t valid,float* scores,size_t stride,QueryAt query) {
  constexpr size_t lanes=8,halves=fast_scan_block_docs/lanes;
  __m256 dots[halves][Rows];
  for (auto& half:dots) for (auto& dot:half) dot=_mm256_setzero_ps();
  for (size_t rank=0;rank<cache.dimensions;++rank) for (size_t half=0;half<halves;++half) {
    const __m256 levels=_mm256_loadu_ps(decoded+rank*fast_scan_block_docs+half*lanes);
    for (size_t row=0;row<Rows;++row)
      dots[half][row]=_mm256_add_ps(dots[half][row],_mm256_mul_ps(levels,_mm256_set1_ps(weights[rank*Rows+row])));
  }
  for (size_t half=0;half<halves && half*lanes<valid;++half) {
    const size_t at=doc+half*lanes;
    const auto cid=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(cache.centroid.data()+at)));
    const __m256 correction=_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cache.correction.data()+at))),factor=_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cache.factor.data()+at)));
    for (size_t row=0;row<Rows;++row) {
      const auto& q=query(row);
      auto score=_mm256_i32gather_ps(q.centroid_scores.data(),cid,sizeof(float));
      score=_mm256_add_ps(score,_mm256_mul_ps(_mm256_i32gather_ps(q.correction_weights.data(),cid,sizeof(float)),correction));
      score=_mm256_add_ps(score,_mm256_mul_ps(factor,dots[half][row]));
      alignas(32) float out[lanes];
      _mm256_store_ps(out,score);
      std::copy_n(out,std::min(lanes,valid-half*lanes),scores+row*stride+half*lanes);
    }
  }
}

template<class QueryAt>
inline void scan_codes_avx512_tile(const FastScanCache& cache,size_t rows,
    size_t first,size_t count,float* scores,size_t stride,QueryAt query) {
  static_assert(fast_code_query_tile_avx512==fast_scan_block_docs/2,"tile terms transpose 16 documents by 8 rows");
  const auto segments=code_table_segments(cache);
  const CentroidTiles centroids(rows,query);
  if (rows==1) {
    scan_single_code_avx512(cache,segments,query(0).weights.data(),centroids.scores(0),centroids.corrections(0),
        first,count,scores);
    return;
  }
  scan_code_blocks(cache,rows,fast_code_query_tile_avx512,first,count,scores,stride,query,
      [&](size_t block,float* decoded) { decode_code_block_avx512(cache,segments,block,decoded); },
      [&](const float* decoded,size_t tile,const float* weights,size_t height,size_t doc,size_t valid,float* out,size_t out_stride,auto) {
        dispatch_code_rows<fast_code_query_tile_avx512>(height,[&](auto tile_rows) {
          accumulate_codes_avx512<decltype(tile_rows)::value>(cache,decoded,weights,
              centroids.scores(tile),centroids.corrections(tile),doc,valid,out,out_stride);
        });
      });
}
template<class QueryAt>
inline void scan_codes_avx2_tile(const FastScanCache& cache,size_t rows,
    size_t first,size_t count,float* scores,size_t stride,QueryAt query) {
  scan_code_blocks(cache,rows,fast_code_query_tile_avx2,first,count,scores,stride,query,
      [&](size_t block,float* decoded) { decode_code_block(cache,block,decoded); },
      [&](const float* decoded,size_t,const float* weights,size_t height,size_t doc,size_t valid,float* out,size_t out_stride,auto at) {
        dispatch_code_rows<fast_code_query_tile_avx2>(height,[&](auto tile) {
          accumulate_codes_avx2<decltype(tile)::value>(cache,decoded,weights,doc,valid,out,out_stride,at);
        });
      });
}
inline void scan_codes_avx512(const FastScanCache& cache,const TieredQQuery& q,
    size_t first,size_t count,float* scores) {
  scan_codes_avx512_tile(cache,1,first,count,scores,count,[&](size_t)->const TieredQQuery& {return q;});
}
inline void scan_codes_avx2(const FastScanCache& cache,const TieredQQuery& q,
    size_t first,size_t count,float* scores) {
  scan_codes_avx2_tile(cache,1,first,count,scores,count,[&](size_t)->const TieredQQuery& {return q;});
}
} // namespace tq::tieredq_detail
