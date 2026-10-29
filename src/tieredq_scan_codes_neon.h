// NEON FP32 accumulation over decoded packed-code blocks preserves high-bit reconstruction levels.
#pragma once
#include "tieredq_scan_codes.h"
#include <arm_neon.h>

namespace dv::tieredq_detail {
constexpr size_t fast_code_query_tile_neon=4;

constexpr size_t code_neon_lanes=4,code_neon_quarters=fast_scan_block_docs/code_neon_lanes;
template<size_t Rows,class QueryAt>
inline void store_code_scores_neon(const FastScanCache& cache,const float32x4_t (&dots)[code_neon_quarters][Rows],
    size_t doc,size_t valid,float* scores,size_t stride,QueryAt query) {
  constexpr size_t lanes=code_neon_lanes;
  for (size_t quarter=0;quarter<code_neon_quarters && quarter*lanes<valid;++quarter) {
    const size_t at=doc+quarter*lanes;
    float corrections_at[lanes],factors_at[lanes];
    for (size_t lane=0;lane<lanes;++lane) { corrections_at[lane]=cache.correction_at(at+lane); factors_at[lane]=cache.factor_at(at+lane); }
    const auto correction=vld1q_f32(corrections_at),factor=vld1q_f32(factors_at);
    for (size_t row=0;row<Rows;++row) {
      const auto& q=query(row);
      float centers[lanes],corrections[lanes];
      for (size_t lane=0;lane<lanes;++lane) {
        const auto cid=cache.centroid[at+lane];
        centers[lane]=q.centroid_scores[cid];corrections[lane]=q.correction_weights[cid];
      }
      auto score=vaddq_f32(vld1q_f32(centers),vmulq_f32(vld1q_f32(corrections),correction));
      score=vaddq_f32(score,vmulq_f32(factor,dots[quarter][row]));
      float out[lanes];vst1q_f32(out,score);
      std::copy_n(out,std::min(lanes,valid-quarter*lanes),scores+row*stride+quarter*lanes);
    }
  }
}
template<size_t Rows,class QueryAt>
inline void accumulate_codes_neon(const FastScanCache& cache,const float* decoded,const float* weights,
    size_t doc,size_t valid,float* scores,size_t stride,QueryAt query) {
  float32x4_t dots[code_neon_quarters][Rows];
  for (auto& quarter:dots) for (auto& dot:quarter) dot=vdupq_n_f32(0);
  for (size_t rank=0;rank<cache.dimensions;++rank) {
    const float* w=weights+rank*Rows;
    for (size_t quarter=0;quarter<code_neon_quarters;++quarter) {
      const auto levels=vld1q_f32(decoded+rank*fast_scan_block_docs+quarter*code_neon_lanes);
      for (size_t row=0;row<Rows;++row)
        dots[quarter][row]=vaddq_f32(dots[quarter][row],vmulq_n_f32(levels,w[row]));
    }
  }
  store_code_scores_neon<Rows>(cache,dots,doc,valid,scores,stride,query);
}
// A single query consumes table values directly instead of storing and reloading a decoded block.
inline void scan_single_code_neon(const FastScanCache& cache,const TieredQQuery& q,
    size_t first,size_t count,float* scores) {
  for (size_t doc=first;doc<first+count;doc+=fast_scan_block_docs) {
    const int8_t* base=cache.levels.data()+doc/fast_scan_block_docs*cache.level_block_bytes();
    float32x4_t dots[code_neon_quarters][1];
    for (auto& quarter:dots) quarter[0]=vdupq_n_f32(0);
    uint32_t codes[fast_scan_block_docs];
    for (size_t rank=0;rank<cache.dimensions;++rank) {
      const float* book=cache.codebooks[cache.tier_of[rank]].data();
      read_block_codes(base,cache.code_offset[rank],fast_code_width(cache,rank),codes);
      for (size_t quarter=0;quarter<code_neon_quarters;++quarter) {
        const uint32_t* lane=codes+quarter*code_neon_lanes;
        float32x4_t levels=vdupq_n_f32(book[lane[0]]);
        levels=vsetq_lane_f32(book[lane[1]],levels,1);
        levels=vsetq_lane_f32(book[lane[2]],levels,2);
        levels=vsetq_lane_f32(book[lane[3]],levels,3);
        dots[quarter][0]=vaddq_f32(dots[quarter][0],vmulq_n_f32(levels,q.weights[rank]));
      }
    }
    store_code_scores_neon<1>(cache,dots,doc,std::min(fast_scan_block_docs,first+count-doc),scores+doc-first,count,
        [&](size_t)->const TieredQQuery& {return q;});
  }
}

template<class QueryAt>
inline void scan_codes_neon_tile(const FastScanCache& cache,size_t rows,
    size_t first,size_t count,float* scores,size_t stride,QueryAt query) {
  if (rows==1) { scan_single_code_neon(cache,query(0),first,count,scores); return; }
  scan_code_blocks(cache,rows,fast_code_query_tile_neon,first,count,scores,stride,query,
      [&](size_t block,float* decoded) { decode_code_block(cache,block,decoded); },
      [&](const float* decoded,size_t,const float* weights,size_t height,size_t doc,size_t valid,float* out,size_t out_stride,auto at) {
        dispatch_code_rows<fast_code_query_tile_neon>(height,[&](auto tile) {
          accumulate_codes_neon<decltype(tile)::value>(cache,decoded,weights,doc,valid,out,out_stride,at);
        });
      });
}
inline void scan_codes_neon(const FastScanCache& cache,const TieredQQuery& q,
    size_t first,size_t count,float* scores) {
  scan_codes_neon_tile(cache,1,first,count,scores,count,[&](size_t)->const TieredQQuery& {return q;});
}
} // namespace dv::tieredq_detail
