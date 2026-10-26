// Experimental register-resident code-ID kernels (bench only).
#pragma once
#include "tieredq_scan_codes.h"
#include <immintrin.h>
#include <array>
#include <utility>

namespace bench {
using namespace dv::tieredq_detail;
using dv::TieredQQuery;

struct Plan {
  std::vector<const float*> books;
  std::vector<float> weights; // rank-major, rows per rank
  size_t rows;
};
template<class QueryAt>
Plan make_plan(const FastScanCache& cache,size_t rows,QueryAt query) {
  Plan plan{std::vector<const float*>(cache.dimensions),std::vector<float>(cache.dimensions*rows),rows};
  for (size_t rank=0;rank<cache.dimensions;++rank) {
    plan.books[rank]=cache.codebooks[cache.tier_of[rank]].data();
    for (size_t row=0;row<rows;++row) plan.weights[rank*rows+row]=query(row).weights[rank];
  }
  return plan;
}

template<size_t Rows,size_t Blocks>
__attribute__((target("avx512f,avx512bw"),always_inline))
inline void accumulate512(__m512 (&acc)[Blocks][Rows],const __m512i* codes,const float* book,const float* w) {
  for (size_t b=0;b<Blocks;++b) {
    const __m512 levels=_mm512_i32gather_ps(codes[b],book,sizeof(float));
    for (size_t r=0;r<Rows;++r) acc[b][r]=_mm512_add_ps(acc[b][r],_mm512_mul_ps(levels,_mm512_set1_ps(w[r])));
  }
}
template<size_t Rows,size_t Blocks,class QueryAt>
__attribute__((target("avx512f,avx512bw")))
inline void kernel512(const FastScanCache& cache,const Plan& plan,size_t first,size_t count,
    float* scores,size_t stride,QueryAt query) {
  constexpr size_t lanes=16,step=lanes*Blocks;
  const size_t end=first+count,full_groups=cache.dimensions/fast_scan_coordinates_per_group;
  const float* const* books=plan.books.data();
  const float* weights=plan.weights.data();
  const __m512i byte_mask=_mm512_set1_epi32(UINT8_MAX);
  const size_t padded_end=std::min((end+lanes-1)/lanes*lanes,cache.blocks*fast_scan_block_docs);
  size_t doc=first;
  for (;doc+step<=padded_end;doc+=step) {
    __m512 acc[Blocks][Rows];
    for (auto& block:acc) for (auto& value:block) value=_mm512_setzero_ps();
    const int8_t* base=cache.levels.data()+doc/fast_scan_block_docs*cache.groups*fast_scan_group_bytes;
    for (size_t g=0;g<full_groups;++g) {
      __m512i packed[Blocks];
      for (size_t b=0;b<Blocks;++b) packed[b]=_mm512_loadu_si512(base+(b*cache.groups+g)*fast_scan_group_bytes);
      __m512i codes[Blocks];
      for (size_t b=0;b<Blocks;++b) codes[b]=_mm512_and_si512(packed[b],byte_mask);
      accumulate512<Rows,Blocks>(acc,codes,books[g*4],weights+(g*4)*Rows);
      for (size_t b=0;b<Blocks;++b) codes[b]=_mm512_and_si512(_mm512_srli_epi32(packed[b],8),byte_mask);
      accumulate512<Rows,Blocks>(acc,codes,books[g*4+1],weights+(g*4+1)*Rows);
      for (size_t b=0;b<Blocks;++b) codes[b]=_mm512_and_si512(_mm512_srli_epi32(packed[b],16),byte_mask);
      accumulate512<Rows,Blocks>(acc,codes,books[g*4+2],weights+(g*4+2)*Rows);
      for (size_t b=0;b<Blocks;++b) codes[b]=_mm512_srli_epi32(packed[b],24);
      accumulate512<Rows,Blocks>(acc,codes,books[g*4+3],weights+(g*4+3)*Rows);
    }
    for (size_t rank=full_groups*4;rank<cache.dimensions;++rank) {
      __m512i codes[Blocks];
      for (size_t b=0;b<Blocks;++b) codes[b]=_mm512_and_si512(_mm512_srli_epi32(
          _mm512_loadu_si512(base+(b*cache.groups+full_groups)*fast_scan_group_bytes),int(rank%4*8)),byte_mask);
      accumulate512<Rows,Blocks>(acc,codes,books[rank],weights+(rank)*Rows);
    }
    for (size_t b=0;b<Blocks;++b) {
      const size_t at=doc+b*lanes;
      if (at>=end) break;
      const auto cid=_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cache.centroid.data()+at)));
      const __m512 correction=_mm512_loadu_ps(cache.correction.data()+at),factor=_mm512_loadu_ps(cache.factor.data()+at);
      const __mmask16 mask=__mmask16((1u<<std::min(lanes,end-at))-1);
      for (size_t r=0;r<Rows;++r) {
        const auto& q=query(r);
        auto score=_mm512_i32gather_ps(cid,q.centroid_scores.data(),sizeof(float));
        score=_mm512_add_ps(score,_mm512_mul_ps(_mm512_i32gather_ps(cid,q.correction_weights.data(),sizeof(float)),correction));
        score=_mm512_add_ps(score,_mm512_mul_ps(factor,acc[b][r]));
        _mm512_mask_storeu_ps(scores+r*stride+at-first,mask,score);
      }
    }
  }
  if constexpr (Blocks>1) if (doc<end) kernel512<Rows,1>(cache,plan,doc,end-doc,scores+(doc-first),stride,query);
}
} // namespace bench

namespace bench {
// Register-table lookup: pairs of 16-float registers resolve the low five code bits with vpermt2ps,
// then the remaining bits select between pairs. Exact: returns the stored FP32 codebook value.
constexpr size_t table_lanes=16,pair_entries=32,pair_bits=5;
template<size_t Pairs>
__attribute__((target("avx512f,avx512bw"),always_inline))
inline __m512 table_lookup512(const __m512* table,__m512i code) {
  __m512 level[Pairs];
  for (size_t p=0;p<Pairs;++p) level[p]=_mm512_permutex2var_ps(table[2*p],code,table[2*p+1]);
  size_t live=Pairs;
  for (int bit=pair_bits;live>1;++bit,live/=2) {
    const __mmask16 high=_mm512_test_epi32_mask(code,_mm512_set1_epi32(1<<bit));
    for (size_t p=0;p<live/2;++p) level[p]=_mm512_mask_blend_ps(high,level[2*p],level[2*p+1]);
  }
  return level[0];
}
struct TablePlan {
  struct Segment { size_t begin,end,pairs; const float* table; };
  std::vector<Segment> segments;
  std::vector<std::vector<float>> tables; // per tier, zero padded to whole register pairs
  std::vector<float> weights;
};
template<class QueryAt>
TablePlan make_table_plan(const FastScanCache& cache,size_t rows,QueryAt query) {
  TablePlan plan;
  plan.tables.resize(cache.codebooks.size());
  for (size_t t=0;t<cache.codebooks.size();++t) {
    const size_t entries=std::max(pair_entries,cache.codebooks[t].size());
    plan.tables[t].assign(entries,0);
    std::copy(cache.codebooks[t].begin(),cache.codebooks[t].end(),plan.tables[t].begin());
  }
  for (size_t rank=0;rank<cache.dimensions;) {
    size_t end=rank;
    while (end<cache.dimensions && cache.tier_of[end]==cache.tier_of[rank]) ++end;
    const auto& table=plan.tables[cache.tier_of[rank]];
    plan.segments.push_back({rank,end,table.size()/pair_entries,table.data()});
    rank=end;
  }
  plan.weights.resize(cache.dimensions*rows);
  for (size_t rank=0;rank<cache.dimensions;++rank) for (size_t row=0;row<rows;++row)
    plan.weights[rank*rows+row]=query(row).weights[rank];
  return plan;
}
template<size_t Rows,size_t Blocks,size_t Pairs>
__attribute__((target("avx512f,avx512bw"),always_inline))
inline void table_segment512(__m512 (&acc)[Blocks][Rows],const int8_t* base,size_t groups,
    const TablePlan::Segment& segment,const float* weights) {
  __m512 table[2*Pairs];
  for (size_t i=0;i<2*Pairs;++i) table[i]=_mm512_loadu_ps(segment.table+i*table_lanes);
  for (size_t rank=segment.begin;rank<segment.end;++rank) {
    const __m512i shift=_mm512_set1_epi32(int(rank%fast_scan_coordinates_per_group*fast_scan_code_bits));
    const float* w=weights+rank*Rows;
    for (size_t b=0;b<Blocks;++b) {
      const __m512i packed=_mm512_loadu_si512(base+(b*groups+rank/fast_scan_coordinates_per_group)*fast_scan_group_bytes);
      const __m512 levels=table_lookup512<Pairs>(table,_mm512_srlv_epi32(packed,shift));
      for (size_t r=0;r<Rows;++r) acc[b][r]=_mm512_add_ps(acc[b][r],_mm512_mul_ps(levels,_mm512_set1_ps(w[r])));
    }
  }
}
template<size_t Rows,size_t Blocks,class QueryAt>
__attribute__((target("avx512f,avx512bw")))
inline void table512(const FastScanCache& cache,const TablePlan& plan,size_t first,size_t count,
    float* scores,size_t stride,QueryAt query) {
  constexpr size_t lanes=16,step=lanes*Blocks;
  const size_t end=first+count;
  const size_t padded_end=std::min((end+lanes-1)/lanes*lanes,cache.blocks*fast_scan_block_docs);
  size_t doc=first;
  for (;doc+step<=padded_end;doc+=step) {
    __m512 acc[Blocks][Rows];
    for (auto& block:acc) for (auto& value:block) value=_mm512_setzero_ps();
    const int8_t* base=cache.levels.data()+doc/fast_scan_block_docs*cache.groups*fast_scan_group_bytes;
    for (const auto& segment:plan.segments) {
      switch (segment.pairs) {
        case 1: table_segment512<Rows,Blocks,1>(acc,base,cache.groups,segment,plan.weights.data()); break;
        case 2: table_segment512<Rows,Blocks,2>(acc,base,cache.groups,segment,plan.weights.data()); break;
        case 4: table_segment512<Rows,Blocks,4>(acc,base,cache.groups,segment,plan.weights.data()); break;
        default: table_segment512<Rows,Blocks,8>(acc,base,cache.groups,segment,plan.weights.data()); break;
      }
    }
    for (size_t b=0;b<Blocks;++b) {
      const size_t at=doc+b*lanes;
      if (at>=end) break;
      const auto cid=_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cache.centroid.data()+at)));
      const __m512 correction=_mm512_loadu_ps(cache.correction.data()+at),factor=_mm512_loadu_ps(cache.factor.data()+at);
      const __mmask16 mask=__mmask16((1u<<std::min(lanes,end-at))-1);
      for (size_t r=0;r<Rows;++r) {
        const auto& q=query(r);
        auto score=_mm512_i32gather_ps(cid,q.centroid_scores.data(),sizeof(float));
        score=_mm512_add_ps(score,_mm512_mul_ps(_mm512_i32gather_ps(cid,q.correction_weights.data(),sizeof(float)),correction));
        score=_mm512_add_ps(score,_mm512_mul_ps(factor,acc[b][r]));
        _mm512_mask_storeu_ps(scores+r*stride+at-first,mask,score);
      }
    }
  }
  if constexpr (Blocks>1) if (doc<end) table512<Rows,1>(cache,plan,doc,end-doc,scores+(doc-first),stride,query);
}
} // namespace bench

namespace bench {
// Decode each 16-document block once into FP32 levels, then apply every query tile to the decoded block.
template<size_t Pairs>
__attribute__((target("avx512f,avx512bw"),always_inline))
inline void decode_segment512(float* out,const int8_t* base,const TablePlan::Segment& segment) {
  __m512 table[2*Pairs];
  for (size_t i=0;i<2*Pairs;++i) table[i]=_mm512_loadu_ps(segment.table+i*table_lanes);
  size_t rank=segment.begin;
  for (;rank<segment.end && rank%4;++rank)
    _mm512_storeu_ps(out+rank*16,table_lookup512<Pairs>(table,_mm512_srlv_epi32(_mm512_loadu_si512(base+rank/4*64),_mm512_set1_epi32(int(rank%4*8)))));
  for (;rank+4<=segment.end;rank+=4) {
    const __m512i packed=_mm512_loadu_si512(base+rank/4*64);
    _mm512_storeu_ps(out+rank*16,table_lookup512<Pairs>(table,packed));
    _mm512_storeu_ps(out+(rank+1)*16,table_lookup512<Pairs>(table,_mm512_srli_epi32(packed,8)));
    _mm512_storeu_ps(out+(rank+2)*16,table_lookup512<Pairs>(table,_mm512_srli_epi32(packed,16)));
    _mm512_storeu_ps(out+(rank+3)*16,table_lookup512<Pairs>(table,_mm512_srli_epi32(packed,24)));
  }
  for (;rank<segment.end;++rank)
    _mm512_storeu_ps(out+rank*16,table_lookup512<Pairs>(table,_mm512_srlv_epi32(_mm512_loadu_si512(base+rank/4*64),_mm512_set1_epi32(int(rank%4*8)))));
}
__attribute__((target("avx512f,avx512bw")))
inline void decode_block512(float* out,const int8_t* base,const TablePlan& plan) {
  for (const auto& segment:plan.segments) switch (segment.pairs) {
    case 1: decode_segment512<1>(out,base,segment); break;
    case 2: decode_segment512<2>(out,base,segment); break;
    case 4: decode_segment512<4>(out,base,segment); break;
    default: decode_segment512<8>(out,base,segment); break;
  }
}
template<size_t Rows>
__attribute__((target("avx512f,avx512bw")))
inline void apply512(const float* decoded,size_t dims,const float* weights,__m512* result) {
  __m512 acc[Rows];
  for (auto& a:acc) a=_mm512_setzero_ps();
  for (size_t rank=0;rank<dims;++rank) {
    const __m512 levels=_mm512_loadu_ps(decoded+rank*16);
    const float* w=weights+rank*Rows;
    for (size_t r=0;r<Rows;++r) acc[r]=_mm512_add_ps(acc[r],_mm512_mul_ps(levels,_mm512_set1_ps(w[r])));
  }
  for (size_t r=0;r<Rows;++r) result[r]=acc[r];
}
// plans: one TablePlan (weights) per tile of Rows queries.
template<size_t Rows,class QueryAt>
__attribute__((target("avx512f,avx512bw")))
inline void decoded512(const FastScanCache& cache,const std::vector<TablePlan>& plans,size_t first,size_t count,
    float* scores,size_t stride,QueryAt query,std::vector<float>& decoded) {
  constexpr size_t lanes=16;
  decoded.resize(cache.dimensions*lanes);
  const size_t end=first+count;
  for (size_t doc=first;doc<end;doc+=lanes) {
    const int8_t* base=cache.levels.data()+doc/fast_scan_block_docs*cache.groups*fast_scan_group_bytes;
    decode_block512(decoded.data(),base,plans[0]);
    const auto cid=_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cache.centroid.data()+doc)));
    const __m512 correction=_mm512_loadu_ps(cache.correction.data()+doc),factor=_mm512_loadu_ps(cache.factor.data()+doc);
    const __mmask16 mask=__mmask16((1u<<std::min(lanes,end-doc))-1);
    for (size_t t=0;t<plans.size();++t) {
      __m512 dots[Rows];
      apply512<Rows>(decoded.data(),cache.dimensions,plans[t].weights.data(),dots);
      for (size_t r=0;r<Rows;++r) {
        const auto& q=query(t*Rows+r);
        auto score=_mm512_i32gather_ps(cid,q.centroid_scores.data(),sizeof(float));
        score=_mm512_add_ps(score,_mm512_mul_ps(_mm512_i32gather_ps(cid,q.correction_weights.data(),sizeof(float)),correction));
        score=_mm512_add_ps(score,_mm512_mul_ps(factor,dots[r]));
        _mm512_mask_storeu_ps(scores+(t*Rows+r)*stride+doc-first,mask,score);
      }
    }
  }
}
} // namespace bench
