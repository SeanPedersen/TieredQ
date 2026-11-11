// AArch64 int8 full scan over the opt-in FastScanCache. Four coordinates share one int32 lane;
// one vdotq_s32 accumulates four documents per vector.
#pragma once
#include "tieredq_scan_int8.h"
#include "tieredq_fast_dispatch.h"
#include "tieredq_scan_codes_neon.h"
#include "tieredq_scan_lookup_neon.h"
#include <arm_neon.h>
#include <omp.h>

namespace tq::tieredq_detail {
inline bool fast_scan_available() { return native_fast_scan_available(); }
struct FastScanQuery {
  TieredQQuery codes;
  FastScanLookupQuery lookup;
  std::vector<int8x16_t> qvecs;
  std::vector<float> centroid_scores, correction_weights;
  float scale = 1;
};

inline FastScanQuery fast_scan_query(const FastScanCache& cache, const TieredQQuery& q) {
  if (q.weights.size() != cache.dimensions)
    throw std::runtime_error("TieredQ: int8 query weight shape mismatch");
  if (cache.code_lookup()) { FastScanQuery prepared; prepared.codes=q; return prepared; }
  float peak = 0;
  for (size_t r=0; r<cache.dimensions; ++r) {
    const size_t ti = cache.tier_of[r];
    if (ti == 0xFF) continue;
    peak = std::max(peak, std::fabs(q.weights[r] * cache.level_scale[ti]));
  }
  const float scale = peak > 0 ? peak / float(fast_scan_quant_max) : 1.0f;
  std::vector<int8_t> qint(cache.groups * 4, 0);
  for (size_t r=0; r<cache.dimensions; ++r) {
    const size_t ti = cache.tier_of[r];
    if (ti == 0xFF) continue;
    const long value = std::lround(q.weights[r] * cache.level_scale[ti] / scale);
    qint[r] = int8_t(std::clamp<long>(value, -long(fast_scan_quant_max), long(fast_scan_quant_max)));
  }
  FastScanQuery prepared;
  prepared.scale = scale;
  prepared.centroid_scores = q.centroid_scores;
  prepared.correction_weights = q.correction_weights;
  prepared.qvecs.resize(cache.groups);
  for (size_t g=0; g<cache.groups; ++g) {
    alignas(16) int8_t lanes[16];
    for (size_t v=0; v<4; ++v) for (size_t c=0; c<4; ++c) lanes[v*4+c] = qint[g*4+c];
    prepared.qvecs[g] = vld1q_s8(lanes);
  }
  return prepared;
}

constexpr size_t fast_scan_query_tile_neon = 4;
// A compile-time query count keeps every accumulator in registers; with a runtime count Clang spills them around
// each vdotq_s32, which made single-query scans 3.6x slower.
template<size_t Queries>
inline void scan_fast_neon_levels(const FastScanCache& cache,const FastScanQuery* queries,
    size_t first,size_t count,float* scores,size_t stride) {
  const int8x16_t* qvecs[Queries];
  for (size_t t=0;t<Queries;++t) qvecs[t]=queries[t].qvecs.data();
  for (size_t block=first/fast_scan_block_docs; block<(first+count+fast_scan_block_docs-1)/fast_scan_block_docs; ++block) {
    int32x4_t sums[Queries][4];
    for (auto& query_sums:sums) for (auto& sum:query_sums) sum=vdupq_n_s32(0);
    const int8_t* base=cache.levels.data()+block*cache.groups*64;
#pragma clang loop unroll_count(4)
    for (size_t g=0;g<cache.groups;++g) {
      const int8_t* p=base+g*64;
      const int8x16_t levels[4]={vld1q_s8(p),vld1q_s8(p+16),vld1q_s8(p+32),vld1q_s8(p+48)};
      for (size_t t=0;t<Queries;++t) for (size_t v=0;v<4;++v)
        sums[t][v]=vdotq_s32(sums[t][v],levels[v],qvecs[t][g]);
    }
    alignas(16) int32_t values[Queries][4][4];
    for (size_t t=0;t<Queries;++t) for (size_t v=0;v<4;++v) vst1q_s32(values[t][v],sums[t][v]);
    for (size_t v=0;v<4;++v) for (size_t c=0;c<4;++c) {
      const size_t doc=block*fast_scan_block_docs+v*4+c;
      if (doc>=first+count) continue;
      const uint8_t cid=cache.centroid[doc];
      for (size_t t=0;t<Queries;++t) {
        const auto& q=queries[t];
        if (cid>=q.centroid_scores.size() || q.centroid_scores.size()!=q.correction_weights.size())
          throw std::runtime_error("TieredQ: invalid centroid shape or id");
        scores[t*stride+doc-first]=q.centroid_scores[cid]+q.correction_weights[cid]*cache.correction_at(doc)+
            cache.factor_at(doc)*(float(values[t][v][c])*q.scale);
      }
    }
  }
}
inline void scan_fast_neon_tile(const FastScanCache& cache,const FastScanQuery* queries,
    size_t query_count,size_t first,size_t count,float* scores,size_t stride) {
  if (cache.code_lookup()) {
    scan_codes_neon_tile(cache,query_count,first,count,scores,stride,
        [&](size_t row)->const TieredQQuery& {return queries[row].codes;});
    return;
  }
  if (cache.lookup) {
    scan_lookup_query_group(query_count,first,count,scores,stride,[&](size_t row,size_t begin,size_t documents,float* out) {
      scan_fast_lookup_neon_range(cache,queries[row].lookup,begin,documents,out);
    });
    return;
  }
  static_assert(fast_scan_query_tile_neon==4,"dispatch covers query tiles of one to four");
  switch (query_count) {
    case 0: return;
    case 1: return scan_fast_neon_levels<1>(cache,queries,first,count,scores,stride);
    case 2: return scan_fast_neon_levels<2>(cache,queries,first,count,scores,stride);
    case 3: return scan_fast_neon_levels<3>(cache,queries,first,count,scores,stride);
    case 4: return scan_fast_neon_levels<4>(cache,queries,first,count,scores,stride);
    default: throw std::invalid_argument("TieredQ: NEON query tile exceeds four queries");
  }
}
inline void scan_fast_neon(const FastScanCache& cache,const FastScanQuery& query,float* scores) {
  scan_fast_neon_tile(cache,&query,1,0,cache.count,scores,cache.count);
}
inline void scan_fast_neon_batch(const FastScanCache& cache,
    const std::vector<FastScanQuery>& queries,float* scores) {
#pragma omp parallel for schedule(static) if(queries.size()>fast_scan_query_tile_neon && !omp_in_parallel())
  for (size_t first=0;first<queries.size();first+=fast_scan_query_tile_neon)
    scan_fast_neon_tile(cache,queries.data()+first,std::min(fast_scan_query_tile_neon,queries.size()-first),
        0,cache.count,scores+first*cache.count,cache.count);
}
} // namespace tq::tieredq_detail
