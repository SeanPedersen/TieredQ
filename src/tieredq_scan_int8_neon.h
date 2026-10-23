// AArch64 int8 full scan over the opt-in FastScanCache. Four coordinates share one int32 lane;
// one vdotq_s32 accumulates four documents per vector.
#pragma once
#include "tieredq_scan_int8.h"
#include <arm_neon.h>

namespace dv::tieredq_detail {
inline bool fast_scan_available() { return true; }
struct FastScanQuery {
  std::vector<int8x16_t> qvecs;
  std::vector<float> centroid_scores, correction_weights;
  float scale = 1;
};

inline FastScanQuery fast_scan_query(const FastScanCache& cache, const TieredQQuery& q) {
  if (q.weights.size() != cache.dimensions)
    throw std::runtime_error("TieredQ: int8 query weight shape mismatch");
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

inline void scan_fast_neon(const FastScanCache& cache, const FastScanQuery& query,
                           float* scores) {
  if (query.centroid_scores.size() != query.correction_weights.size())
    throw std::runtime_error("TieredQ: int8 query centroid shape mismatch");
  for (size_t block=0; block<cache.blocks; ++block) {
    int32x4_t sums[4] = {vdupq_n_s32(0),vdupq_n_s32(0),vdupq_n_s32(0),vdupq_n_s32(0)};
    const int8_t* base = cache.levels.data() + block*cache.groups*64;
#pragma clang loop unroll_count(4)
    for (size_t g=0; g<cache.groups; ++g) {
      const int8_t* p = base + g*64;
      for (size_t v=0; v<4; ++v)
        sums[v] = vdotq_s32(sums[v], vld1q_s8(p + v*16), query.qvecs[g]);
    }
    alignas(16) int32_t values[4];
    for (size_t v=0; v<4; ++v) {
      vst1q_s32(values, sums[v]);
      for (size_t c=0; c<4; ++c) {
        const size_t doc = block*16 + v*4 + c;
        if (doc >= cache.count) break;
        const uint8_t cid = cache.centroid[doc];
        if (cid >= query.centroid_scores.size())
          throw std::runtime_error("TieredQ: invalid centroid id");
        scores[doc] = query.centroid_scores[cid] + query.correction_weights[cid]*cache.correction[doc] +
                      cache.factor[doc] * (float(values[c]) * query.scale);
      }
    }
  }
}
inline void scan_fast_neon_batch(const FastScanCache& cache,
                                 const std::vector<FastScanQuery>& queries, float* scores) {
  if (queries.empty()) return;
  constexpr size_t query_tile = 4;
  const size_t nq = queries.size();
#pragma omp parallel for schedule(static) if(nq>query_tile)
  for (size_t q0=0; q0<nq; q0+=query_tile) {
    const size_t qn = std::min(query_tile, nq-q0);
    for (size_t block=0; block<cache.blocks; ++block) {
      int32x4_t sums[query_tile][4];
      for (size_t t=0; t<qn; ++t) for (size_t v=0; v<4; ++v) sums[t][v]=vdupq_n_s32(0);
      const int8_t* base = cache.levels.data() + block*cache.groups*64;
#pragma clang loop unroll_count(4)
      for (size_t g=0; g<cache.groups; ++g) {
        const int8_t* p = base + g*64;
        const int8x16_t d0=vld1q_s8(p), d1=vld1q_s8(p+16), d2=vld1q_s8(p+32), d3=vld1q_s8(p+48);
        for (size_t t=0; t<qn; ++t) {
          const int8x16_t qv=queries[q0+t].qvecs[g];
          sums[t][0]=vdotq_s32(sums[t][0],d0,qv); sums[t][1]=vdotq_s32(sums[t][1],d1,qv);
          sums[t][2]=vdotq_s32(sums[t][2],d2,qv); sums[t][3]=vdotq_s32(sums[t][3],d3,qv);
        }
      }
      alignas(16) int32_t values[query_tile][4][4];
      for (size_t t=0; t<qn; ++t) for (size_t v=0; v<4; ++v) vst1q_s32(values[t][v], sums[t][v]);
      for (size_t v=0; v<4; ++v) for (size_t c=0; c<4; ++c) {
        const size_t doc = block*16 + v*4 + c;
        if (doc >= cache.count) continue;
        const uint8_t cid = cache.centroid[doc];
        for (size_t t=0; t<qn; ++t) {
          const auto& q=queries[q0+t];
          scores[(q0+t)*cache.count + doc] = q.centroid_scores[cid] +
              q.correction_weights[cid]*cache.correction[doc] +
              cache.factor[doc] * (float(values[t][v][c]) * q.scale);
        }
      }
    }
  }
}

} // namespace dv::tieredq_detail
