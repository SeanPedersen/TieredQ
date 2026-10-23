// x86 fast-scan dispatch: narrow models use compact query-specific byte lookup tables.
// Wider models retain the AVX-512BW level-dot fallback with a bias removed per query;
// vpmaddubsw/vpmaddwd support CPUs without VNNI.
#pragma once
#include "tieredq_scan_int8.h"
#include "tieredq_scan_lookup_avx512.h"
#include <immintrin.h>
#include <omp.h>

namespace dv::tieredq_detail {
struct FastScanQueryX86 {
  FastScanLookupQuery lookup;
  std::vector<int32_t> qwords;
  std::vector<float> centroid_scores, correction_weights;
  float scale = 1, offset = 0;
};

inline bool fast_scan_available() {
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
}

__attribute__((target("avx512f,avx512bw")))
inline FastScanQueryX86 fast_scan_query_x86(const FastScanCache& cache, const TieredQQuery& q) {
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
  long sum = 0;
  for (size_t r=0; r<cache.dimensions; ++r) {
    const size_t ti = cache.tier_of[r];
    if (ti == 0xFF) continue;
    const long value = std::lround(q.weights[r] * cache.level_scale[ti] / scale);
    qint[r] = int8_t(std::clamp<long>(value, -long(fast_scan_quant_max), long(fast_scan_quant_max)));
    sum += qint[r];
  }
  FastScanQueryX86 prepared;
  prepared.scale = scale;
  prepared.offset = float((cache.level_max + 1) * sum);
  prepared.centroid_scores = q.centroid_scores;
  prepared.correction_weights = q.correction_weights;
  prepared.qwords.resize(cache.groups);
  for (size_t g=0; g<cache.groups; ++g)
    prepared.qwords[g] = int32_t(uint32_t(uint8_t(qint[g*4])) | (uint32_t(uint8_t(qint[g*4+1]))<<8)
                               | (uint32_t(uint8_t(qint[g*4+2]))<<16) | (uint32_t(uint8_t(qint[g*4+3]))<<24));
  return prepared;
}

inline FastScanQueryX86 fast_scan_query_x86(const FastScanCache& cache,
    const TieredQ& codec,const float* vector) {
  if (!cache.lookup) return fast_scan_query_x86(cache,codec.query(vector));
  FastScanQueryX86 query;
  query.lookup=prepare_fast_lookup_query(*cache.lookup,vector);
  return query;
}

__attribute__((target("avx512f,avx512bw")))
inline __mmask16 fast_mask(size_t n) {
  return n >= 16 ? __mmask16(0xFFFF) : __mmask16((1u<<n)-1);
}

__attribute__((target("avx512f,avx512bw")))
inline void scan_fast_avx512(const FastScanCache& cache, const FastScanQueryX86& query,
                             float* scores) {
  if (cache.lookup) { scan_fast_lookup_avx512(cache,query.lookup,scores); return; }
  const __m512i ones = _mm512_set1_epi16(1);
  const __m512i bias = _mm512_set1_epi8(char(cache.level_max + 1));
  const __m512 scale = _mm512_set1_ps(query.scale);
  const __m512 offset = _mm512_set1_ps(query.offset);
  for (size_t block=0; block<cache.blocks; ++block) {
    __m512i sum = _mm512_setzero_si512();
    const int8_t* base = cache.levels.data() + block*cache.groups*64;
    for (size_t g=0; g<cache.groups; ++g) {
      const __m512i levels = _mm512_add_epi8(_mm512_loadu_si512(base + g*64), bias);
      const __m512i pairs = _mm512_maddubs_epi16(levels, _mm512_set1_epi32(query.qwords[g]));
      sum = _mm512_add_epi32(sum, _mm512_madd_epi16(pairs, ones));
    }
    __m512 dots = _mm512_mul_ps(_mm512_sub_ps(_mm512_cvtepi32_ps(sum), offset), scale);
    const __m512i cid = _mm512_cvtepu8_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(cache.centroid.data()+block*16)));
    __m512 score = _mm512_i32gather_ps(cid, query.centroid_scores.data(), 4);
    score = _mm512_add_ps(score, _mm512_mul_ps(
        _mm512_i32gather_ps(cid, query.correction_weights.data(), 4),
        _mm512_loadu_ps(cache.correction.data()+block*16)));
    score = _mm512_add_ps(score, _mm512_mul_ps(
        _mm512_loadu_ps(cache.factor.data()+block*16), dots));
    const size_t n = std::min<size_t>(16, cache.count-block*16);
    _mm512_mask_storeu_ps(scores+block*16, fast_mask(n), score);
  }
}

__attribute__((target("avx512f,avx512bw")))
inline void scan_fast_avx512_batch(const FastScanCache& cache,
                                   const std::vector<FastScanQueryX86>& queries, float* scores) {
  if (queries.empty()) return;
  if (cache.lookup) {
    const size_t nq=queries.size();
#pragma omp parallel for schedule(static) if(nq>1 && !omp_in_parallel())
    for (size_t row=0;row<nq;++row)
      scan_fast_lookup_avx512(cache,queries[row].lookup,scores+row*cache.count);
    return;
  }
  constexpr size_t query_tile = 16;
  const size_t nq = queries.size();
  const __m512i ones = _mm512_set1_epi16(1);
  const __m512i bias = _mm512_set1_epi8(char(cache.level_max + 1));
#pragma omp parallel for schedule(static) if(nq>query_tile)
  for (size_t q0=0; q0<nq; q0+=query_tile) {
    const size_t qn = std::min(query_tile, nq-q0);
    for (size_t block=0; block<cache.blocks; ++block) {
      __m512i sums[query_tile];
      for (size_t t=0; t<qn; ++t) sums[t]=_mm512_setzero_si512();
      const int8_t* base = cache.levels.data() + block*cache.groups*64;
      for (size_t g=0; g<cache.groups; ++g) {
        const __m512i levels = _mm512_add_epi8(_mm512_loadu_si512(base + g*64), bias);
        for (size_t t=0; t<qn; ++t)
          sums[t] = _mm512_add_epi32(sums[t],
              _mm512_madd_epi16(_mm512_maddubs_epi16(levels, _mm512_set1_epi32(queries[q0+t].qwords[g])), ones));
      }
      const __m512i cid = _mm512_cvtepu8_epi32(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(cache.centroid.data()+block*16)));
      const __m512 correction = _mm512_loadu_ps(cache.correction.data()+block*16);
      const __m512 factor = _mm512_loadu_ps(cache.factor.data()+block*16);
      const size_t n = std::min<size_t>(16, cache.count-block*16);
      const __mmask16 mask = fast_mask(n);
      for (size_t t=0; t<qn; ++t) {
        const auto& q=queries[q0+t];
        __m512 dots = _mm512_mul_ps(
            _mm512_sub_ps(_mm512_cvtepi32_ps(sums[t]), _mm512_set1_ps(q.offset)),
            _mm512_set1_ps(q.scale));
        __m512 score = _mm512_i32gather_ps(cid, q.centroid_scores.data(), 4);
        score = _mm512_add_ps(score, _mm512_mul_ps(
            _mm512_i32gather_ps(cid, q.correction_weights.data(), 4), correction));
        score = _mm512_add_ps(score, _mm512_mul_ps(factor, dots));
        _mm512_mask_storeu_ps(scores+(q0+t)*cache.count+block*16, mask, score);
      }
    }
  }
}
} // namespace dv::tieredq_detail
