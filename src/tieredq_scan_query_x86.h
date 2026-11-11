// Shared query preparation for AVX2 and AVX-512 fast scans; contains no ISA-specific instructions.
#pragma once
#include "tieredq_scan_lookup_x86.h"
#include "tieredq_fast_dispatch.h"
namespace tq::tieredq_detail {
struct FastScanQueryX86 {
  TieredQQuery codes;
  FastScanLookupQuery lookup;
  std::vector<int32_t> qwords;
  std::vector<float> centroid_scores, correction_weights;
  float scale = 1, offset = 0;
};

inline bool fast_scan_available() {
  return native_fast_scan_available();
}

inline FastScanQueryX86 fast_scan_query_x86(const FastScanCache& cache, const TieredQQuery& q) {
  if (q.weights.size() != cache.dimensions)
    throw std::runtime_error("TieredQ: int8 query weight shape mismatch");
  if (cache.code_lookup()) { FastScanQueryX86 prepared; prepared.codes=q; return prepared; }
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
  if (!cache.lookup) return fast_scan_query_x86(cache,codec.model_query_weights(vector));
  FastScanQueryX86 query;
  query.lookup=prepare_fast_lookup_query(*cache.lookup,vector,FastQueryBasis::from(codec));
  return query;
}

} // namespace tq::tieredq_detail
