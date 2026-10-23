// In-memory linear map from query coordinates to weights and centroid terms for approximate scans.
// Output panels keep each reduction ordered while vectorizing independent outputs.
// Folding constants changes FP32 rounding, so this map is confined to approximate scans.
#pragma once
#include <algorithm>
#include <cmath>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif
#include <stdexcept>
#include <vector>

namespace dv::tieredq_detail {
constexpr size_t query_projection_lanes = 16;
struct FastQueryProjection {
  size_t dimensions = 0, weights = 0, centroids = 0, width = 0;
  std::vector<float> panels;
};
inline FastQueryProjection build_fast_query_projection(size_t dimensions,
    const std::vector<float>& basis, const std::vector<float>& sigma,
    const std::vector<float>& mean, const std::vector<float>& centroids,
    const std::vector<float>& half_norms) {
  FastQueryProjection map;
  map.dimensions=dimensions; map.weights=sigma.size(); map.centroids=half_norms.size();
  const size_t outputs=map.weights+2*map.centroids;
  map.width=(outputs+query_projection_lanes-1)/query_projection_lanes*query_projection_lanes;
  map.panels.assign(dimensions*map.width,0);
  for (size_t i=0;i<dimensions;++i) {
    float correction=0;
    for (size_t r=0;r<map.weights;++r) correction+=basis[i*map.weights+r]*mean[r];
    for (size_t col=0;col<outputs;++col) {
      float value;
      if (col<map.weights) value=basis[i*map.weights+col]*sigma[col];
      else if (col<map.weights+map.centroids) value=centroids[(col-map.weights)*dimensions+i]+correction;
      else {
        const size_t cid=col-map.weights-map.centroids;
        const float norm=2*half_norms[cid];
        value=norm>0 ? centroids[cid*dimensions+i]/norm : 0;
      }
      map.panels[(col/query_projection_lanes*dimensions+i)*query_projection_lanes+col%query_projection_lanes]=value;
    }
  }
  return map;
}
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f")))
inline std::vector<float> project_fast_query_avx512(const FastQueryProjection& map,const float* vector) {
  if (!vector) throw std::runtime_error("TieredQ: null query data");
  for (size_t i=0;i<map.dimensions;++i)
    if (!std::isfinite(vector[i])) throw std::runtime_error("TieredQ: nonfinite query data");
  std::vector<float> output(map.width);
  for (size_t col=0;col<map.width;col+=query_projection_lanes) {
    const float* panel=map.panels.data()+col*map.dimensions;
    __m512 sum=_mm512_setzero_ps();
    for (size_t i=0;i<map.dimensions;++i)
      sum=_mm512_add_ps(sum,_mm512_mul_ps(_mm512_set1_ps(vector[i]),_mm512_loadu_ps(panel+i*query_projection_lanes)));
    _mm512_storeu_ps(output.data()+col,sum);
  }
  return output;
}
#endif
inline std::vector<float> project_fast_query(const FastQueryProjection& map,const float* vector) {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
  if (__builtin_cpu_supports("avx512f")) return project_fast_query_avx512(map,vector);
#endif
  if (!vector) throw std::runtime_error("TieredQ: null query data");
  for (size_t i=0;i<map.dimensions;++i)
    if (!std::isfinite(vector[i])) throw std::runtime_error("TieredQ: nonfinite query data");
  std::vector<float> output(map.width,0);
  for (size_t col=0;col<map.width;++col) for (size_t i=0;i<map.dimensions;++i)
    output[col]+=vector[i]*map.panels[(col/query_projection_lanes*map.dimensions+i)*query_projection_lanes+col%query_projection_lanes];
  return output;
}
} // namespace dv::tieredq_detail
