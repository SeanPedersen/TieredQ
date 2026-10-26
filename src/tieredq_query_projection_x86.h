// Compact centroid map and regenerated weight coefficients for narrow approximate scans.
// Output panels keep each reduction ordered while vectorizing independent outputs.
// Folding constants changes FP32 rounding, so this map is confined to approximate scans.
#pragma once
#include "tieredq/tieredq.h"
#include <algorithm>
#include <cmath>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif
#include <stdexcept>
#include <vector>

namespace dv::tieredq_detail {
constexpr size_t query_projection_lanes = 16;
// Borrow only for a query call: caches may outlive, move or copy their model.
struct FastQueryBasis {
  const std::vector<float>& basis;
  const std::vector<float>& sigma;
  static FastQueryBasis from(const TieredQ& codec) { return {codec.basis_,codec.sigma_}; }
};
struct FastQueryProjection {
  size_t dimensions = 0, weights = 0, centroids = 0, width = 0;
  std::vector<float> panels;
  size_t allocated_bytes() const { return panels.capacity()*sizeof(float); }
};
inline FastQueryProjection build_fast_query_projection(size_t dimensions,
    const std::vector<float>& basis, const std::vector<float>& sigma,
    const std::vector<float>& mean, const std::vector<float>& centroids,
    const std::vector<float>& half_norms) {
  FastQueryProjection map;
  map.dimensions=dimensions; map.weights=sigma.size(); map.centroids=half_norms.size();
  const size_t outputs=2*map.centroids;
  map.width=(outputs+query_projection_lanes-1)/query_projection_lanes*query_projection_lanes;
  map.panels.assign(dimensions*map.width,0);
  for (size_t i=0;i<dimensions;++i) {
    float correction=0;
    for (size_t r=0;r<map.weights;++r) correction+=basis[i*map.weights+r]*mean[r];
    for (size_t col=0;col<outputs;++col) {
      float value;
      if (col<map.centroids) value=centroids[col*dimensions+i]+correction;
      else {
        const size_t cid=col-map.centroids;
        const float norm=2*half_norms[cid];
        value=norm>0 ? centroids[cid*dimensions+i]/norm : 0;
      }
      map.panels[(col/query_projection_lanes*dimensions+i)*query_projection_lanes+col%query_projection_lanes]=value;
    }
  }
  return map;
}
inline void validate_fast_query(const FastQueryProjection& map,const float* vector) {
  if (!vector) throw std::runtime_error("TieredQ: null query data");
  for (size_t i=0;i<map.dimensions;++i)
    if (!std::isfinite(vector[i])) throw std::runtime_error("TieredQ: nonfinite query data");
}
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
inline std::vector<float> project_fast_query_avx2(const FastQueryProjection& map,const float* vector) {
  validate_fast_query(map,vector);
  constexpr size_t lanes=8;
  std::vector<float> output(map.width);
  for (size_t col=0;col<map.width;col+=query_projection_lanes) {
    const float* panel=map.panels.data()+col*map.dimensions;
    auto low=_mm256_setzero_ps(),high=_mm256_setzero_ps();
    for (size_t i=0;i<map.dimensions;++i) {
      const auto value=_mm256_set1_ps(vector[i]);
      low=_mm256_add_ps(low,_mm256_mul_ps(value,_mm256_loadu_ps(panel+i*query_projection_lanes)));
      high=_mm256_add_ps(high,_mm256_mul_ps(value,_mm256_loadu_ps(panel+i*query_projection_lanes+lanes)));
    }
    _mm256_storeu_ps(output.data()+col,low);
    _mm256_storeu_ps(output.data()+col+lanes,high);
  }
  return output;
}
__attribute__((target("avx512f")))
inline std::vector<float> project_fast_query_avx512(const FastQueryProjection& map,const float* vector) {
  validate_fast_query(map,vector);
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
inline std::vector<float> project_fast_centroids(const FastQueryProjection& map,const float* vector) {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
  if (__builtin_cpu_supports("avx512f")) return project_fast_query_avx512(map,vector);
  if (__builtin_cpu_supports("avx2")) return project_fast_query_avx2(map,vector);
#endif
  validate_fast_query(map,vector);
  std::vector<float> output(map.width,0);
  for (size_t col=0;col<map.width;++col) for (size_t i=0;i<map.dimensions;++i)
    output[col]+=vector[i]*map.panels[(col/query_projection_lanes*map.dimensions+i)*query_projection_lanes+col%query_projection_lanes];
  return output;
}
// Regenerate basis*sigma before multiplying by the query, preserving the old FP32 rounding.
// Basis rows are contiguous; vectorizing ranks avoids a second dense serving matrix.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
inline void project_fast_weights_avx2(const FastQueryProjection& map,const float* vector,
    FastQueryBasis source,float* output) {
  constexpr size_t lanes=8;
  for (size_t i=0;i<map.dimensions;++i) {
    const auto query=_mm256_set1_ps(vector[i]);
    size_t rank=0;
    for (;rank+lanes<=map.weights;rank+=lanes) {
      const auto coefficient=_mm256_mul_ps(_mm256_loadu_ps(source.basis.data()+i*map.weights+rank),
          _mm256_loadu_ps(source.sigma.data()+rank));
      const auto sum=_mm256_add_ps(_mm256_loadu_ps(output+rank),_mm256_mul_ps(query,coefficient));
      _mm256_storeu_ps(output+rank,sum);
    }
    for (;rank<map.weights;++rank)
      output[rank]+=vector[i]*(source.basis[i*map.weights+rank]*source.sigma[rank]);
  }
}
__attribute__((target("avx512f")))
inline void project_fast_weights_avx512(const FastQueryProjection& map,const float* vector,
    FastQueryBasis source,float* output) {
  constexpr size_t lanes=16;
  for (size_t i=0;i<map.dimensions;++i) {
    const auto query=_mm512_set1_ps(vector[i]);
    for (size_t rank=0;rank<map.weights;rank+=lanes) {
      const auto mask=__mmask16((1u<<std::min(lanes,map.weights-rank))-1);
      const auto coefficient=_mm512_mul_ps(_mm512_maskz_loadu_ps(mask,source.basis.data()+i*map.weights+rank),
          _mm512_maskz_loadu_ps(mask,source.sigma.data()+rank));
      const auto sum=_mm512_add_ps(_mm512_maskz_loadu_ps(mask,output+rank),_mm512_mul_ps(query,coefficient));
      _mm512_mask_storeu_ps(output+rank,mask,sum);
    }
  }
}
#endif
inline std::vector<float> project_fast_query(const FastQueryProjection& map,const float* vector,FastQueryBasis source) {
  auto centers=project_fast_centroids(map,vector);
  std::vector<float> output(map.weights+centers.size(),0);
  std::copy(centers.begin(),centers.end(),output.begin()+map.weights);
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
  if (__builtin_cpu_supports("avx512f")) project_fast_weights_avx512(map,vector,source,output.data());
  else if (__builtin_cpu_supports("avx2")) project_fast_weights_avx2(map,vector,source,output.data());
  else
#endif
  for (size_t i=0;i<map.dimensions;++i) for (size_t rank=0;rank<map.weights;++rank)
    output[rank]+=vector[i]*(source.basis[i*map.weights+rank]*source.sigma[rank]);
  return output;
}
} // namespace dv::tieredq_detail
