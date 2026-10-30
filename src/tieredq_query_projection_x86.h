// Query projection for narrow approximate lookup scans, regenerated from the model without serving panels.
// Weight outputs keep their reduction order; centroid outputs are folded dot products (see below).
// Folding constants changes FP32 rounding, so this map is confined to approximate scans.
#pragma once
#include "tieredq/tieredq.h"
#include "tieredq_layout.h"
#include "tieredq_basis_panels.h"
#include <algorithm>
#include <cmath>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif
#include <stdexcept>
#include <vector>

namespace dv::tieredq_detail {
// Borrow only for a query call: caches may outlive, move or copy their model.
// Exactly one of basis (FP32 row-major, format 2) and basis_half (binary16 rank panels, format 3) is populated.
struct FastQueryBasis {
  const std::vector<float>& basis;
  const std::vector<uint16_t>& basis_half;
  const std::vector<float>& sigma;
  const std::vector<float>& centroids;
  const std::vector<float>& half_norms;
  size_t inputs;
  static FastQueryBasis from(const TieredQ& codec) {
    return {codec.basis_,codec.basis_half_,codec.sigma_,codec.centroids_,codec.centroid_half_norms_,codec.model_dimensions()};
  }
  bool half() const { return !basis_half.empty(); }
  float at(size_t input,size_t rank) const {
    return half() ? half_decode(basis_half[basis_panel_position(inputs,input,rank)]) : basis[input*sigma.size()+rank];
  }
};
// Centroid outputs are regenerated from the model per query; only the per-dimension mean shift is kept.
struct FastQueryProjection {
  size_t dimensions = 0, weights = 0, centroids = 0;
  std::vector<float> correction;
  size_t allocated_bytes() const { return correction.capacity()*sizeof(float); }
};
inline FastQueryProjection build_fast_query_projection(size_t dimensions,FastQueryBasis source,
    const std::vector<float>& mean) {
  FastQueryProjection map;
  map.dimensions=dimensions; map.weights=source.sigma.size(); map.centroids=source.half_norms.size();
  map.correction.assign(dimensions,0);
  for (size_t i=0;i<dimensions;++i)
    for (size_t r=0;r<map.weights;++r) map.correction[i]+=source.at(i,r)*mean[r];
  return map;
}
inline void validate_fast_query(const FastQueryProjection& map,const float* vector) {
  if (!vector) throw std::runtime_error("TieredQ: null query data");
  for (size_t i=0;i<map.dimensions;++i)
    if (!std::isfinite(vector[i])) throw std::runtime_error("TieredQ: nonfinite query data");
}
// Both centroid outputs derive from one q·centroid_c dot product: output c adds the query's mean shift and
// output k+c divides by 2·half_norm_c. Rounding differs from the old per-coordinate panels, which only
// feed approximate int8 lookup scans; nothing per-centroid is cached.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma")))
inline float fast_query_dot_avx2(const float* a,const float* b,size_t n) {
  constexpr size_t lanes=8;
  auto low=_mm256_setzero_ps(),high=_mm256_setzero_ps();
  size_t i=0;
  for (;i+2*lanes<=n;i+=2*lanes) {
    low=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),low);
    high=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+lanes),_mm256_loadu_ps(b+i+lanes),high);
  }
  alignas(32) float sums[lanes];
  _mm256_store_ps(sums,_mm256_add_ps(low,high));
  float sum=0;
  for (float value:sums) sum+=value;
  for (;i<n;++i) sum+=a[i]*b[i];
  return sum;
}
__attribute__((target("avx512f")))
inline float fast_query_dot_avx512(const float* a,const float* b,size_t n) {
  constexpr size_t lanes=16;
  auto low=_mm512_setzero_ps(),high=_mm512_setzero_ps();
  size_t i=0;
  for (;i+2*lanes<=n;i+=2*lanes) {
    low=_mm512_fmadd_ps(_mm512_loadu_ps(a+i),_mm512_loadu_ps(b+i),low);
    high=_mm512_fmadd_ps(_mm512_loadu_ps(a+i+lanes),_mm512_loadu_ps(b+i+lanes),high);
  }
  for (;i<n;i+=lanes) {
    const auto mask=__mmask16((1u<<std::min(lanes,n-i))-1);
    low=_mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask,a+i),_mm512_maskz_loadu_ps(mask,b+i),low);
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(low,high));
}
#endif
inline float fast_query_dot(const float* a,const float* b,size_t n) {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
  if (__builtin_cpu_supports("avx512f")) return fast_query_dot_avx512(a,b,n);
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return fast_query_dot_avx2(a,b,n);
#endif
  float sum=0;
  for (size_t i=0;i<n;++i) sum+=a[i]*b[i];
  return sum;
}
inline void project_fast_centroids(const FastQueryProjection& map,const float* vector,
    FastQueryBasis source,float* output) {
  validate_fast_query(map,vector);
  const size_t d=map.dimensions,k=map.centroids;
  const float shift=fast_query_dot(vector,map.correction.data(),d);
  for (size_t cid=0;cid<k;++cid) {
    const float dot=fast_query_dot(vector,source.centroids.data()+cid*d,d);
    const float norm=2*source.half_norms[cid];
    output[cid]=dot+shift;
    output[k+cid]=norm>0 ? dot/norm : 0;
  }
}
// Regenerate basis*sigma before multiplying by the query, preserving the old FP32 rounding. A binary16 basis
// projects through its rank panels (tieredq_basis_panels.h) with the same arithmetic; the FP32 basis of format-2
// models keeps its row order.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
inline void project_fast_weights_avx2(const FastQueryProjection& map,const float* vector,
    FastQueryBasis source,float* output) {
  constexpr size_t lanes=8;
  for (size_t i=0;i<map.dimensions;++i) {
    const auto query=_mm256_set1_ps(vector[i]);
    const size_t row=i*map.weights;
    size_t rank=0;
    for (;rank+lanes<=map.weights;rank+=lanes) {
      const auto coefficient=_mm256_mul_ps(_mm256_loadu_ps(source.basis.data()+row+rank),_mm256_loadu_ps(source.sigma.data()+rank));
      _mm256_storeu_ps(output+rank,_mm256_add_ps(_mm256_loadu_ps(output+rank),_mm256_mul_ps(query,coefficient)));
    }
    for (;rank<map.weights;++rank)
      output[rank]+=vector[i]*(source.basis[row+rank]*source.sigma[rank]);
  }
}
__attribute__((target("avx512f")))
inline void project_fast_weights_avx512(const FastQueryProjection& map,const float* vector,
    FastQueryBasis source,float* output) {
  constexpr size_t lanes=16;
  for (size_t i=0;i<map.dimensions;++i) {
    const auto query=_mm512_set1_ps(vector[i]);
    const size_t row=i*map.weights;
    for (size_t rank=0;rank<map.weights;rank+=lanes) {
      const auto mask=__mmask16((1u<<std::min(lanes,map.weights-rank))-1);
      const auto coefficient=_mm512_mul_ps(_mm512_maskz_loadu_ps(mask,source.basis.data()+row+rank),
          _mm512_maskz_loadu_ps(mask,source.sigma.data()+rank));
      const auto sum=_mm512_add_ps(_mm512_maskz_loadu_ps(mask,output+rank),_mm512_mul_ps(query,coefficient));
      _mm512_mask_storeu_ps(output+rank,mask,sum);
    }
  }
}
#endif
inline void project_fast_weights_fp32(const FastQueryProjection& map,const float* vector,FastQueryBasis source,float* output) {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
  if (__builtin_cpu_supports("avx512f")) return project_fast_weights_avx512(map,vector,source,output);
  if (__builtin_cpu_supports("avx2")) return project_fast_weights_avx2(map,vector,source,output);
#endif
  for (size_t i=0;i<map.dimensions;++i) for (size_t rank=0;rank<map.weights;++rank)
    output[rank]+=vector[i]*(source.basis[i*map.weights+rank]*source.sigma[rank]);
}
// Per query: weights [0, m), centroid scores [m, m + k) and correction weights [m + k, m + 2k). A binary16
// basis projects up to basis_query_tile queries per pass over the basis.
inline std::vector<std::vector<float>> project_fast_queries(const FastQueryProjection& map,const float* vectors,
    size_t rows,FastQueryBasis source) {
  std::vector<std::vector<float>> outputs(rows);
  for (size_t row=0;row<rows;++row) {
    outputs[row].assign(map.weights+2*map.centroids,0);
    project_fast_centroids(map,vectors+row*map.dimensions,source,outputs[row].data()+map.weights);
  }
  if (!source.half()) {
    for (size_t row=0;row<rows;++row) project_fast_weights_fp32(map,vectors+row*map.dimensions,source,outputs[row].data());
    return outputs;
  }
  std::vector<float> weights(std::min(rows,basis_query_tile)*map.weights);
  for (size_t first=0;first<rows;first+=basis_query_tile) {
    const size_t count=std::min(basis_query_tile,rows-first);
    project_basis_panels({source.basis_half.data(),map.dimensions,map.weights,source.sigma.data(),
        vectors+first*map.dimensions,map.dimensions,weights.data(),map.weights},count);
    for (size_t row=0;row<count;++row)
      std::copy_n(weights.data()+row*map.weights,map.weights,outputs[first+row].data());
  }
  return outputs;
}
inline std::vector<float> project_fast_query(const FastQueryProjection& map,const float* vector,FastQueryBasis source) {
  return std::move(project_fast_queries(map,vector,1,source)[0]);
}
} // namespace dv::tieredq_detail
