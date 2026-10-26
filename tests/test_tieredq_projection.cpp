// Compare compact lookup projection with the previous materialized coefficients, including SIMD tails.
#include "tieredq_query_projection_x86.h"
#include <cmath>
#include <stdexcept>

void tieredq_projection_tests() {
  using namespace dv::tieredq_detail;
  constexpr size_t dimensions=37,centroids=19;
  for (size_t weights:{1u,7u,8u,15u,16u,17u,40u,918u}) {
    std::vector<float> basis(dimensions*weights),sigma(weights),mean(weights),centers(dimensions*centroids),norms(centroids),query(dimensions);
    for (size_t i=0;i<basis.size();++i) basis[i]=std::sin(float(i));
    for (size_t i=0;i<weights;++i) { sigma[i]=std::cos(float(i));mean[i]=std::sin(float(i+1)); }
    for (size_t i=0;i<centers.size();++i) centers[i]=std::cos(float(i));
    for (size_t i=0;i<centroids;++i) norms[i]=float(i);
    for (size_t i=0;i<dimensions;++i) query[i]=std::sin(float(i+3));
    const auto map=build_fast_query_projection(dimensions,basis,sigma,mean,centers,norms);
    if (map.panels.size()!=dimensions*((2*centroids+query_projection_lanes-1)/query_projection_lanes*query_projection_lanes))
      throw std::runtime_error("compact projection retains scaled basis panels");
    std::vector<float> expected(weights+map.width,0);
    for (size_t i=0;i<dimensions;++i) {
      float correction=0;
      for (size_t rank=0;rank<weights;++rank) correction+=basis[i*weights+rank]*mean[rank];
      for (size_t rank=0;rank<weights;++rank) expected[rank]+=query[i]*(basis[i*weights+rank]*sigma[rank]);
      for (size_t cid=0;cid<centroids;++cid) {
        expected[weights+cid]+=query[i]*(centers[cid*dimensions+i]+correction);
        const float norm=2*norms[cid];
        expected[weights+centroids+cid]+=query[i]*(norm>0 ? centers[cid*dimensions+i]/norm : 0);
      }
    }
    if (project_fast_query(map,query.data(),{basis,sigma})!=expected)
      throw std::runtime_error("compact projection changes materialized FP32 results");
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    std::vector<float> actual(weights,0);
    if (__builtin_cpu_supports("avx2")) {
      project_fast_weights_avx2(map,query.data(),{basis,sigma},actual.data());
      if (!std::equal(actual.begin(),actual.end(),expected.begin())) throw std::runtime_error("AVX2 regenerated weights differ");
    }
    if (__builtin_cpu_supports("avx512f")) {
      std::fill(actual.begin(),actual.end(),0);
      project_fast_weights_avx512(map,query.data(),{basis,sigma},actual.data());
      if (!std::equal(actual.begin(),actual.end(),expected.begin())) throw std::runtime_error("AVX512 regenerated weights differ");
    }
#endif
  }
}
