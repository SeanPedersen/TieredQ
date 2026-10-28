// Compare regenerated lookup projections with the previous materialized coefficients, including SIMD tails.
// Weights stay bit-identical; centroid outputs are folded dot products checked within FP32 tolerance.
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
    const auto map=build_fast_query_projection(dimensions,basis,sigma,mean,norms);
    if (map.allocated_bytes()!=dimensions*sizeof(float))
      throw std::runtime_error("compact projection retains per-centroid panels");
    std::vector<float> expected(weights+2*centroids,0),term_magnitude(expected.size(),0);
    for (size_t i=0;i<dimensions;++i) {
      float correction=0;
      for (size_t rank=0;rank<weights;++rank) correction+=basis[i*weights+rank]*mean[rank];
      for (size_t rank=0;rank<weights;++rank) expected[rank]+=query[i]*(basis[i*weights+rank]*sigma[rank]);
      for (size_t cid=0;cid<centroids;++cid) {
        const float shifted=query[i]*(centers[cid*dimensions+i]+correction);
        const float norm=2*norms[cid];
        const float scaled=query[i]*(norm>0 ? centers[cid*dimensions+i]/norm : 0);
        expected[weights+cid]+=shifted; term_magnitude[weights+cid]+=std::fabs(shifted);
        expected[weights+centroids+cid]+=scaled; term_magnitude[weights+centroids+cid]+=std::fabs(scaled);
      }
    }
    const FastQueryBasis source{basis,sigma,centers,norms};
    const auto actual_query=project_fast_query(map,query.data(),source);
    if (!std::equal(actual_query.begin(),actual_query.begin()+weights,expected.begin()))
      throw std::runtime_error("compact projection changes materialized FP32 weight results");
    // Centroid outputs are folded dot products: equal up to FP32 reassociation of a d-term sum.
    constexpr float reassociation_tolerance=1e-5f;
    for (size_t col=weights;col<expected.size();++col)
      if (std::fabs(actual_query[col]-expected[col])>reassociation_tolerance*(term_magnitude[col]+1e-30f))
        throw std::runtime_error("folded centroid projection differs");
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    for (size_t length:{size_t(1),size_t(15),size_t(16),size_t(33),dimensions}) {
      double reference=0,magnitude=0;
      for (size_t i=0;i<length;++i) { reference+=double(query[i])*centers[i]; magnitude+=std::fabs(double(query[i])*centers[i]); }
      auto close=[&](float value) { return std::fabs(value-reference)<=1e-6*(magnitude+1); };
      if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && !close(fast_query_dot_avx2(query.data(),centers.data(),length)))
        throw std::runtime_error("AVX2 centroid dot product differs");
      if (__builtin_cpu_supports("avx512f") && !close(fast_query_dot_avx512(query.data(),centers.data(),length)))
        throw std::runtime_error("AVX512 centroid dot product differs");
    }
    std::vector<float> actual(weights,0);
    if (__builtin_cpu_supports("avx2")) {
      project_fast_weights_avx2(map,query.data(),source,actual.data());
      if (!std::equal(actual.begin(),actual.end(),expected.begin())) throw std::runtime_error("AVX2 regenerated weights differ");
    }
    if (__builtin_cpu_supports("avx512f")) {
      std::fill(actual.begin(),actual.end(),0);
      project_fast_weights_avx512(map,query.data(),source,actual.data());
      if (!std::equal(actual.begin(),actual.end(),expected.begin())) throw std::runtime_error("AVX512 regenerated weights differ");
    }
#endif
  }
}
