// Exact lower bound on squared distances that prunes restricted list assignment (#45). With P the top d' principal
// directions of the lists' rounded centroids (orthonormal rows) and r the part of a vector outside them,
//   |x - c|^2 = |P(x - c)|^2 + |r_x - r_c|^2 >= |Px|^2 + |Pc|^2 - 2 Px.Pc + (|r_x| - |r_c|)^2,
// for any orthonormal P, so a candidate whose bound exceeds a known distance can never be the nearest list. The
// basis only decides how tight the bound is. Candidates are scored on their d' projected coordinates (FP32 panels,
// one GEMM per block of rows) and only survivors are rescored in full, on contiguous binary16 rows.
#pragma once
#include "tieredq_encode_float.h"
#include "tieredq_ivf_coarse.h"
#include "tieredq_train_linalg.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace tq::tieredq_detail {
// row . v for v a binary16 row (both of length depth): the bound's full rescoring of a surviving candidate.
inline float half_row_dot_scalar(const float* row,const uint16_t* halves,size_t depth) {
  float sum=0;
  for (size_t k=0;k<depth;++k) sum+=row[k]*half_decode(halves[k]);
  return sum;
}
#ifdef TQ_FLOAT_X86
__attribute__((target("avx512f"))) inline float half_row_dot_avx512(const float* row,const uint16_t* halves,size_t depth) {
  constexpr size_t lanes=16;
  __m512 even=_mm512_setzero_ps(),odd=_mm512_setzero_ps();
  size_t k=0;
  for (;k+2*lanes<=depth;k+=2*lanes) {
    even=_mm512_fmadd_ps(_mm512_loadu_ps(row+k),
                         _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(halves+k))),even);
    odd=_mm512_fmadd_ps(_mm512_loadu_ps(row+k+lanes),
                        _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(halves+k+lanes))),odd);
  }
  float total=_mm512_reduce_add_ps(_mm512_add_ps(even,odd));
  for (;k<depth;++k) total+=row[k]*half_decode(halves[k]);
  return total;
}
#endif
inline float half_row_dot(const float* row,const uint16_t* halves,size_t depth) {
#ifdef TQ_FLOAT_X86
  static const bool avx512=__builtin_cpu_supports("avx512f") && !std::getenv("TQ_NO_AVX512");
  if (avx512) return half_row_dot_avx512(row,halves,depth);
#endif
  return half_row_dot_scalar(row,halves,depth);
}

// The projection of the bound: basis, per list |Pc|^2 and |r_c|, and projections of rows on demand.
class ProjectionBound {
 public:
  // Fits the top `dimensions` principal directions of the rounded centroids, decoded a chunk at a time.
  ProjectionBound(const CoarseQuantizer& coarse,size_t dimensions) : d_(coarse.dimensions),dp_(dimensions) {
    fit_basis(coarse);
    list_projected_.resize(product(coarse.lists,dp_));
    std::vector<float> chunk(product(std::min(decode_rows,coarse.lists),d_));
    for (size_t first=0;first<coarse.lists;first+=decode_rows) {
      const size_t n=std::min(decode_rows,coarse.lists-first);
      decode(coarse,first,n,chunk.data());
      project(chunk.data(),n,list_projected_.data()+first*dp_);
    }
    list_projected_sq_.resize(coarse.lists); list_residual_.resize(coarse.lists);
    for (size_t c=0;c<coarse.lists;++c) {
      const float sq=squared_norm(list_projected_.data()+c*dp_,dp_);
      list_projected_sq_[c]=sq;
      list_residual_[c]=std::sqrt(std::max(2*coarse.half_norms[c]-sq,0.0f));
    }
  }
  size_t dimensions() const { return dp_; }
  float projected_sq(uint32_t list) const { return list_projected_sq_[list]; }
  float residual(uint32_t list) const { return list_residual_[list]; }

  // FP32 panels of the given lists' projections (depth d').
  TieredQPanels panels(const std::vector<uint32_t>& lists) const {
    return pack_panels(dp_,lists.size(),[&](size_t k,size_t m) { return list_projected_[size_t(lists[m])*dp_+k]; });
  }
  // Frees the per-list projections once every panel is packed.
  void release_projections() { list_projected_=std::vector<float>(); }

  // out (count x d', row-major) = rows (count x d) projected, one sgemm on the caller's BLAS threads.
  void project(const float* rows,size_t count,float* out) const {
    const int mi=int(dp_),ni=int(count),di=int(d_); const float one=1,zero=0; const char transposed='T',normal='N';
    TrainingLinalg::get().gemm(&transposed,&normal,&mi,&ni,&di,&one,basis_.data(),&di,rows,&di,&zero,out,&mi);
  }
  static float squared_norm(const float* v,size_t n) { float sum=0; for (size_t j=0;j<n;++j) sum+=v[j]*v[j]; return sum; }

 private:
  static constexpr size_t decode_rows=4096;

  // Rounded centroids [first, first + n) as FP32 rows.
  void decode(const CoarseQuantizer& coarse,size_t first,size_t n,float* out) const {
    std::vector<uint16_t> halves(d_);
    for (size_t i=0;i<n;++i) {
      coarse.half_row(first+i,halves.data());
      const float scale=coarse.scales[first+i];
      for (size_t k=0;k<d_;++k) out[i*d_+k]=half_decode(halves[k])*scale;
    }
  }

  void fit_basis(const CoarseQuantizer& coarse) {
    const int d=int(d_);
    const size_t lists=coarse.lists;
    std::vector<float> row(d_);
    std::vector<double> mean(d_,0.0);
    for (size_t c=0;c<lists;++c) { decode(coarse,c,1,row.data()); for (size_t j=0;j<d_;++j) mean[j]+=row[j]; }
    for (double& value:mean) value/=double(lists);
    auto covariance=TrainingLinalg::get().covariance_chunked(d,lists,[&](size_t first,size_t n,double* out) {
      std::vector<float> chunk(product(n,d_));
      decode(coarse,first,n,chunk.data());
      for (size_t i=0;i<n;++i) for (size_t j=0;j<d_;++j) out[i*d_+j]=double(chunk[i*d_+j])-mean[j];
    });
    std::vector<double> values(d_);
    const char vectors='V',lower='L'; int work_size=-1,iwork_size=-1,info=0; double work_query=0; int iwork_query=0;
    const auto& linalg=TrainingLinalg::get();
    linalg.eigen(&vectors,&lower,&d,covariance.data(),&d,values.data(),&work_query,&work_size,&iwork_query,&iwork_size,&info);
    work_size=int(work_query); iwork_size=iwork_query;
    std::vector<double> work(static_cast<size_t>(work_size));
    std::vector<int> iwork(static_cast<size_t>(iwork_size));
    linalg.eigen(&vectors,&lower,&d,covariance.data(),&d,values.data(),work.data(),&work_size,iwork.data(),&iwork_size,&info);
    if (info) bad("assignment bound: eigendecomposition failed");
    // Eigenvalues ascend; eigenvector j is column j of the column-major result.
    basis_.resize(product(dp_,d_));
    for (size_t k=0;k<dp_;++k)
      for (size_t j=0;j<d_;++j) basis_[k*d_+j]=float(covariance[(d_-1-k)*d_+j]);
  }

  size_t d_,dp_;
  std::vector<float> basis_;              // d' x d, orthonormal rows
  std::vector<float> list_projected_;     // lists x d' until release_projections()
  std::vector<float> list_projected_sq_;  // |Pc|^2
  std::vector<float> list_residual_;      // |c - P^T P c|
};
} // namespace tq::tieredq_detail
