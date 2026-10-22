// Ordered double GEMM with ordered reductions: SIMD spans outputs, and each output visits depth in ascending order.
#pragma once
#include "tieredq/tieredq.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define TIEREDQ_ENCODING_X86 1
#endif

namespace dv::tieredq_detail {
// Resolves automatic to the best kernel this CPU runs; an explicit kernel the CPU lacks fails fast.
inline TieredQKernel encoding_kernel(TieredQKernel kernel) {
#ifdef TIEREDQ_ENCODING_X86
  const bool avx512=__builtin_cpu_supports("avx512f"),avx2=__builtin_cpu_supports("avx2");
#else
  const bool avx512=false,avx2=false;
#endif
  if (kernel==TieredQKernel::automatic) return avx512 ? TieredQKernel::avx512 : avx2 ? TieredQKernel::avx2 : TieredQKernel::scalar;
  if ((kernel==TieredQKernel::avx512 && !avx512) || (kernel==TieredQKernel::avx2 && !avx2))
    throw std::runtime_error("TieredQ: requested encoder kernel is unavailable");
  return kernel;
}
template<bool centroid=false> inline void ordered_matrix_scalar(const double* input,size_t rows,const TieredQPanels& panels,double* output) {
  constexpr size_t row_tile=4,column_tile=8,panel_width=TieredQPanels::PANEL;
  const size_t stride=panels.padded_columns();
  for (size_t panel=0;panel<stride;panel+=panel_width) for (size_t col=0;col<panel_width;col+=column_tile) {
    const float* basis=panels.data.data()+panel*panels.depth+col;
    for (size_t first=0;first<rows;first+=row_tile) {
      std::array<std::array<double,column_tile>,row_tile> sums{};
      for (size_t k=0;k<panels.depth;++k) {
        for (size_t i=0;i<row_tile;++i) {
          const double value=input[std::min(first+i,rows-1)*panels.depth+k];
#pragma omp simd
          for (size_t j=0;j<column_tile;++j) {
            const double b=basis[k*panel_width+j];
            if constexpr (centroid) sums[i][j]+=b*b/2-value*b;
            else sums[i][j]+=value*b;
          }
        }
      }
      for (size_t i=0;i<std::min(row_tile,rows-first);++i)
        std::copy_n(sums[i].data(),column_tile,output+(first+i)*stride+panel+col);
    }
  }
}
#ifdef TIEREDQ_ENCODING_X86
template<bool centroid=false>
__attribute__((target("avx2"))) inline void ordered_matrix_avx2(
    const double* input,size_t rows,const TieredQPanels& panels,double* output) {
  constexpr size_t row_tile=4,column_tile=8,lanes=4,panel_width=TieredQPanels::PANEL;
  const size_t stride=panels.padded_columns();
  for (size_t panel=0;panel<stride;panel+=panel_width) for (size_t col=0;col<panel_width;col+=column_tile) {
    const float* basis=panels.data.data()+panel*panels.depth+col;
    for (size_t first=0;first<rows;first+=row_tile) {
      __m256d sums[row_tile][column_tile/lanes];
      for (auto& row:sums) for (auto& sum:row) sum=_mm256_setzero_pd();
      const double* row[row_tile];
      for (size_t i=0;i<row_tile;++i) row[i]=input+std::min(first+i,rows-1)*panels.depth;
      for (size_t k=0;k<panels.depth;++k) {
        const __m256d b0=_mm256_cvtps_pd(_mm_loadu_ps(basis+k*panel_width));
        const __m256d b1=_mm256_cvtps_pd(_mm_loadu_ps(basis+k*panel_width+lanes));
        for (size_t i=0;i<row_tile;++i) {
          const __m256d value=_mm256_set1_pd(row[i][k]);
          __m256d term0=_mm256_mul_pd(value,b0),term1=_mm256_mul_pd(value,b1);
          if constexpr (centroid) {
            term0=_mm256_sub_pd(_mm256_mul_pd(_mm256_mul_pd(b0,b0),_mm256_set1_pd(.5)),term0);
            term1=_mm256_sub_pd(_mm256_mul_pd(_mm256_mul_pd(b1,b1),_mm256_set1_pd(.5)),term1);
          }
          sums[i][0]=_mm256_add_pd(sums[i][0],term0);
          sums[i][1]=_mm256_add_pd(sums[i][1],term1);
        }
      }
      for (size_t i=0;i<std::min(row_tile,rows-first);++i) for (size_t h=0;h<column_tile/lanes;++h)
        _mm256_storeu_pd(output+(first+i)*stride+panel+col+h*lanes,sums[i][h]);
    }
  }
}
template<bool centroid=false>
__attribute__((target("avx512f"))) inline void ordered_matrix_avx512(
    const double* input,size_t rows,const TieredQPanels& panels,double* output) {
  constexpr size_t row_tile=4,lanes=8,panel_width=TieredQPanels::PANEL;
  const size_t stride=panels.padded_columns();
  for (size_t panel=0;panel<stride;panel+=panel_width) {
    const float* basis=panels.data.data()+panel*panels.depth;
    for (size_t first=0;first<rows;first+=row_tile) {
      __m512d sums[row_tile][panel_width/lanes];
      for (auto& row:sums) for (auto& sum:row) sum=_mm512_setzero_pd();
      const double* row[row_tile];
      for (size_t i=0;i<row_tile;++i) row[i]=input+std::min(first+i,rows-1)*panels.depth;
      for (size_t k=0;k<panels.depth;++k) {
        for (size_t h=0;h<panel_width/lanes;++h) {
          const __m512d b=_mm512_cvtps_pd(_mm256_loadu_ps(basis+k*panel_width+h*lanes));
          for (size_t i=0;i<row_tile;++i) {
            const __m512d value=_mm512_set1_pd(row[i][k]);
            __m512d term=_mm512_mul_pd(value,b);
            if constexpr (centroid) term=_mm512_sub_pd(_mm512_mul_pd(_mm512_mul_pd(b,b),_mm512_set1_pd(.5)),term);
            sums[i][h]=_mm512_add_pd(sums[i][h],term);
          }
        }
      }
      for (size_t i=0;i<std::min(row_tile,rows-first);++i) for (size_t h=0;h<panel_width/lanes;++h)
        _mm512_storeu_pd(output+(first+i)*stride+panel+h*lanes,sums[i][h]);
    }
  }
}
#endif
template<bool centroid=false> inline void ordered_matrix(const double* input,size_t rows,const TieredQPanels& panels,double* output,
                      TieredQKernel kernel=TieredQKernel::automatic) {
  if (!rows) return;
  switch (encoding_kernel(kernel)) {
#ifdef TIEREDQ_ENCODING_X86
    case TieredQKernel::avx512: return ordered_matrix_avx512<centroid>(input,rows,panels,output);
    case TieredQKernel::avx2: return ordered_matrix_avx2<centroid>(input,rows,panels,output);
#endif
    default: return ordered_matrix_scalar<centroid>(input,rows,panels,output);
  }
}
} // namespace dv::tieredq_detail
