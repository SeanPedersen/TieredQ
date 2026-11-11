// Small centered covariance: SIMD spans output columns while every entry keeps row-ordered sums.
#pragma once
#include <algorithm>
#include <array>
#include <vector>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace tq::tieredq_detail {
inline std::vector<double> small_covariance_reference(const std::vector<double>& centered,size_t dimensions,size_t rows) {
  std::vector<double> output(dimensions*dimensions);
  const double alpha=1.0/double(rows-1);
  for (size_t i=0;i<dimensions;++i) for (size_t j=0;j<=i;++j) {
    double sum=0;
    for (size_t row=0;row<rows;++row) sum+=centered[row*dimensions+i]*centered[row*dimensions+j];
    output[j*dimensions+i]=sum*alpha;
  }
  return output;
}
#if defined(__x86_64__) || defined(__i386__)
inline constexpr size_t covariance_tile_rows=8,covariance_tile_columns=16;

// Keep intrinsics out of OpenMP's outlined function, which does not inherit Clang target attributes.
__attribute__((target("avx512f"))) inline void small_covariance_tile_avx512(
    const std::vector<double>& centered,size_t dimensions,size_t rows,
    size_t first_row,size_t first_column,double* output) {
  constexpr size_t tile_rows=covariance_tile_rows,tile_columns=covariance_tile_columns,lanes=8;
  const double alpha=1.0/double(rows-1);
  const size_t width=std::min(tile_columns,dimensions-first_column);
  const __mmask8 low_mask=__mmask8((1u<<std::min(lanes,width))-1);
  const __mmask8 high_mask=width>lanes ? __mmask8((1u<<(width-lanes))-1) : 0;
  std::array<size_t,tile_rows> indices{};
  for (size_t i=0;i<tile_rows;++i) indices[i]=std::min(first_row+i,dimensions-1);
  __m512d sums[tile_rows][2];
  for (auto& row:sums) for (auto& sum:row) sum=_mm512_setzero_pd();
  for (size_t r=0;r<rows;++r) {
    const double* row=centered.data()+r*dimensions;
    const __m512d low=_mm512_maskz_loadu_pd(low_mask,row+first_column);
    const __m512d high=_mm512_maskz_loadu_pd(high_mask,row+first_column+std::min(lanes,width));
#pragma GCC unroll 8
    for (size_t i=0;i<tile_rows;++i) {
      const __m512d value=_mm512_set1_pd(row[indices[i]]);
      sums[i][0]=_mm512_add_pd(sums[i][0],_mm512_mul_pd(value,low));
      sums[i][1]=_mm512_add_pd(sums[i][1],_mm512_mul_pd(value,high));
    }
  }
  for (size_t i=0;i<std::min(tile_rows,dimensions-first_row);++i) {
    std::array<double,tile_columns> values{};
    _mm512_storeu_pd(values.data(),sums[i][0]); _mm512_storeu_pd(values.data()+lanes,sums[i][1]);
    for (size_t j=0;j<width && first_column+j<=first_row+i;++j)
      output[(first_column+j)*dimensions+first_row+i]=values[j]*alpha;
  }
}

inline std::vector<double> small_covariance_avx512(
    const std::vector<double>& centered,size_t dimensions,size_t rows) {
  const size_t row_tiles=(dimensions+covariance_tile_rows-1)/covariance_tile_rows;
  const size_t column_tiles=(dimensions+covariance_tile_columns-1)/covariance_tile_columns;
  std::vector<double> output(dimensions*dimensions);
#pragma omp parallel for schedule(static)
  for (size_t task=0;task<row_tiles*column_tiles;++task) {
    const size_t first_row=task/column_tiles*covariance_tile_rows;
    const size_t first_column=task%column_tiles*covariance_tile_columns;
    if (first_column>=first_row+covariance_tile_rows) continue;
    small_covariance_tile_avx512(centered,dimensions,rows,first_row,first_column,output.data());
  }
  return output;
}
#endif
} // namespace tq::tieredq_detail
