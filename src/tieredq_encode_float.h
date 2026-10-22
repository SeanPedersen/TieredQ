// Shared FP32 panel multiplication for centroid assignment and encoding; SIMD uses explicit FMA.
#pragma once
#include "tieredq/tieredq.h"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define DV_TIEREDQ_FLOAT_X86 1
#endif
namespace dv::tieredq_detail {
constexpr size_t float_panel_width=TieredQPanels::PANEL,float_tile_rows=8;
inline void float_matrix_scalar(const float* a, size_t rows, size_t lda, const TieredQPanels& b, float* c, size_t ldc) {
  for (size_t p = 0; p < b.padded_columns() / float_panel_width; ++p) {
    const float* panel = b.data.data() + p * b.depth * float_panel_width;
    for (size_t r = 0; r < rows; ++r) {
      float acc[float_panel_width] = {};
      for (size_t k = 0; k < b.depth; ++k) { const float v = a[r * lda + k]; for (size_t j = 0; j < float_panel_width; ++j) acc[j] += v * panel[k * float_panel_width + j]; }
      std::copy_n(acc, float_panel_width, c + r * ldc + p * float_panel_width);
    }
  }
}
#ifdef DV_TIEREDQ_FLOAT_X86
__attribute__((target("avx512f"))) inline void float_matrix_avx512(const float* a, size_t rows, size_t lda, const TieredQPanels& b, float* c, size_t ldc) {
  for (size_t p = 0; p < b.padded_columns() / float_panel_width; ++p) {
    const float* panel = b.data.data() + p * b.depth * float_panel_width;
    for (size_t r0 = 0; r0 < rows; r0 += float_tile_rows) {
      // Short tiles repeat the last row; only valid rows are stored.
      const float* row[float_tile_rows];
      for (size_t i = 0; i < float_tile_rows; ++i) row[i] = a + std::min(r0 + i, rows - 1) * lda;
      __m512 acc[float_tile_rows][3];
      for (size_t i = 0; i < float_tile_rows; ++i) acc[i][0] = acc[i][1] = acc[i][2] = _mm512_setzero_ps();
      for (size_t k = 0; k < b.depth; ++k) {
        const __m512 b0 = _mm512_loadu_ps(panel + k * float_panel_width), b1 = _mm512_loadu_ps(panel + k * float_panel_width + 16), b2 = _mm512_loadu_ps(panel + k * float_panel_width + 32);
        for (size_t i = 0; i < float_tile_rows; ++i) {
          const __m512 v = _mm512_set1_ps(row[i][k]);
          acc[i][0] = _mm512_fmadd_ps(v, b0, acc[i][0]); acc[i][1] = _mm512_fmadd_ps(v, b1, acc[i][1]); acc[i][2] = _mm512_fmadd_ps(v, b2, acc[i][2]);
        }
      }
      for (size_t i = 0; i < std::min(float_tile_rows, rows - r0); ++i) {
        for (size_t h = 0; h < 3; ++h) _mm512_storeu_ps(c + (r0 + i) * ldc + p * float_panel_width + 16 * h, acc[i][h]);
      }
    }
  }
}
// AVX2 tile: float_avx2_rows rows x one third of a panel (3 ymm), 12 accumulators of the 16 registers.
constexpr size_t float_avx2_rows = 4, float_avx2_columns = 24;
__attribute__((target("avx2,fma"))) inline void float_matrix_avx2(const float* a, size_t rows, size_t lda, const TieredQPanels& b, float* c, size_t ldc) {
  for (size_t p = 0; p < b.padded_columns() / float_panel_width; ++p) {
    const float* panel = b.data.data() + p * b.depth * float_panel_width;
    for (size_t r0 = 0; r0 < rows; r0 += float_avx2_rows) {
      const float* row[float_avx2_rows];
      for (size_t i = 0; i < float_avx2_rows; ++i) row[i] = a + std::min(r0 + i, rows - 1) * lda;
      for (size_t c0 = 0; c0 < float_panel_width; c0 += float_avx2_columns) {
        __m256 acc[float_avx2_rows][3];
        for (size_t i = 0; i < float_avx2_rows; ++i) acc[i][0] = acc[i][1] = acc[i][2] = _mm256_setzero_ps();
        for (size_t k = 0; k < b.depth; ++k) {
          const float* bk = panel + k * float_panel_width + c0;
          const __m256 b0 = _mm256_loadu_ps(bk), b1 = _mm256_loadu_ps(bk + 8), b2 = _mm256_loadu_ps(bk + 16);
          for (size_t i = 0; i < float_avx2_rows; ++i) {
            const __m256 v = _mm256_broadcast_ss(row[i] + k);
            acc[i][0] = _mm256_fmadd_ps(v, b0, acc[i][0]); acc[i][1] = _mm256_fmadd_ps(v, b1, acc[i][1]); acc[i][2] = _mm256_fmadd_ps(v, b2, acc[i][2]);
          }
        }
        for (size_t i = 0; i < std::min(float_avx2_rows, rows - r0); ++i)
          for (size_t h = 0; h < 3; ++h) _mm256_storeu_ps(c + (r0 + i) * ldc + p * float_panel_width + c0 + 8 * h, acc[i][h]);
      }
    }
  }
}
#endif
inline void float_matrix(const float* a, size_t rows, size_t lda, const TieredQPanels& b, float* c, size_t ldc,
                         TieredQKernel kernel=TieredQKernel::automatic) {
  if (!rows) return;
  if (kernel==TieredQKernel::scalar) return float_matrix_scalar(a,rows,lda,b,c,ldc);
#ifdef DV_TIEREDQ_FLOAT_X86
  if (kernel==TieredQKernel::avx512 && __builtin_cpu_supports("avx512f"))
    return float_matrix_avx512(a,rows,lda,b,c,ldc);
  if (kernel==TieredQKernel::avx2 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
    return float_matrix_avx2(a,rows,lda,b,c,ldc);
  if (kernel!=TieredQKernel::automatic) throw std::runtime_error("TieredQ: FP32 encoder kernel unavailable");
  static const bool avx512 = __builtin_cpu_supports("avx512f") && !std::getenv("DV_TIEREDQ_NO_AVX512");
  static const bool avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  if (avx512) return float_matrix_avx512(a, rows, lda, b, c, ldc);
  if (avx2) return float_matrix_avx2(a, rows, lda, b, c, ldc);
#endif
  if (kernel!=TieredQKernel::automatic) throw std::runtime_error("TieredQ: FP32 encoder kernel unavailable");
  float_matrix_scalar(a, rows, lda, b, c, ldc);
}
} // namespace dv::tieredq_detail
