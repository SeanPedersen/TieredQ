// Shared panel multiplication for centroid assignment, encoding and coarse list ranking; SIMD uses explicit FMA.
// Panels hold FP32 (float_matrix) or IEEE binary16 (half_matrix, half the bytes per query for coarse ranking, #41);
// both widen to FP32 before the FMA, so each element type gives one result on every backend.
#pragma once
#include "tieredq/tieredq.h"
#include "tieredq_layout.h"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define DV_TIEREDQ_FLOAT_X86 1
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace dv::tieredq_detail {
constexpr size_t float_panel_width=TieredQPanels::PANEL,float_tile_rows=8;
// Column panels as packed by pack_panel_values: element (k, c) at (c / PANEL * depth + k) * PANEL + c % PANEL.
template<class T> struct PanelSpan { const T* data; size_t depth, padded_columns; };
// Binary16 panels for half_matrix (raw IEEE bits), the FP32 TieredQPanels counterpart.
struct HalfPanels {
  size_t depth = 0, columns = 0;
  std::vector<uint16_t> data;
  size_t padded_columns() const { return (columns + TieredQPanels::PANEL - 1) / TieredQPanels::PANEL * TieredQPanels::PANEL; }
  PanelSpan<uint16_t> span() const { return {data.data(), depth, padded_columns()}; }
};
inline PanelSpan<float> span(const TieredQPanels& b) { return {b.data.data(), b.depth, b.padded_columns()}; }
// Calls tile(std::integral_constant<size_t, rows>()) for 1 <= rows <= Max, so a short tile computes only its own
// rows: each row accumulates independently, and a repeated row would redo its whole panel (single-query coarse
// ranking otherwise multiplied one row float_tile_rows times).
template<size_t Max,class Tile> inline void with_tile_rows(size_t rows,Tile&& tile) {
  [&]<size_t... R>(std::index_sequence<R...>) {
    ((rows==R+1 ? tile(std::integral_constant<size_t,R+1>()) : void()),...);
  }(std::make_index_sequence<Max>());
}

inline float panel_value(float value) { return value; }
inline float panel_value(uint16_t value) { return half_decode(value); }
template<class T> void panel_matrix_scalar(const float* a, size_t rows, size_t lda, PanelSpan<T> b, float* c, size_t ldc) {
  for (size_t p = 0; p < b.padded_columns / float_panel_width; ++p) {
    const T* panel = b.data + p * b.depth * float_panel_width;
    for (size_t r = 0; r < rows; ++r) {
      float acc[float_panel_width] = {};
      for (size_t k = 0; k < b.depth; ++k) { const float v = a[r * lda + k]; for (size_t j = 0; j < float_panel_width; ++j) acc[j] += v * panel_value(panel[k * float_panel_width + j]); }
      std::copy_n(acc, float_panel_width, c + r * ldc + p * float_panel_width);
    }
  }
}
#ifdef DV_TIEREDQ_FLOAT_X86
__attribute__((target("avx512f"))) inline __m512 load_panel_avx512(const float* p) { return _mm512_loadu_ps(p); }
__attribute__((target("avx512f"))) inline __m512 load_panel_avx512(const uint16_t* p) {
  return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
}
// Rows x one panel (3 zmm wide).
template<size_t Rows,class T>
__attribute__((target("avx512f"))) inline void panel_tile_avx512(const float* a, size_t lda, const T* panel, size_t depth, float* c, size_t ldc) {
  __m512 acc[Rows][3];
  for (size_t i = 0; i < Rows; ++i) acc[i][0] = acc[i][1] = acc[i][2] = _mm512_setzero_ps();
  for (size_t k = 0; k < depth; ++k) {
    const T* bk = panel + k * float_panel_width;
    const __m512 b0 = load_panel_avx512(bk), b1 = load_panel_avx512(bk + 16), b2 = load_panel_avx512(bk + 32);
    for (size_t i = 0; i < Rows; ++i) {
      const __m512 v = _mm512_set1_ps(a[i * lda + k]);
      acc[i][0] = _mm512_fmadd_ps(v, b0, acc[i][0]); acc[i][1] = _mm512_fmadd_ps(v, b1, acc[i][1]); acc[i][2] = _mm512_fmadd_ps(v, b2, acc[i][2]);
    }
  }
  for (size_t i = 0; i < Rows; ++i)
    for (size_t h = 0; h < 3; ++h) _mm512_storeu_ps(c + i * ldc + 16 * h, acc[i][h]);
}
template<class T>
__attribute__((target("avx512f"))) inline void panel_matrix_avx512(const float* a, size_t rows, size_t lda, PanelSpan<T> b, float* c, size_t ldc) {
  const size_t full = rows / float_tile_rows * float_tile_rows;
  for (size_t p = 0; p < b.padded_columns / float_panel_width; ++p) {
    const T* panel = b.data + p * b.depth * float_panel_width;
    float* out = c + p * float_panel_width;
    for (size_t r0 = 0; r0 < full; r0 += float_tile_rows)
      panel_tile_avx512<float_tile_rows>(a + r0 * lda, lda, panel, b.depth, out + r0 * ldc, ldc);
    if (full < rows) with_tile_rows<float_tile_rows - 1>(rows - full, [&](auto tile) {
      panel_tile_avx512<decltype(tile)::value>(a + full * lda, lda, panel, b.depth, out + full * ldc, ldc);
    });
  }
}
// AVX2 tile: float_avx2_rows rows x one third of a panel (3 ymm), 12 accumulators of the 16 registers. F16C is in
// the target for the binary16 loads; every AVX2 CPU has it, and FP32 panels never execute it.
constexpr size_t float_avx2_rows = 4, float_avx2_columns = 24;
__attribute__((target("avx2,fma,f16c"))) inline __m256 load_panel_avx2(const float* p) { return _mm256_loadu_ps(p); }
__attribute__((target("avx2,fma,f16c"))) inline __m256 load_panel_avx2(const uint16_t* p) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}
template<size_t Rows,class T>
__attribute__((target("avx2,fma,f16c"))) inline void panel_tile_avx2(const float* a, size_t lda, const T* panel, size_t depth, float* c, size_t ldc) {
  for (size_t c0 = 0; c0 < float_panel_width; c0 += float_avx2_columns) {
    __m256 acc[Rows][3];
    for (size_t i = 0; i < Rows; ++i) acc[i][0] = acc[i][1] = acc[i][2] = _mm256_setzero_ps();
    for (size_t k = 0; k < depth; ++k) {
      const T* bk = panel + k * float_panel_width + c0;
      const __m256 b0 = load_panel_avx2(bk), b1 = load_panel_avx2(bk + 8), b2 = load_panel_avx2(bk + 16);
      for (size_t i = 0; i < Rows; ++i) {
        const __m256 v = _mm256_broadcast_ss(a + i * lda + k);
        acc[i][0] = _mm256_fmadd_ps(v, b0, acc[i][0]); acc[i][1] = _mm256_fmadd_ps(v, b1, acc[i][1]); acc[i][2] = _mm256_fmadd_ps(v, b2, acc[i][2]);
      }
    }
    for (size_t i = 0; i < Rows; ++i)
      for (size_t h = 0; h < 3; ++h) _mm256_storeu_ps(c + i * ldc + c0 + 8 * h, acc[i][h]);
  }
}
template<class T>
__attribute__((target("avx2,fma,f16c"))) inline void panel_matrix_avx2(const float* a, size_t rows, size_t lda, PanelSpan<T> b, float* c, size_t ldc) {
  const size_t full = rows / float_avx2_rows * float_avx2_rows;
  for (size_t p = 0; p < b.padded_columns / float_panel_width; ++p) {
    const T* panel = b.data + p * b.depth * float_panel_width;
    float* out = c + p * float_panel_width;
    for (size_t r0 = 0; r0 < full; r0 += float_avx2_rows)
      panel_tile_avx2<float_avx2_rows>(a + r0 * lda, lda, panel, b.depth, out + r0 * ldc, ldc);
    if (full < rows) with_tile_rows<float_avx2_rows - 1>(rows - full, [&](auto tile) {
      panel_tile_avx2<decltype(tile)::value>(a + full * lda, lda, panel, b.depth, out + full * ldc, ldc);
    });
  }
}
#endif
#if defined(__aarch64__)
// Four rows x 24 columns per tile: 24 accumulators, six basis vectors and one broadcast fit the register file.
constexpr size_t float_neon_rows=4,float_neon_columns=24;
inline float32x4_t load_panel_neon(const float* p) { return vld1q_f32(p); }
inline float32x4_t load_panel_neon(const uint16_t* p) { return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p))); }
template<size_t Rows,class T>
inline void panel_tile_neon(const float* a, size_t lda, const T* panel, size_t depth, float* c, size_t ldc) {
  for (size_t c0=0;c0<float_panel_width;c0+=float_neon_columns) {
    float32x4_t acc[Rows][float_neon_columns/4];
    for (auto& r:acc) for (auto& v:r) v=vdupq_n_f32(0.0f);
    for (size_t k=0;k<depth;++k) {
      const T* bk=panel+k*float_panel_width+c0;
      float32x4_t basis[float_neon_columns/4];
      for (size_t h=0;h<float_neon_columns/4;++h) basis[h]=load_panel_neon(bk+4*h);
      for (size_t i=0;i<Rows;++i) {
        const float32x4_t value=vdupq_n_f32(a[i*lda+k]);
        for (size_t h=0;h<float_neon_columns/4;++h) acc[i][h]=vfmaq_f32(acc[i][h],value,basis[h]);
      }
    }
    for (size_t i=0;i<Rows;++i) for (size_t h=0;h<float_neon_columns/4;++h) vst1q_f32(c+i*ldc+c0+4*h,acc[i][h]);
  }
}
template<class T> void panel_matrix_neon(const float* a, size_t rows, size_t lda, PanelSpan<T> b, float* c, size_t ldc) {
  const size_t full=rows/float_neon_rows*float_neon_rows;
  for (size_t p=0;p<b.padded_columns/float_panel_width;++p) {
    const T* panel=b.data+p*b.depth*float_panel_width;
    float* out=c+p*float_panel_width;
    for (size_t r0=0;r0<full;r0+=float_neon_rows) panel_tile_neon<float_neon_rows>(a+r0*lda,lda,panel,b.depth,out+r0*ldc,ldc);
    if (full<rows) with_tile_rows<float_neon_rows-1>(rows-full,[&](auto tile) {
      panel_tile_neon<decltype(tile)::value>(a+full*lda,lda,panel,b.depth,out+full*ldc,ldc);
    });
  }
}
#endif
// Packs b(k, c) for k < depth, c < columns into PANEL-wide column panels (zero padded) of element type T.
template<class T,class F> std::vector<T> pack_panel_values(size_t depth, size_t columns, F at) {
  constexpr size_t PANEL = TieredQPanels::PANEL;
  const size_t padded = (columns + PANEL - 1) / PANEL * PANEL;
  if (columns && depth > std::numeric_limits<size_t>::max() / padded) throw std::runtime_error("TieredQ: size overflow");
  std::vector<T> data(depth * padded, T(0));
  for (size_t c = 0; c < columns; ++c) for (size_t k = 0; k < depth; ++k) data[(c / PANEL * depth + k) * PANEL + c % PANEL] = at(k, c);
  return data;
}
template<class F> TieredQPanels pack_panels(size_t depth, size_t columns, F at) {
  TieredQPanels p; p.depth = depth; p.columns = columns;
  p.data = pack_panel_values<float>(depth, columns, at);
  return p;
}
template<class T> void panel_matrix(const float* a, size_t rows, size_t lda, PanelSpan<T> b, float* c, size_t ldc,
                                    TieredQKernel kernel) {
  if (!rows) return;
  if (kernel==TieredQKernel::scalar) return panel_matrix_scalar(a,rows,lda,b,c,ldc);
#if defined(__aarch64__)
  if (kernel==TieredQKernel::neon) return panel_matrix_neon(a,rows,lda,b,c,ldc);
  if (kernel!=TieredQKernel::automatic) throw std::runtime_error("TieredQ: FP32 encoder kernel unavailable");
#endif
#ifdef DV_TIEREDQ_FLOAT_X86
  static const bool has_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
  if (kernel==TieredQKernel::avx512 && __builtin_cpu_supports("avx512f"))
    return panel_matrix_avx512(a,rows,lda,b,c,ldc);
  if (kernel==TieredQKernel::avx2 && has_avx2)
    return panel_matrix_avx2(a,rows,lda,b,c,ldc);
  if (kernel!=TieredQKernel::automatic) throw std::runtime_error("TieredQ: FP32 encoder kernel unavailable");
  static const bool avx512 = __builtin_cpu_supports("avx512f") && !std::getenv("DV_TIEREDQ_NO_AVX512");
  if (avx512) return panel_matrix_avx512(a, rows, lda, b, c, ldc);
  if (has_avx2) return panel_matrix_avx2(a, rows, lda, b, c, ldc);
#endif
  if (kernel!=TieredQKernel::automatic) throw std::runtime_error("TieredQ: FP32 encoder kernel unavailable");
  panel_matrix_scalar(a, rows, lda, b, c, ldc);
}
// c[r * ldc + j] = a.row(r) . b.column(j) for r < rows, j < b.padded_columns().
inline void float_matrix(const float* a, size_t rows, size_t lda, const TieredQPanels& b, float* c, size_t ldc,
                         TieredQKernel kernel=TieredQKernel::automatic) {
  panel_matrix(a, rows, lda, span(b), c, ldc, kernel);
}
inline void half_matrix(const float* a, size_t rows, size_t lda, const HalfPanels& b, float* c, size_t ldc,
                        TieredQKernel kernel=TieredQKernel::automatic) {
  panel_matrix(a, rows, lda, b.span(), c, ldc, kernel);
}
} // namespace dv::tieredq_detail
