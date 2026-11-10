// FP32 scan filter. Competitive candidates still use the reference double metric.
#include "internal.h"
#include <cmath>
#include <limits>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define GEOMETRY_X86 1
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace geometry_ann {
namespace {
float scalar(const float* a, const float* b, std::size_t n) {
    float sum = 0;
    for (std::size_t j = 0; j < n; ++j) { float delta = a[j] - b[j]; sum += delta * delta; }
    return sum;
}
#ifdef GEOMETRY_X86
// Four double lanes deliberately match l2()'s accumulation order exactly.
__attribute__((target("avx2")))
double centroid_avx2(const float* a, const float* b, std::size_t n) {
    auto sum = _mm256_setzero_pd(); std::size_t j = 0;
    for (; j + 4 <= n; j += 4) {
        auto delta = _mm256_sub_pd(_mm256_cvtps_pd(_mm_loadu_ps(a + j)),
                                  _mm256_cvtps_pd(_mm_loadu_ps(b + j)));
        sum = _mm256_add_pd(sum, _mm256_mul_pd(delta, delta));
    }
    double lanes[4]; _mm256_storeu_pd(lanes, sum);
    double result = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for (; j < n; ++j) { double delta = double(a[j]) - b[j]; result += delta * delta; }
    return result;
}
__attribute__((target("avx512f")))
float avx512(const float* a, const float* b, std::size_t n) {
    __m512 x = _mm512_setzero_ps(), y = x;
    std::size_t j = 0;
    for (; j + 32 <= n; j += 32) {
        auto dx = _mm512_sub_ps(_mm512_loadu_ps(a + j), _mm512_loadu_ps(b + j));
        auto dy = _mm512_sub_ps(_mm512_loadu_ps(a + j + 16), _mm512_loadu_ps(b + j + 16));
        x = _mm512_add_ps(x, _mm512_mul_ps(dx, dx));
        y = _mm512_add_ps(y, _mm512_mul_ps(dy, dy));
    }
    float sum = _mm512_reduce_add_ps(_mm512_add_ps(x, y));
    for (; j < n; ++j) { float delta = a[j] - b[j]; sum += delta * delta; }
    return sum;
}
__attribute__((target("avx2")))
float avx2(const float* a, const float* b, std::size_t n) {
    __m256 x = _mm256_setzero_ps(), y = x;
    std::size_t j = 0;
    for (; j + 16 <= n; j += 16) {
        auto dx = _mm256_sub_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j));
        auto dy = _mm256_sub_ps(_mm256_loadu_ps(a + j + 8), _mm256_loadu_ps(b + j + 8));
        x = _mm256_add_ps(x, _mm256_mul_ps(dx, dx));
        y = _mm256_add_ps(y, _mm256_mul_ps(dy, dy));
    }
    float lanes[8]; _mm256_storeu_ps(lanes, _mm256_add_ps(x, y));
    float sum = 0; for (float v : lanes) sum += v;
    for (; j < n; ++j) { float delta = a[j] - b[j]; sum += delta * delta; }
    return sum;
}
#endif
#if defined(__aarch64__)
float neon(const float* a, const float* b, std::size_t n) {
    auto x = vdupq_n_f32(0), y = x;
    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) {
        auto dx = vsubq_f32(vld1q_f32(a + j), vld1q_f32(b + j));
        auto dy = vsubq_f32(vld1q_f32(a + j + 4), vld1q_f32(b + j + 4));
        x = vaddq_f32(x, vmulq_f32(dx, dx)); y = vaddq_f32(y, vmulq_f32(dy, dy));
    }
    float sum = vaddvq_f32(vaddq_f32(x, y));
    for (; j < n; ++j) { float delta = a[j] - b[j]; sum += delta * delta; }
    return sum;
}
#endif
using Kernel = float (*)(const float*, const float*, std::size_t);
Kernel select_kernel() {
#ifdef GEOMETRY_X86
    if (__builtin_cpu_supports("avx512f")) return avx512;
    if (__builtin_cpu_supports("avx2")) return avx2;
#endif
#if defined(__aarch64__)
    return neon;
#else
    return scalar;
#endif
}
}
double centroid_l2(std::span<const float> a, std::span<const float> b) {
#ifdef GEOMETRY_X86
    static const bool avx2_available = __builtin_cpu_supports("avx2");
    if (avx2_available) return centroid_avx2(a.data(), b.data(), a.size());
#endif
    return l2(a, b);
}
float float_l2(std::span<const float> a, std::span<const float> b) {
    static const Kernel kernel = select_kernel();
    return kernel(a.data(), b.data(), a.size());
}
double scan_lower_bound(std::span<const float> a, std::span<const float> b) {
    static const Kernel kernel = select_kernel();
    double estimate = kernel(a.data(), b.data(), a.size());
    // Nonnegative sums have no cancellation. Allow more than the n+3 rounding
    // steps of subtraction, squaring, and reduction (also covering reference
    // double accumulation). Use min normal, rather than denorm_min, to cover
    // underflow even on hosts with flush-to-zero enabled. Huge dimensions and
    // overflow disable filtering; they must never reject a candidate.
    double error = 8 * (double(a.size()) + 16) * std::numeric_limits<float>::epsilon();
    if (!std::isfinite(estimate) || error >= 0.5) return 0;
    return std::max(0.0, estimate / (1 + error)
        - (double(a.size()) + 16) * std::numeric_limits<float>::min());
}
}
