// Per-coordinate affine SQ8. Queries stay FP32; no full decoded corpus is stored.
#include "internal.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define SQ8_X86 1
#endif
namespace geometry_ann {
namespace {
float scalar(const float* q,const std::int8_t* code,const float* scales,const float* offsets,std::size_t n) {
    float sum=0;
    for (std::size_t j=0;j<n;++j) { float decoded=float(code[j])*scales[j]; if (offsets) decoded+=offsets[j]; float delta=q[j]-decoded; sum+=delta*delta; }
    return sum;
}
#ifdef SQ8_X86
// Four double lanes exactly match an explicitly decoded FP32 row's L2 metric.
__attribute__((target("avx2")))
double exact_avx2(const float* q,const std::int8_t* code,const float* scales,const float* offsets,std::size_t n) {
    auto sum=_mm256_setzero_pd();std::size_t j=0;
    for (;j+4<=n;j+=4) {
        std::int32_t word;std::memcpy(&word,code+j,sizeof(word));
        auto decoded=_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_cvtsi32_si128(word)));
        decoded=_mm_mul_ps(decoded,_mm_loadu_ps(scales+j));
        if (offsets) decoded=_mm_add_ps(decoded,_mm_loadu_ps(offsets+j));
        auto delta=_mm256_sub_pd(_mm256_cvtps_pd(_mm_loadu_ps(q+j)),_mm256_cvtps_pd(decoded));
        sum=_mm256_add_pd(sum,_mm256_mul_pd(delta,delta));
    }
    double lanes[4];_mm256_storeu_pd(lanes,sum);
    double result=(lanes[0]+lanes[1])+(lanes[2]+lanes[3]);
    for (;j<n;++j) { float decoded=float(code[j])*scales[j];if (offsets) decoded+=offsets[j];double delta=double(q[j])-decoded;result+=delta*delta; }
    return result;
}
__attribute__((target("avx512f")))
float avx512(const float* q,const std::int8_t* code,const float* scales,const float* offsets,std::size_t n) {
    auto x=_mm512_setzero_ps(), y=x; std::size_t j=0;
    for (;j+32<=n;j+=32) {
        auto a=_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(code+j))));
        auto b=_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(code+j+16))));
        a=_mm512_mul_ps(a,_mm512_loadu_ps(scales+j)); b=_mm512_mul_ps(b,_mm512_loadu_ps(scales+j+16));
        if (offsets) { a=_mm512_add_ps(a,_mm512_loadu_ps(offsets+j)); b=_mm512_add_ps(b,_mm512_loadu_ps(offsets+j+16)); }
        a=_mm512_sub_ps(_mm512_loadu_ps(q+j),a); b=_mm512_sub_ps(_mm512_loadu_ps(q+j+16),b);
        x=_mm512_fmadd_ps(a,a,x); y=_mm512_fmadd_ps(b,b,y);
    }
    float sum=_mm512_reduce_add_ps(_mm512_add_ps(x,y));
    for (;j<n;++j) { float decoded=float(code[j])*scales[j]; if (offsets) decoded+=offsets[j]; float delta=q[j]-decoded;sum+=delta*delta; }
    return sum;
}
__attribute__((target("avx2")))
float avx2(const float* q,const std::int8_t* code,const float* scales,const float* offsets,std::size_t n) {
    auto sum=_mm256_setzero_ps();std::size_t j=0;
    for (;j+8<=n;j+=8) {
        auto value=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(code+j))));
        value=_mm256_mul_ps(value,_mm256_loadu_ps(scales+j));
        if (offsets) value=_mm256_add_ps(value,_mm256_loadu_ps(offsets+j));
        auto delta=_mm256_sub_ps(_mm256_loadu_ps(q+j),value);
        sum=_mm256_add_ps(sum,_mm256_mul_ps(delta,delta));
    }
    float lanes[8];_mm256_storeu_ps(lanes,sum);float result=0;for (float v:lanes) result+=v;
    for (;j<n;++j) { float decoded=float(code[j])*scales[j]; if (offsets) decoded+=offsets[j]; float delta=q[j]-decoded;result+=delta*delta; }
    return result;
}
#endif
using Kernel=float(*)(const float*,const std::int8_t*,const float*,const float*,std::size_t);
Kernel select() {
#ifdef SQ8_X86
    if (__builtin_cpu_supports("avx512f")) return avx512;
    if (__builtin_cpu_supports("avx2")) return avx2;
#endif
    return scalar;
}
}
double sq8_l2(std::span<const float> q,std::span<const std::int8_t> code,std::span<const float> scales,std::span<const float> offsets) {
    double lanes[4]={};std::size_t j=0;
    for (;j+4<=q.size();j+=4) for (std::size_t lane=0;lane<4;++lane) {
        float decoded=float(code[j+lane])*scales[j+lane]; if (!offsets.empty()) decoded+=offsets[j+lane];
        double delta=double(q[j+lane])-decoded;lanes[lane]+=delta*delta;
    }
    double sum=(lanes[0]+lanes[1])+(lanes[2]+lanes[3]);
    for (;j<q.size();++j) { float decoded=float(code[j])*scales[j]; if (!offsets.empty()) decoded+=offsets[j]; double delta=double(q[j])-decoded;sum+=delta*delta; }
    return sum;
}
double sq8_centroid_l2(std::span<const float> q,std::span<const std::int8_t> code,std::span<const float> scales,std::span<const float> offsets) {
#ifdef SQ8_X86
    static const bool available=__builtin_cpu_supports("avx2");
    if (available) return exact_avx2(q.data(),code.data(),scales.data(),offsets.empty()?nullptr:offsets.data(),q.size());
#endif
    return sq8_l2(q,code,scales,offsets);
}
double sq8_lower_bound(std::span<const float> q,std::span<const std::int8_t> code,std::span<const float> scales,std::span<const float> offsets) {
    static Kernel kernel=select();double estimate=kernel(q.data(),code.data(),scales.data(),offsets.empty()?nullptr:offsets.data(),q.size());
    double error=8*(double(q.size())+16)*std::numeric_limits<float>::epsilon();
    if (!std::isfinite(estimate)||error>=.5) return 0;
    return std::max(0.0,estimate/(1+error)-(double(q.size())+16)*std::numeric_limits<float>::min());
}
}
