// SQ8 rows for the IVF-Flat store: int8 codes with per-dimension scales and offsets, row = codes * scales + offsets
// (the layout of the SQ8 stores the nested IVF reads). A trained codec spans each dimension's [min, max] with the 256
// levels. Decoding multiplies, then adds (no FMA) on every backend, so AVX-512, AVX2 and scalar decode the same floats.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define TQ_SQ8_X86 1
#endif

namespace tq::tieredq_detail {
constexpr float sq8_code_min=-128,sq8_code_max=127;
constexpr float sq8_code_steps=sq8_code_max-sq8_code_min;

inline void sq8_decode_scalar(const int8_t* codes,size_t rows,size_t d,const float* scales,const float* offsets,
                              float* out) {
  for (size_t r=0;r<rows;++r)
    for (size_t j=0;j<d;++j) out[r*d+j]=float(codes[r*d+j])*scales[j]+offsets[j];
}
#ifdef TQ_SQ8_X86
__attribute__((target("avx512f"))) inline void sq8_decode_avx512(const int8_t* codes,size_t rows,size_t d,
    const float* scales,const float* offsets,float* out) {
  constexpr size_t lanes=16;
  const size_t wide=d/lanes*lanes;
  for (size_t r=0;r<rows;++r) {
    const int8_t* code=codes+r*d; float* row=out+r*d;
    for (size_t j=0;j<wide;j+=lanes) {
      const __m512 level=_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(code+j))));
      _mm512_storeu_ps(row+j,_mm512_add_ps(_mm512_mul_ps(level,_mm512_loadu_ps(scales+j)),_mm512_loadu_ps(offsets+j)));
    }
    for (size_t j=wide;j<d;++j) row[j]=float(code[j])*scales[j]+offsets[j];
  }
}
__attribute__((target("avx2"))) inline void sq8_decode_avx2(const int8_t* codes,size_t rows,size_t d,
    const float* scales,const float* offsets,float* out) {
  constexpr size_t lanes=8;
  const size_t wide=d/lanes*lanes;
  for (size_t r=0;r<rows;++r) {
    const int8_t* code=codes+r*d; float* row=out+r*d;
    for (size_t j=0;j<wide;j+=lanes) {
      const __m256 level=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(code+j))));
      _mm256_storeu_ps(row+j,_mm256_add_ps(_mm256_mul_ps(level,_mm256_loadu_ps(scales+j)),_mm256_loadu_ps(offsets+j)));
    }
    for (size_t j=wide;j<d;++j) row[j]=float(code[j])*scales[j]+offsets[j];
  }
}
#endif

struct SQ8Codec {
  std::vector<float> scales,offsets;

  size_t dimensions() const { return scales.size(); }
  size_t allocated_bytes() const { return (scales.capacity()+offsets.capacity())*sizeof(float); }

  // Levels -128..127 span [low, high] per dimension; a constant dimension decodes to its value.
  static SQ8Codec spanning(const std::vector<float>& low,const std::vector<float>& high) {
    SQ8Codec codec;
    codec.scales.resize(low.size()); codec.offsets.resize(low.size());
    for (size_t j=0;j<low.size();++j) {
      codec.scales[j]=(high[j]-low[j])/sq8_code_steps;
      codec.offsets[j]=low[j]-sq8_code_min*codec.scales[j];
    }
    return codec;
  }
  void encode(const float* row,int8_t* out) const {
    for (size_t j=0;j<dimensions();++j) {
      if (!(scales[j]>0)) { out[j]=0; continue; }
      const float level=std::nearbyint((row[j]-offsets[j])/scales[j]);
      out[j]=int8_t(std::clamp(level,sq8_code_min,sq8_code_max));
    }
  }
  void decode(const int8_t* codes,size_t rows,float* out) const {
    using Kernel=void(*)(const int8_t*,size_t,size_t,const float*,const float*,float*);
    static const Kernel kernel=[] {
#ifdef TQ_SQ8_X86
      if (__builtin_cpu_supports("avx512f") && !std::getenv("TQ_NO_AVX512")) return Kernel(&sq8_decode_avx512);
      if (__builtin_cpu_supports("avx2")) return Kernel(&sq8_decode_avx2);
#endif
      return Kernel(&sq8_decode_scalar);
    }();
    kernel(codes,rows,dimensions(),scales.data(),offsets.data(),out);
  }
};

// Per-dimension [min, max] of the rows seen, for SQ8Codec::spanning.
struct SQ8Range {
  std::vector<float> low,high;
  explicit SQ8Range(size_t d)
      : low(d,std::numeric_limits<float>::infinity()),high(d,-std::numeric_limits<float>::infinity()) {}
  void add(const float* row) {
    for (size_t j=0;j<low.size();++j) { low[j]=std::min(low[j],row[j]); high[j]=std::max(high[j],row[j]); }
  }
  void merge(const SQ8Range& other) {
    for (size_t j=0;j<low.size();++j) { low[j]=std::min(low[j],other.low[j]); high[j]=std::max(high[j],other.high[j]); }
  }
};
} // namespace tq::tieredq_detail
