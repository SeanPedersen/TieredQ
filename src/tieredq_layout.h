// TieredQ's persisted row contract and portable IEEE binary16 conversion.
#pragma once
#include "tieredq/tieredq.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace tq::tieredq_detail {
// Any code up to 12 bits (#26), which may span three bytes; the scan kernels' read_code covers at most two.
inline uint32_t read_wide_code(const uint8_t* data, uint32_t offset, unsigned bits) {
  uint32_t window=0;
  for (uint32_t byte=0;byte*8<offset%8+bits;++byte) window|=uint32_t(data[offset/8+byte])<<(8*byte);
  return (window>>(offset%8)) & ((1u<<bits)-1);
}
inline void write_code(uint8_t* data, uint32_t offset, unsigned bits, unsigned value) {
  const uint32_t window=uint32_t(value)<<(offset%8);
  for (uint32_t byte=0;byte*8<offset%8+bits;++byte) data[offset/8+byte] |= uint8_t(window>>(8*byte));
}
constexpr double half_max = 65504;
inline uint16_t half_encode(double value) {
  if (!std::isfinite(value) || std::fabs(value)>half_max)
    throw std::runtime_error("TieredQ: fp16 metadata overflow");
  const uint16_t sign=std::signbit(value) ? 0x8000 : 0;
  const double magnitude=std::fabs(value);
  if (!magnitude) return sign;
  int exponent; std::frexp(magnitude,&exponent);
  const int shift=std::max(exponent-11,-24);
  const double scaled=std::ldexp(magnitude,-shift),floor=std::floor(scaled),fraction=scaled-floor;
  const uint32_t rounded=uint32_t(floor)+(fraction>0.5 || (fraction==0.5 && (uint32_t(floor)&1)));
  if (shift==-24 && rounded<1024) return sign|uint16_t(rounded);
  const uint32_t encoded=uint32_t(shift+24)*1024+rounded;
  return sign|uint16_t(encoded);
}
// Exact bit-level conversion without libm: scans decode per-document metadata and basis halves through it, and an
// out-of-line std::ldexp there cost about half of an ARM fast single-query search.
inline float half_decode(uint16_t value) {
  constexpr unsigned mantissa_bits=10,float_mantissa_shift=23-mantissa_bits,sign_shift=31-15;
  constexpr uint32_t rebias=uint32_t(127-15)<<23,float_infinity=0x7f800000,float_quiet_nan=0x7fc00000;
  constexpr float subnormal_unit=0x1p-24f;
  const unsigned exponent=(value>>mantissa_bits)&31,mantissa=value&1023;
  const uint32_t sign=uint32_t(value&0x8000)<<sign_shift;
  if (exponent==31) return std::bit_cast<float>(sign|(mantissa ? float_quiet_nan : float_infinity));
  if (!exponent) return std::bit_cast<float>(sign|std::bit_cast<uint32_t>(float(mantissa)*subnormal_unit));
  return std::bit_cast<float>(sign|((uint32_t(value&0x7fff)<<float_mantissa_shift)+rebias));
}
// Binary16 to FP32 is exact, so the F16C and portable paths produce identical values.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx,f16c")))
inline void half_decode_row_f16c(const uint16_t* in,float* out,size_t count) {
  constexpr size_t lanes=8;
  size_t i=0;
  for (;i+lanes<=count;i+=lanes)
    _mm256_storeu_ps(out+i,_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(in+i))));
  for (;i<count;++i) out[i]=half_decode(in[i]);
}
#endif
inline void half_decode_row(const uint16_t* in,float* out,size_t count) {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
  if (__builtin_cpu_supports("f16c")) return half_decode_row_f16c(in,out,count);
#endif
  for (size_t i=0;i<count;++i) out[i]=half_decode(in[i]);
}
struct Layout {
  size_t code_bytes;
  size_t metadata_bytes() const { return TieredQ::metadata_bytes; }
  size_t centroid_offset() const { return code_bytes; }
  size_t factor_offset() const { return code_bytes+1; }
  size_t correction_offset() const { return code_bytes+3; }
  uint16_t half(const uint8_t* doc,size_t offset,size_t step=1) const {
    return uint16_t(doc[offset*step]) | uint16_t(doc[(offset+1)*step])<<8;
  }
  float factor(const uint8_t* doc,size_t step=1) const {
    return half_decode(half(doc,factor_offset(),step));
  }
  float correction(const uint8_t* doc,size_t step=1) const {
    return half_decode(half(doc,correction_offset(),step));
  }
  void store_half(uint8_t* doc,size_t offset,double value) const {
    const auto bits=half_encode(value); doc[offset]=uint8_t(bits); doc[offset+1]=uint8_t(bits>>8);
  }
  void validate_payload(const uint8_t* doc) const {
    if (!std::isfinite(factor(doc)) || !std::isfinite(correction(doc)))
      throw std::runtime_error("TieredQ: nonfinite fp16 metadata");
  }
  void validate_weights(size_t weights,size_t centroids) const {
    if (weights!=centroids)
      throw std::runtime_error("TieredQ: query correction weight shape mismatch");
  }
};
} // namespace tq::tieredq_detail
