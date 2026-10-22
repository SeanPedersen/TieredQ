// TieredQ's persisted row contract and portable IEEE binary16 conversion.
#pragma once
#include "tieredq/tieredq.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dv::tieredq_detail {
inline void write_code(uint8_t* data, uint32_t offset, unsigned bits, unsigned value) {
  data[offset/8] |= uint8_t(value << (offset%8));
  if (offset%8+bits>8) data[offset/8+1] |= uint8_t(value >> (8-offset%8));
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
inline float half_decode(uint16_t value) {
  const unsigned exponent=(value>>10)&31,mantissa=value&1023;
  const float magnitude=exponent==31 ? (mantissa ? NAN : INFINITY) :
      exponent ? std::ldexp(float(1024+mantissa),int(exponent)-25) : std::ldexp(float(mantissa),-24);
  return value&0x8000 ? -magnitude : magnitude;
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
} // namespace dv::tieredq_detail
