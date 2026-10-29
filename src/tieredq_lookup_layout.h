// Geometry of the x86 nibble-lookup fast layout, shared by training (padded storage) and the lookup cache.
// Same-width coordinates pack into 4-bit groups (1-bit x4, 2-bit x2, 4-bit x1) and two groups share a byte.
// An additive 3-bit codebook {b} U {b+delta} splits each code into a 2-bit part (levels b) and a 1-bit part
// (levels {0, delta}); both parts pack like ordinary 2- and 1-bit codes, so every group table stays an exact
// sum of per-coordinate levels. Any other 3-bit codebook keeps one code per nibble, padded to 4 bits.
#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace dv::tieredq_detail {
constexpr uint32_t lookup_code_bits=4; // widest tier of the narrow x86 lookup cache
constexpr size_t lookup_pair_groups=2;
constexpr uint32_t split_code_bits=3,split_low_bits=2,split_high_bits=split_code_bits-split_low_bits;
constexpr size_t split_levels=size_t(1)<<split_code_bits;
enum class LookupPart : uint32_t { whole, low, high };
struct AdditiveSplit {
  std::array<uint8_t,split_levels> code{};    // sorted 3-bit code -> low | high << split_low_bits
  std::array<std::vector<float>,2> part_levels; // low part b (ascending), high part {0, delta}
};
// The smallest level must be some b, so each candidate delta is a difference to it; pairing then proceeds
// greedily from the smallest unpaired level. Differences are compared exactly, so only codebooks whose
// sums b+delta are exact in float split.
inline std::optional<AdditiveSplit> additive_three_split(const std::vector<float>& levels) {
  if (levels.size()!=split_levels) return std::nullopt;
  for (size_t candidate=1;candidate<levels.size();++candidate) {
    const float delta=levels[candidate]-levels[0];
    AdditiveSplit split; split.part_levels[1]={0.0f,delta};
    auto& low_levels=split.part_levels[0];
    std::array<bool,split_levels> paired{};
    bool valid=delta>0;
    for (size_t i=0;i<levels.size() && valid;++i) {
      if (paired[i]) continue;
      size_t k=i+1;
      while (k<levels.size() && (paired[k] || levels[k]-levels[i]!=delta)) ++k;
      valid=k<levels.size();
      if (!valid) continue;
      paired[i]=paired[k]=true;
      const auto low=uint8_t(low_levels.size());
      split.code[i]=low; split.code[k]=uint8_t(low|1u<<split_low_bits);
      low_levels.push_back(levels[i]);
    }
    if (valid) return split;
  }
  return std::nullopt;
}
inline uint32_t ceil_div(uint32_t value,uint32_t divisor) { return (value+divisor-1)/divisor; }
inline uint32_t lookup_coordinates_per_group(uint32_t bits) { return lookup_code_bits/bits; }
// Nibble groups of one tier; split 3-bit tiers emit a 2-bit and a 1-bit group run.
inline uint32_t lookup_tier_groups(uint32_t bits,uint32_t dimensions,bool split) {
  if (split && bits==split_code_bits)
    return ceil_div(dimensions,lookup_coordinates_per_group(split_low_bits))+
        ceil_div(dimensions,lookup_coordinates_per_group(split_high_bits));
  return ceil_div(dimensions,lookup_coordinates_per_group(bits));
}
inline size_t lookup_group_bytes(size_t groups) { return (groups+lookup_pair_groups-1)/lookup_pair_groups; }
} // namespace dv::tieredq_detail
