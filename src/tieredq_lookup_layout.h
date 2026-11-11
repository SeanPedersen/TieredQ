// Geometry of the x86 nibble-lookup fast layout, shared by training (padded storage) and the lookup cache.
// Same-width coordinates pack into 4-bit groups (1-bit x4, 2-bit x2, 4-bit x1) and two groups share a byte.
// An additive 3-bit codebook {b} U {b+delta} splits each code into a 2-bit part (levels b) and a 1-bit part
// (levels {0, delta}). The parts share groups with the 2- and 1-bit tiers; each group table is the sum of its
// coordinates' own level tables, so scoring stays exact. Any other 3-bit codebook keeps one code per nibble.
// Canonical additive 5-8 bit codebooks (tieredq_additive_books.h) split the same way into 1-, 2- and 4-bit parts, so
// wide tiers score at nibble-lookup cost; their byte tables make that path approximate (#13).
#pragma once
#include "tieredq_additive_books.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <optional>
#include <vector>

namespace tq::tieredq_detail {
constexpr uint32_t lookup_code_bits=4; // widest tier of the narrow x86 lookup cache
constexpr size_t lookup_pair_groups=2;
constexpr uint32_t split_code_bits=3,split_low_bits=2,split_high_bits=split_code_bits-split_low_bits;
constexpr size_t split_levels=size_t(1)<<split_code_bits;
constexpr uint32_t widest_code_bits=8,whole_code_part=UINT32_MAX;
// Widest fine-tier code of fast models (#26); its top bits are a code this layout stores.
constexpr uint32_t widest_fine_code_bits=12;
constexpr uint32_t additive_wide_max_bits=6; // automatic cap of fast IP/cosine training, below fine-section widths (#13)
struct AdditiveSplit {
  std::vector<uint32_t> part_bits;
  std::vector<uint8_t> code;                 // sorted code -> part codes packed from the low bit, in part order
  std::vector<std::vector<float>> part_levels;
};
using AdditiveSplits=std::array<std::optional<AdditiveSplit>,widest_code_bits+1>; // indexed by code width
// The smallest level must be some b, so each candidate delta is a difference to it; pairing then proceeds
// greedily from the smallest unpaired level. Differences are compared exactly, so only codebooks whose
// sums b+delta are exact in float split.
inline std::optional<AdditiveSplit> additive_three_split(const std::vector<float>& levels) {
  if (levels.size()!=split_levels) return std::nullopt;
  for (size_t candidate=1;candidate<levels.size();++candidate) {
    const float delta=levels[candidate]-levels[0];
    AdditiveSplit split; split.part_bits={split_low_bits,split_high_bits}; split.code.resize(split_levels);
    split.part_levels.resize(2); split.part_levels[1]={0.0f,delta};
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
// Canonical book of a wide code width as its parts: every sum in float, each code's parts in mixed radix.
inline AdditiveSplit canonical_wide_split(const AdditiveBookParts& book,std::vector<float>& sums) {
  AdditiveSplit split; split.part_bits.assign(book.bits.begin(),book.bits.end()); split.part_levels=book.levels;
  sums.assign(1,0.0f); std::vector<uint8_t> packed(1,0);
  uint32_t shift=0;
  for (size_t part=0;part<book.bits.size();++part) {
    std::vector<float> next_sums; std::vector<uint8_t> next_packed;
    for (size_t code=0;code<book.levels[part].size();++code) for (size_t i=0;i<sums.size();++i) {
      next_sums.push_back(sums[i]+book.levels[part][code]);
      next_packed.push_back(uint8_t(packed[i]|code<<shift));
    }
    sums=std::move(next_sums); packed=std::move(next_packed); shift+=book.bits[part];
  }
  std::vector<size_t> order(sums.size()); std::iota(order.begin(),order.end(),size_t(0));
  std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b) { return sums[a]<sums[b]; });
  split.code.resize(order.size());
  std::vector<float> sorted(order.size());
  for (size_t i=0;i<order.size();++i) { split.code[i]=packed[order[i]]; sorted[i]=sums[order[i]]; }
  sums=std::move(sorted);
  return split;
}
inline const std::vector<float>& additive_wide_codebook(uint32_t bits) {
  static const auto books=[] {
    std::array<std::vector<float>,widest_code_bits+1> all;
    for (uint32_t b=lookup_code_bits+1;b<=widest_code_bits;++b) canonical_wide_split(additive_book_parts()[b],all[b]);
    return all;
  }();
  return books.at(bits);
}
// Splits of a model's codebooks: the 3-bit detection above, and wide tables only when they equal the canonical book.
inline AdditiveSplits additive_splits(const std::vector<std::vector<float>>& levels) {
  AdditiveSplits splits;
  if (levels.size()<=split_code_bits) return splits;
  splits[split_code_bits]=additive_three_split(levels[split_code_bits]);
  for (uint32_t b=lookup_code_bits+1;b<=widest_code_bits && b<levels.size();++b) {
    std::vector<float> sums;
    auto split=canonical_wide_split(additive_book_parts()[b],sums);
    if (sums==levels[b]) splits[b]=std::move(split);
  }
  return splits;
}
inline uint32_t ceil_div(uint32_t value,uint32_t divisor) { return (value+divisor-1)/divisor; }
inline uint32_t lookup_coordinates_per_group(uint32_t bits) { return lookup_code_bits/bits; }
using LookupWidthCounts=std::array<uint32_t,lookup_code_bits+1>; // coordinates per code width, one tier each
// Nibble groups of a model; split 3-bit codes join the 2- and 1-bit streams.
inline uint32_t lookup_group_count(const LookupWidthCounts& dimensions,bool split) {
  auto stream=[&](uint32_t bits,uint32_t count) { return ceil_div(count,lookup_coordinates_per_group(bits)); };
  if (!split) {
    uint32_t groups=0;
    for (uint32_t b=1;b<=lookup_code_bits;++b) groups+=stream(b,dimensions[b]);
    return groups;
  }
  const uint32_t three=dimensions[split_code_bits];
  return stream(lookup_code_bits,dimensions[lookup_code_bits])+stream(split_low_bits,dimensions[split_low_bits]+three)+
      stream(split_high_bits,dimensions[split_high_bits]+three);
}
inline size_t lookup_group_bytes(size_t groups) { return (groups+lookup_pair_groups-1)/lookup_pair_groups; }
} // namespace tq::tieredq_detail
