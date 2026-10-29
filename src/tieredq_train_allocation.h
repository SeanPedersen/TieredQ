// Greedy rate-distortion bit allocation over PCA ranks, and the fast-target choice between additive 3-bit and
// 2/4-bit allocations under the padded storage of the x86 nibble-lookup layout.
#pragma once
#include "tieredq_io.h"
#include "tieredq_lookup_layout.h"
#include <algorithm>
#include <array>
#include <climits>
#include <queue>
#include <vector>

namespace dv::tieredq_detail {
constexpr unsigned max_code_bits=8;
// Per-rank variance (descending PCA order) and bit cap, and the distortion of each code width.
struct AllocationModel {
  std::vector<double> variance;
  std::vector<uint32_t> cap;
  std::array<double,max_code_bits+1> error{};
};
using Allocation=std::vector<unsigned>;

// Greedy allocation by gain per stored bit. With skip_three a coordinate moves 2 -> 4 bits in one 2-bit step,
// so no tier has 3 bits; a final bit no step can take is left unused.
inline Allocation greedy_allocation(const AllocationModel& model,uint32_t budget,bool skip_three) {
  struct Gain { double value; int rank; bool operator<(const Gain& b) const { return value<b.value || (value==b.value && rank>b.rank); } };
  const int d=int(model.variance.size());
  Allocation bits(d,0);
  auto step=[&](int rank) { return skip_three && bits[rank]==2 ? 2u : 1u; };
  auto gain=[&](int rank) {
    const unsigned b=bits[rank],s=step(rank);
    return std::max(0.0,model.variance[rank])*(model.error[b]-model.error[b+s])/s;
  };
  std::priority_queue<Gain> gains;
  for (int j=0;j<d;++j) gains.push({gain(j),j});
  for (uint32_t used=0;used<budget;) {
    if (gains.empty()) {
      if (skip_three) break;
      bad("code_bits exceeds the allocation cap");
    }
    const int j=gains.top().rank; gains.pop();
    const unsigned s=step(j);
    if (used+s>budget || bits[j]+s>model.cap[j]) continue;
    bits[j]+=s; used+=s;
    if (bits[j]<model.cap[j]) gains.push({gain(j),j});
  }
  return bits;
}
inline double predicted_distortion(const AllocationModel& model,const Allocation& bits) {
  double total=0;
  for (size_t rank=0;rank<bits.size();++rank) total+=std::max(0.0,model.variance[rank])*model.error[bits[rank]];
  return total;
}
inline unsigned widest_tier(const Allocation& bits) { return bits.empty() ? 0 : *std::max_element(bits.begin(),bits.end()); }
inline bool has_three_bit_tier(const Allocation& bits) { return std::count(bits.begin(),bits.end(),split_code_bits)>0; }
// Bits per vector the nibble lookup cache stores for an allocation whose tiers are all <= 4 bits,
// including nibble and byte padding.
inline uint32_t lookup_stored_bits(const Allocation& bits,bool split) {
  LookupWidthCounts dimensions{};
  for (unsigned b:bits) ++dimensions[b];
  return uint32_t(lookup_group_bytes(lookup_group_count(dimensions,split))*CHAR_BIT);
}
struct FastLookupAllocation { Allocation bits; bool additive_three; };
// Fast target whose dense allocation fits the nibble layout. The additive 3-bit allocation is kept only if its
// padded storage is no more than code_bits (or than a nibble 2/4-bit allocation, whose own padding may already
// exceed it) and it predicts less distortion than the 2/4-bit allocation. Trimming its budget to absorb padding
// loses recall on non-Gaussian data (SIFT, #20), so an overflow falls back instead. The 2/4 steps can reach
// 5 bits, which moves that fallback to packed codes. model.error[3] must be the additive codebook's distortion.
inline FastLookupAllocation fast_lookup_allocation(const AllocationModel& model,uint32_t code_bits,const Allocation& dense) {
  const auto fallback=has_three_bit_tier(dense) ? greedy_allocation(model,code_bits,true) : dense;
  auto bits=greedy_allocation(model,code_bits,false);
  if (widest_tier(bits)>lookup_code_bits || !has_three_bit_tier(bits)) return {fallback,false};
  const bool nibble_fallback=widest_tier(fallback)<=lookup_code_bits;
  const uint32_t limit=nibble_fallback ? std::max(code_bits,lookup_stored_bits(fallback,false)) : code_bits;
  if (lookup_stored_bits(bits,true)>limit || predicted_distortion(model,bits)>=predicted_distortion(model,fallback))
    return {fallback,false};
  return {std::move(bits),true};
}
} // namespace dv::tieredq_detail
