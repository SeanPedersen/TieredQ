// Greedy rate-distortion bit allocation over PCA ranks, and the exact fast-target allocation (additive 3-bit tiers
// included) under the padded storage of the x86 nibble-lookup layout.
#pragma once
#include "tieredq_io.h"
#include "tieredq_lookup_layout.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <queue>
#include <vector>

namespace dv::tieredq_detail {
constexpr unsigned max_code_bits=widest_fine_code_bits;
using DistortionTable=std::array<double,max_code_bits+1>;
// Per-rank variance (descending PCA order), bit cap, and distortion of each code width in units of that variance.
struct AllocationModel {
  std::vector<double> variance;
  std::vector<uint32_t> cap;
  std::vector<DistortionTable> error;
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
    return std::max(0.0,model.variance[rank])*(model.error[rank][b]-model.error[rank][b+s])/s;
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
  for (size_t rank=0;rank<bits.size();++rank) total+=std::max(0.0,model.variance[rank])*model.error[rank][bits[rank]];
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
// The 2/4-bit fast allocation of a dense allocation that fits the nibble layout (its 2 -> 4 steps can reach 5 bits).
// Fast targets no longer ship it; it only sets the padded storage limit below, which keeps the stored bytes of the
// pre-#9 rule. error[rank][3] must be the additive codebook's distortion.
inline Allocation lookup_fallback(const AllocationModel& model,uint32_t code_bits,const Allocation& dense) {
  return has_three_bit_tier(dense) ? greedy_allocation(model,code_bits,true) : dense;
}
// Padded storage a fast allocation may use: code_bits, or the 2/4-bit fallback's own padded size if larger.
inline uint32_t lookup_storage_limit(const Allocation& fallback,uint32_t code_bits) {
  return widest_tier(fallback)<=lookup_code_bits ? std::max(code_bits,lookup_stored_bits(fallback,false)) : code_bits;
}
struct FastLookupAllocation { Allocation bits; bool additive_three; };
// Bits per vector stored by a nibble layout whose codes all split into 1-, 2- and 4-bit parts. Codes wider than
// prefix_bits (fine tiers, #26) store their top prefix_bits in that layout and the remaining bits unpadded beside it.
inline uint32_t additive_stored_bits(const Allocation& bits,unsigned prefix_bits=widest_code_bits) {
  LookupWidthCounts parts{};
  uint32_t fine=0;
  for (unsigned b:bits) {
    const unsigned scanned=std::min(b,prefix_bits);
    fine+=b-scanned;
    if (scanned==split_code_bits) { ++parts[split_low_bits]; ++parts[split_high_bits]; }
    else if (scanned<=lookup_code_bits) ++parts[scanned];
    else for (unsigned part:additive_book_parts()[scanned].bits) ++parts[part];
  }
  return uint32_t(lookup_group_bytes(lookup_group_count(parts,false))*CHAR_BIT)+fine;
}
// Greedy allocation under the additive tables, trimmed until its padded nibble storage fits code_bits.
inline Allocation additive_wide_allocation(const AllocationModel& model,uint32_t code_bits,
                                           unsigned prefix_bits=widest_code_bits) {
  for (uint32_t budget=code_bits;;--budget) {
    auto bits=greedy_allocation(model,budget,false);
    if (additive_stored_bits(bits,prefix_bits)<=code_bits || !budget) return bits;
  }
}
// Default power of the variance weights (see TieredQ::train) at code_bits over dimensions model ranks: 1.5 from
// 1 bit per dimension up (#9), falling linearly to 1 (MSE) at 0.5 bits per dimension and below (#22).
inline double default_allocation_power(uint32_t code_bits,uint32_t dimensions) {
  constexpr double flat_power=1.0,full_power=1.5,flat_below=0.5;
  return flat_power+std::clamp(double(code_bits)/dimensions-flat_below,0.0,full_power-flat_power);
}
// Exact nibble-layout allocation of at most max_width bits per rank under padded storage <= limit_bits (#9), for
// one distortion table shared by all ranks that is convex in the code width (error[3] is the additive codebook's). With G = 2 * limit_bits / 8 groups,
// 4-bit codes take a group each, 2-bit codes and 3-bit low parts pair up and 1-bit codes and 3-bit high parts fill
// groups of four, so an allocation fits iff its total bits plus 2 when the count of 2- and 3-bit codes is odd is at
// most 4G. Unit steps 2 and 4 flip that parity. Greedy at 4G maximizes the gain without the parity term. Swapping an
// extra step of an optimum (beyond greedy) for a missing one (below greedy) never loses gain, so case analysis on the
// parity-preserving swaps shows some optimum differs from greedy by at most exchange_window removed and as many
// added unit steps. A DP over ranks inside that window is therefore exact, and takes O(d) after the greedy.
constexpr uint32_t exchange_window=4;
inline FastLookupAllocation exact_lookup_allocation(const std::vector<double>& variance,const DistortionTable& error,
                                                    uint32_t limit_bits,uint32_t max_width=lookup_code_bits) {
  const uint32_t d=uint32_t(variance.size()),capacity=uint32_t(limit_bits/CHAR_BIT*lookup_pair_groups*lookup_code_bits);
  max_width=std::min(max_width,lookup_code_bits);
  const AllocationModel model{variance,std::vector<uint32_t>(d,max_width),std::vector(d,error)};
  const auto greedy=greedy_allocation(model,std::min(capacity,d*max_width),false);
  auto paired=[](unsigned b) { return b==split_low_bits || b==split_code_bits; };
  uint32_t greedy_bits=0; bool odd=false;
  for (unsigned b:greedy) { greedy_bits+=b; odd^=paired(b); }
  // State: removed steps, added steps and pair parity; each rank keeps the choice that reached a state for backtracking.
  constexpr uint32_t side=exchange_window+1,states=side*side*2;
  auto state=[](uint32_t removed,uint32_t added,bool parity) { return (removed*side+added)*2+parity; };
  std::vector<double> cost(states,INFINITY),next(states);
  struct Step { uint8_t width,from; };
  std::vector<std::array<Step,states>> choice(d);
  cost[state(0,0,odd)]=0;
  for (uint32_t rank=0;rank<d;++rank) {
    std::fill(next.begin(),next.end(),INFINITY);
    const double v=std::max(0.0,variance[rank]); const unsigned g=greedy[rank];
    for (uint32_t removed=0;removed<side;++removed) for (uint32_t added=0;added<side;++added) for (bool parity:{false,true}) {
      const double base=cost[state(removed,added,parity)];
      if (base==INFINITY) continue;
      for (unsigned b=0;b<=max_width;++b) {
        const uint32_t r=removed+(b<g ? g-b : 0),a=added+(b>g ? b-g : 0);
        if (r>exchange_window || a>exchange_window) continue;
        const uint32_t to=state(r,a,parity^(paired(b)!=paired(g)));
        const double value=base+v*error[b];
        if (value<next[to]) { next[to]=value; choice[rank][to]={uint8_t(b),uint8_t(state(removed,added,parity))}; }
      }
    }
    std::swap(cost,next);
  }
  const uint32_t unpaired_padding=split_low_bits; // the empty half of the last 2-bit group
  uint32_t best=states;
  for (uint32_t removed=0;removed<side;++removed) for (uint32_t added=0;added<side;++added) for (bool parity:{false,true}) {
    const uint32_t s=state(removed,added,parity);
    if (greedy_bits-removed+added+(parity ? unpaired_padding : 0)<=capacity && cost[s]<INFINITY && (best==states || cost[s]<cost[best])) best=s;
  }
  Allocation bits(d);
  for (uint32_t rank=d;rank-->0;) { bits[rank]=choice[rank][best].width; best=choice[rank][best].from; }
  const bool additive_three=has_three_bit_tier(bits);
  return {std::move(bits),additive_three};
}
} // namespace dv::tieredq_detail
