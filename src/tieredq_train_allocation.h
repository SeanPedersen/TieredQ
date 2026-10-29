// Greedy rate-distortion bit allocation over PCA ranks, and the fast-target choice between additive 3-bit and
// 2/4-bit allocations under the padded storage of the x86 nibble-lookup layout.
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
constexpr unsigned max_code_bits=8;
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
struct FastLookupAllocation { Allocation bits; bool additive_three; };
// Fast target whose dense allocation fits the nibble layout. The additive 3-bit allocation is kept only if its
// padded storage is no more than code_bits (or than a nibble 2/4-bit allocation, whose own padding may already
// exceed it) and it predicts less distortion than the 2/4-bit allocation. Trimming its budget to absorb padding
// loses recall on non-Gaussian data (SIFT, #20), so an overflow falls back instead. The 2/4 steps can reach
// 5 bits, which moves that fallback to packed codes. error[rank][3] must be the additive codebook's distortion.
inline Allocation lookup_fallback(const AllocationModel& model,uint32_t code_bits,const Allocation& dense) {
  return has_three_bit_tier(dense) ? greedy_allocation(model,code_bits,true) : dense;
}
// Padded storage a fast allocation may use: code_bits, or the 2/4-bit fallback's own padded size if larger.
inline uint32_t lookup_storage_limit(const Allocation& fallback,uint32_t code_bits) {
  return widest_tier(fallback)<=lookup_code_bits ? std::max(code_bits,lookup_stored_bits(fallback,false)) : code_bits;
}
inline FastLookupAllocation fast_lookup_allocation(const AllocationModel& model,uint32_t code_bits,const Allocation& dense) {
  const auto fallback=lookup_fallback(model,code_bits,dense);
  auto bits=greedy_allocation(model,code_bits,false);
  if (widest_tier(bits)>lookup_code_bits || !has_three_bit_tier(bits)) return {fallback,false};
  if (lookup_stored_bits(bits,true)>lookup_storage_limit(fallback,code_bits) ||
      predicted_distortion(model,bits)>=predicted_distortion(model,fallback))
    return {fallback,false};
  return {std::move(bits),true};
}
// Exact nibble-layout allocation under padded storage <= limit_bits (#9), for one distortion table shared by all
// ranks (error[3] is the additive codebook's). For fixed per-width counts the best allocation gives wider codes to
// larger variances, and 1-bit codes always help, so enumerating the 4-, 3- and 2-bit counts over ranks sorted by
// variance and filling the remaining padded 1-bit slots is exact. Without 3-bit tiers the layout is plain 1/2/4.
inline FastLookupAllocation exact_lookup_allocation(const std::vector<double>& variance,const DistortionTable& error,uint32_t limit_bits) {
  const uint32_t d=uint32_t(variance.size()),groups=uint32_t(limit_bits/CHAR_BIT*lookup_pair_groups);
  const uint32_t low_slots=lookup_coordinates_per_group(split_low_bits),high_slots=lookup_coordinates_per_group(split_high_bits);
  std::vector<uint32_t> order(d);
  for (uint32_t r=0;r<d;++r) order[r]=r;
  std::stable_sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b) { return variance[a]>variance[b]; });
  std::vector<double> prefix(d+1,0);
  for (uint32_t r=0;r<d;++r) prefix[r+1]=prefix[r]+std::max(0.0,variance[order[r]]);
  auto span=[&](uint32_t begin,uint32_t count) { return prefix[begin+count]-prefix[begin]; };
  struct Best { double distortion=INFINITY; LookupWidthCounts counts{}; } best;
#pragma omp parallel
  {
    Best local;
#pragma omp for schedule(dynamic)
    for (uint32_t four=0;four<=std::min(d,groups);++four)
      for (uint32_t three=0;three<=d-four;++three)
        for (uint32_t two=0;two+three+four<=d;++two) {
          const uint32_t fixed=four+ceil_div(two+three,low_slots);
          if (fixed>groups) break;
          if ((groups-fixed)*high_slots<three) continue;
          const uint32_t coded=four+three+two,one=std::min(d-coded,(groups-fixed)*high_slots-three);
          const double value=error[4]*span(0,four)+error[3]*span(four,three)+error[2]*span(four+three,two)+
                             error[1]*span(coded,one)+error[0]*span(coded+one,d-coded-one);
          if (value<local.distortion) local={value,{d-coded-one,one,two,three,four}};
        }
#pragma omp critical
    if (local.distortion<best.distortion) best=local;
  }
  Allocation bits(d);
  uint32_t position=0;
  for (uint32_t b=lookup_code_bits+1;b-->0;) for (uint32_t i=0;i<best.counts[b];++i) bits[order[position++]]=b;
  return {std::move(bits),best.counts[split_code_bits]>0};
}
} // namespace dv::tieredq_detail
