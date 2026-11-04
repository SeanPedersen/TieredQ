// Prefix-section range checkpoint for nibble byte-table scans (#13). The prefix is the leading sections: by default
// all but a fixed share, else as many as hold the groups of the model's first prefix tiers. The suffix bound comes from the query's own tables, so with confidence 0 it
// is exact for the nibble score, and a positive confidence clips it to that many standard deviations of the suffix
// score for unit-Gaussian coordinates (approximate), as for the code-lookup checkpoint.
#pragma once
#include "tieredq_range_checkpoint.h"
#include "tieredq_scan_lookup_x86.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dv::tieredq_detail {
struct LookupRangeBounds { size_t prefix_sections; FastRangeBounds bounds; };
struct LookupRangeCheckpoint { LookupRangeBounds bounds; float threshold; };
// Kernels without per-instruction directed rounding (NEON, AVX2) evaluate the bound in the default round-to-nearest
// mode and add this multiple of the sum of the bound's absolute terms. Each term passes through at most five roundings
// of relative error epsilon/2, and the slack's own product and the final sum round once more; four epsilons (eight
// half-ulps) cover that with margin, so the bound still rounds outward without touching the FP control register.
constexpr float range_nearest_slack=4*std::numeric_limits<float>::epsilon();

// Share of sections the automatic prefix leaves to the suffix. A larger suffix skips more work per rejected document
// but loosens the bound; the exact bound loosens fastest. Single-query sweeps over 384- to 1536-dimensional sets on
// NEON, AVX2 and AVX-512 (2026-11-04) favored these: exact gained 1.08-1.12x on average with no cell below 0.95x
// (0.25 gained more on NEON and AVX2 but lost up to 11% on AVX-512), confidence 8 gained 1.26-1.43x.
constexpr double lookup_exact_suffix_share=0.125,lookup_confident_suffix_share=0.375;

// Sections before the suffix that cover as many groups as the first prefix_tiers tiers' slots occupy; 0 when no
// suffix is left.
inline size_t lookup_tier_prefix_sections(const FastScanLookupCache& cache,const std::vector<TieredQTier>& tiers,
    size_t prefix_tiers) {
  if (tiers.size()<=prefix_tiers) return 0;
  const uint32_t prefix_rank=tiers[prefix_tiers].start;
  const size_t groups=size_t(std::count_if(cache.groups.begin(),cache.groups.end(),[&](const FastLookupGroup& group) {
    return std::any_of(group.slots.begin(),group.slots.begin()+group.count,[&](const LookupSlot& slot) { return slot.rank<prefix_rank; });
  }));
  const auto& ends=cache.section_ends;
  const size_t prefix=size_t(std::lower_bound(ends.begin(),ends.end(),uint32_t(groups))-ends.begin())+1;
  return prefix<ends.size() ? prefix : 0;
}
// Sections before the suffix; 0 when no suffix is left. The automatic prefix leaves at least the share for the
// bound's confidence, and more where the first fast_range_prefix_tiers tiers already do (the earlier default, faster
// on OpenAI-1536 at 2 bits/dim); explicit prefix_tiers use the tier prefix alone.
inline size_t lookup_prefix_sections(const FastScanLookupCache& cache,const std::vector<TieredQTier>& tiers,
    size_t prefix_tiers,double confidence) {
  if (prefix_tiers!=TieredQPruning::automatic_prefix) return lookup_tier_prefix_sections(cache,tiers,prefix_tiers);
  const size_t sections=cache.section_ends.size();
  const double share=confidence>0 ? lookup_confident_suffix_share : lookup_exact_suffix_share;
  const size_t suffix=std::max<size_t>(1,size_t(std::lround(share*double(sections))));
  const size_t shared=sections>suffix ? sections-suffix : 0;
  const size_t tiered=lookup_tier_prefix_sections(cache,tiers,fast_range_prefix_tiers);
  return tiered && shared ? std::min(tiered,shared) : std::max(tiered,shared);
}
inline bool lookup_range_checkpoint_supported(const FastScanCache& cache,const std::vector<TieredQTier>& tiers,
    size_t prefix_tiers=TieredQPruning::automatic_prefix,double confidence=0) {
  // Pruned scans assume whole lookup blocks; caller block views with a short tail use the unpruned kernels.
  return cache.lookup && cache.lookup->tail_block==SIZE_MAX && lookup_prefix_sections(*cache.lookup,tiers,prefix_tiers,confidence)>0;
}
inline LookupRangeBounds prepare_lookup_range_bounds(const FastScanCache& cache,const std::vector<TieredQTier>& tiers,
    const FastScanLookupQuery& query,size_t prefix_tiers=TieredQPruning::automatic_prefix,double confidence=0) {
  const auto& lookup=*cache.lookup;
  const size_t prefix=lookup_prefix_sections(lookup,tiers,prefix_tiers,confidence);
  if (!prefix || query.scales.size()!=lookup.section_ends.size() || !(confidence>=0) || !std::isfinite(confidence))
    throw std::invalid_argument("TieredQ: incompatible lookup range checkpoint");
  double high=0,low=0,magnitude=0,variance=0;
  for (size_t section=0,first=0;section<lookup.section_ends.size();first=lookup.section_ends[section++]) {
    const double scale=query.scales[section];
    for (size_t g=first;g<lookup.section_ends[section];++g) {
      const int8_t* table=query.tables.data()+g*lookup_table_entries;
      const auto [min_it,max_it]=std::minmax_element(table,table+lookup_table_entries);
      magnitude+=std::max(std::abs(int(*min_it)),std::abs(int(*max_it)))*scale;
      if (section<prefix) continue;
      high+=*max_it*scale; low+=*min_it*scale;
      if (confidence>0) for (size_t code=0;code<lookup_table_entries;++code)
        variance+=lookup.code_probability[g*lookup_table_entries+code]*double(table[code])*table[code]*scale*scale;
    }
  }
  if (confidence>0) {
    const double spread=confidence*std::sqrt(variance);
    high=std::min(high,spread); low=std::max(low,-spread);
  }
  // Integer section sums are exact; each section adds one product and one sum, the epilogue a fixed few.
  constexpr double epsilon=double(std::numeric_limits<float>::epsilon());
  constexpr double operations_per_group=2.0,finalization_operations=8.0;
  const double rounding_error=(operations_per_group*double(lookup.groups.size())+finalization_operations)*epsilon;
  if (rounding_error>=1.0) throw std::invalid_argument("TieredQ: groups exceed range-bound rounding budget");
  return {prefix,{range_round_up(high),range_round_down(low),range_round_up(magnitude),
      range_round_up(rounding_error/(1.0-rounding_error))}};
}
} // namespace dv::tieredq_detail
