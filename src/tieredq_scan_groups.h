// Grouped FP32 scan plan: whole coordinates of a tier share one register-table lookup.
// Narrow coordinates are packed into groups of at most five code bits whose float table holds the sum of their
// LUT entries; vpermps (16 entries) and vpermt2ps (32 entries) read only the low index bits, so tables repeat
// every 2^bits entries and the kernel never masks. Wider coordinates score their shared codebook level times
// the coordinate weight, the same product as their LUT entry. In-memory only: payload bytes are unchanged.
#pragma once
#include "tieredq/tieredq.h"
#include "tieredq/detail/tieredq_scan_plan.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace dv::tieredq_detail {
constexpr uint32_t group_table_bits = 5;       // widest group a vpermt2ps register pair indexes
constexpr uint32_t short_table_bits = 4;       // groups up to this width use one 16-entry register
constexpr uint32_t short_table_entries = 16;
constexpr uint32_t long_table_entries = 32;

inline uint32_t group_table_entries(uint32_t bits) {
  return bits<=short_table_bits ? short_table_entries : long_table_entries;
}

// Appends groups and tables for an FP32 query whose lut, offsets and weights are already built.
inline void build_scan_groups(const std::vector<TieredQTier>& tiers, TieredQQuery& q) {
  q.scan_groups.clear(); q.scan_tables.clear();
  for (const auto& t:tiers) for (uint32_t j=0;j<t.dimensions;) {
    const uint32_t rank=t.start+j, bit=t.bit_offset+j*t.bits;
    if (t.bits>group_table_bits) {
      q.scan_groups.push_back({bit,t.bits,rank,1,TieredQScanGroup::wide});
      ++j; continue;
    }
    const uint32_t count=std::min(group_table_bits/t.bits,t.dimensions-j), bits=count*t.bits;
    const uint32_t entries=group_table_entries(bits), code_mask=(1u<<bits)-1, coordinate_mask=(1u<<t.bits)-1;
    q.scan_groups.push_back({bit,bits,rank,count,uint32_t(q.scan_tables.size())});
    for (uint32_t entry=0;entry<entries;++entry) {
      const uint32_t code=entry&code_mask;
      float sum=0;
      for (uint32_t c=0;c<count;++c) sum+=q.lut[q.offsets[rank+c]+((code>>(c*t.bits))&coordinate_mask)];
      q.scan_tables.push_back(sum);
    }
    j+=count;
  }
}

// One plan entry per group built by build_scan_groups, in the same order. The shift, window and straddle flag
// depend only on the tier layout, so the hot loop reads them instead of recomputing them per document block.
inline void build_scan_plan(const std::vector<TieredQTier>& tiers, std::vector<ScanGroupPlan>& plan) {
  plan.clear();
  uint32_t table=0;
  for (const auto& t:tiers) for (uint32_t j=0;j<t.dimensions;) {
    const uint32_t rank=t.start+j, bit=t.bit_offset+j*t.bits, shift=bit%32;
    if (t.bits>group_table_bits) {
      plan.push_back({bit/32,shift,32-shift,rank,t.bits,UINT32_MAX,shift+t.bits>32});
      ++j; continue;
    }
    const uint32_t count=std::min(group_table_bits/t.bits,t.dimensions-j), bits=count*t.bits, entries=group_table_entries(bits);
    plan.push_back({bit/32,shift,32-shift,rank,bits,table,shift+bits>32});
    table+=entries; j+=count;
  }
}

inline bool same_group(const TieredQScanGroup& a, const TieredQScanGroup& b) {
  return a.bit_offset==b.bit_offset && a.bits==b.bits && a.first==b.first && a.count==b.count && a.table==b.table;
}

// Query validation: the plan and tables must be exactly what this model derives from the query's LUT.
inline bool scan_groups_match(const std::vector<TieredQTier>& tiers, const TieredQQuery& q) {
  TieredQQuery expected;
  expected.lut=q.lut; expected.offsets=q.offsets;
  build_scan_groups(tiers,expected);
  return expected.scan_tables==q.scan_tables &&
         std::equal(expected.scan_groups.begin(),expected.scan_groups.end(),q.scan_groups.begin(),q.scan_groups.end(),same_group);
}
} // namespace dv::tieredq_detail
