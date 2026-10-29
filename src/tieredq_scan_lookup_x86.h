// Compact full-scan cache and query-specific byte tables for x86 models with at most four bits per coordinate.
// Pairs of lookup codes share a byte; groups of tables share a scale to bound quantization error.
// Additive 3-bit tiers are scored as a 2-bit and a 1-bit group per code (tieredq_lookup_layout.h); groups are
// re-derived from the model's tiers and levels, so persisted lookup codes need no layout tag.
#pragma once
#include "tieredq_scan_int8.h"
#include "tieredq_query_projection_x86.h"
#include <array>
#include <limits>
#include <optional>

namespace dv::tieredq_detail {
constexpr size_t lookup_vector_docs=16, lookup_chunk_docs=64;
constexpr size_t lookup_block_docs=256, lookup_table_entries=size_t(1)<<lookup_code_bits, lookup_scale_groups=32;
constexpr int lookup_quant_max=127;
static_assert(lookup_scale_groups%2==0 && lookup_scale_groups*lookup_quant_max<=INT16_MAX);
// bit is the row offset of the first coordinate's whole code; bits is the width of the group's (part) codes.
struct FastLookupGroup { uint32_t bit, rank, count, bits; LookupPart part=LookupPart::whole; };
struct FastScanLookupCache {
  FastQueryProjection projection;
  std::vector<FastLookupGroup> groups;
  std::vector<uint8_t> codes;
  std::vector<std::vector<float>> levels;
  std::optional<AdditiveSplit> split; // present when the 3-bit codebook is additive
  size_t pairs() const { return lookup_group_bytes(groups.size()); }
  size_t allocated_bytes() const {
    size_t bytes=sizeof(*this)+projection.allocated_bytes()+groups.capacity()*sizeof(FastLookupGroup)+codes.capacity();
    bytes+=levels.capacity()*sizeof(std::vector<float>);
    for (const auto& book:levels) bytes+=book.capacity()*sizeof(float);
    if (split) for (const auto& book:split->part_levels) bytes+=book.capacity()*sizeof(float);
    return bytes;
  }
  // Levels of one group's codes: the codebook of its width, or one additive part.
  const std::vector<float>& group_levels(const FastLookupGroup& group) const {
    if (group.part==LookupPart::whole) return levels[group.bits];
    return split->part_levels[group.part==LookupPart::low ? 0 : 1];
  }
};
inline std::vector<FastLookupGroup> fast_lookup_groups(const std::vector<TieredQTier>& tiers,bool split) {
  std::vector<FastLookupGroup> groups;
  auto append=[&](const TieredQTier& tier,uint32_t bits,LookupPart part) {
    for (uint32_t j=0;j<tier.dimensions;) {
      const uint32_t n=std::min(lookup_coordinates_per_group(bits),tier.dimensions-j);
      groups.push_back({tier.bit_offset+j*tier.bits,tier.start+j,n,bits,part});
      j+=n;
    }
  };
  for (const auto& tier:tiers) {
    if (!split || tier.bits!=split_code_bits) { append(tier,tier.bits,LookupPart::whole); continue; }
    append(tier,split_low_bits,LookupPart::low);
    append(tier,split_high_bits,LookupPart::high);
  }
  return groups;
}
// Packed nibble code of one group for a document row.
inline uint32_t lookup_group_code(const FastScanLookupCache& cache,const FastLookupGroup& group,const uint8_t* row) {
  if (group.part==LookupPart::whole) return uint32_t(read_code(row,group.bit,group.count*group.bits));
  const uint32_t shift=group.part==LookupPart::low ? 0 : split_low_bits, mask=(1u<<group.bits)-1;
  uint32_t code=0;
  for (uint32_t j=0;j<group.count;++j) {
    const uint32_t whole=uint32_t(read_code(row,group.bit+j*split_code_bits,split_code_bits));
    code|=(cache.split->code[whole]>>shift&mask)<<(j*group.bits);
  }
  return code;
}
inline size_t fast_lookup_position(const FastScanLookupCache& cache,size_t doc,size_t group) {
  return (doc/lookup_block_docs*cache.pairs()+group/lookup_pair_groups)*lookup_block_docs+doc%lookup_block_docs;
}
struct FastScanLookupQuery {
  std::vector<int8_t> tables;
  std::vector<float> scales, centroid_scores, correction_weights;
};
inline bool fast_lookup_supported(const std::vector<TieredQTier>& tiers) {
  return std::all_of(tiers.begin(),tiers.end(),[](const auto& tier){return tier.bits<=lookup_code_bits;});
}
inline std::shared_ptr<FastScanCache> build_fast_lookup_cache(
    const std::vector<std::vector<float>>& levels, const std::vector<TieredQTier>& tiers,
    FastQueryProjection projection, size_t count, size_t stride, size_t code_bytes,
    size_t num_centroids, const uint8_t* rows) {
  auto cache=std::make_shared<FastScanCache>();
  auto lookup=std::make_shared<FastScanLookupCache>();
  cache->count=count; cache->dimensions=projection.weights;
  cache->blocks=(count+lookup_block_docs-1)/lookup_block_docs;
  lookup->projection=std::move(projection); lookup->levels=levels;
  lookup->split=additive_three_split(levels[split_code_bits]);
  lookup->groups=fast_lookup_groups(tiers,lookup->split.has_value());
  cache->groups=lookup->groups.size();
  lookup->codes.assign(cache->blocks*lookup->pairs()*lookup_block_docs,0);
  build_fast_scan_metadata(*cache,cache->blocks*lookup_block_docs,stride,code_bytes,num_centroids,rows);
  for (size_t doc=0;doc<count;++doc) {
    const uint8_t* row=rows+doc*stride;
    for (size_t g=0;g<lookup->groups.size();++g) {
      const auto& group=lookup->groups[g];
      const uint8_t code=uint8_t(lookup_group_code(*lookup,group,row));
      const size_t pos=fast_lookup_position(*lookup,doc,g);
      lookup->codes[pos]|=uint8_t(code<<(g%lookup_pair_groups*lookup_code_bits));
    }
  }
  cache->lookup=std::move(lookup);
  return cache;
}
inline FastScanLookupQuery prepare_fast_lookup_query(const FastScanLookupCache& cache,const float* vector,FastQueryBasis source) {
  const auto output=project_fast_query(cache.projection,vector,source);
  const size_t m=cache.projection.weights,k=cache.projection.centroids;
  FastScanLookupQuery query;
  query.centroid_scores.assign(output.begin()+m,output.begin()+m+k);
  query.correction_weights.assign(output.begin()+m+k,output.begin()+m+2*k);
  query.tables.assign(cache.pairs()*lookup_pair_groups*lookup_table_entries,0);
  std::array<float,lookup_scale_groups*lookup_table_entries> values;
  for (size_t start=0;start<cache.groups.size();start+=lookup_scale_groups) {
    const size_t count=std::min(lookup_scale_groups,cache.groups.size()-start);
    float peak=0;
    for (size_t g=0;g<count;++g) {
      const auto& group=cache.groups[start+g];
      const auto& levels=cache.group_levels(group);
      const uint32_t mask=(1u<<group.bits)-1;
      for (size_t code=0;code<lookup_table_entries;++code) {
        float value=0;
        for (uint32_t j=0;j<group.count;++j)
          value+=output[group.rank+j]*levels[(code>>(j*group.bits))&mask];
        if (!std::isfinite(value)) throw std::runtime_error("TieredQ: nonfinite fast scan table");
        values[g*lookup_table_entries+code]=value;
        peak=std::max(peak,std::fabs(value));
      }
    }
    const float scale=peak>0 ? std::max(peak/lookup_quant_max,std::numeric_limits<float>::min()) : 1;
    query.scales.push_back(scale);
    for (size_t i=0;i<count*lookup_table_entries;++i)
      query.tables[start*lookup_table_entries+i]=int8_t(std::clamp<long>(std::lround(values[i]/scale),-lookup_quant_max,lookup_quant_max));
  }
  return query;
}
} // namespace dv::tieredq_detail
