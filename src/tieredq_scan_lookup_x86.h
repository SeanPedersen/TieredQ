// Compact full-scan cache and query-specific byte tables for x86 models with at most four bits per coordinate.
// Pairs of lookup codes share a byte; groups of tables share a scale to bound quantization error.
// Additive 3-bit codes are scored as a 2-bit and a 1-bit part that share groups with the 2- and 1-bit tiers
// (tieredq_lookup_layout.h); groups are re-derived from the model's tiers and levels, so persisted lookup codes
// need no layout tag. Models without the split keep one run of groups per tier.
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
// One coordinate of a group: the row offset of its whole code, its rank and which part of the code it scores
// (whole_code_part, or an index into its width's split, whose bits start at part_shift of the split code).
struct LookupSlot { uint32_t bit, rank, code_bits, part, part_shift; };
// count coordinates whose (part) codes are bits wide, packed from the low nibble bit up.
struct FastLookupGroup { uint32_t count, bits; std::array<LookupSlot,lookup_code_bits> slots; };
struct FastScanLookupCache {
  FastQueryProjection projection;
  std::vector<FastLookupGroup> groups;
  std::vector<uint8_t> codes;
  std::vector<std::vector<float>> levels;
  AdditiveSplits splits; // per code width, present when that codebook is additive
  std::vector<uint32_t> section_ends; // exclusive last group of each section of groups sharing a table scale
  size_t pairs() const { return lookup_group_bytes(groups.size()); }
  size_t allocated_bytes() const {
    size_t bytes=sizeof(*this)+projection.allocated_bytes()+groups.capacity()*sizeof(FastLookupGroup)+codes.capacity();
    bytes+=section_ends.capacity()*sizeof(uint32_t);
    bytes+=levels.capacity()*sizeof(std::vector<float>);
    for (const auto& book:levels) bytes+=book.capacity()*sizeof(float);
    for (const auto& split:splits) if (split) for (const auto& book:split->part_levels) bytes+=book.capacity()*sizeof(float);
    return bytes;
  }
  // Levels of one coordinate's (part) codes: its codebook, or one additive part.
  const std::vector<float>& slot_levels(const LookupSlot& slot) const {
    if (slot.part==whole_code_part) return levels[slot.code_bits];
    return splits[slot.code_bits]->part_levels[slot.part];
  }
};
// Without a split, each tier packs its own groups in tier order. With one, coordinates join per-width streams
// (split parts beside the tiers of their width), emitted in order of each width's first appearance.
inline std::vector<FastLookupGroup> fast_lookup_groups(const std::vector<TieredQTier>& tiers,const AdditiveSplits& splits) {
  const bool split=std::any_of(splits.begin(),splits.end(),[](const auto& s) { return s.has_value(); });
  std::vector<std::vector<LookupSlot>> streams;
  std::vector<uint32_t> stream_bits;
  std::array<int,lookup_code_bits+1> stream_of; stream_of.fill(-1);
  auto stream=[&](uint32_t bits) -> std::vector<LookupSlot>& {
    if (!split || stream_of[bits]<0) { stream_of[bits]=int(streams.size()); streams.emplace_back(); stream_bits.push_back(bits); }
    return streams[size_t(stream_of[bits])];
  };
  for (const auto& tier:tiers) {
    if (!splits[tier.bits]) {
      auto& whole=stream(tier.bits);
      for (uint32_t j=0;j<tier.dimensions;++j) whole.push_back({tier.bit_offset+j*tier.bits,tier.start+j,tier.bits,whole_code_part,0});
      continue;
    }
    uint32_t shift=0;
    for (uint32_t part=0;part<splits[tier.bits]->part_bits.size();++part) {
      const uint32_t bits=splits[tier.bits]->part_bits[part];
      auto& parts=stream(bits);
      for (uint32_t j=0;j<tier.dimensions;++j) parts.push_back({tier.bit_offset+j*tier.bits,tier.start+j,tier.bits,part,shift});
      shift+=bits;
    }
  }
  std::vector<FastLookupGroup> groups;
  for (size_t s=0;s<streams.size();++s) {
    const auto& slots=streams[s];
    if (slots.empty()) continue;
    const uint32_t bits=stream_bits[s];
    const uint32_t per_group=lookup_coordinates_per_group(bits);
    for (size_t j=0;j<slots.size();j+=per_group) {
      FastLookupGroup group{uint32_t(std::min<size_t>(per_group,slots.size()-j)),bits,{}};
      std::copy_n(slots.begin()+j,group.count,group.slots.begin());
      groups.push_back(group);
    }
  }
  return groups;
}
// Packed nibble code of one group for a document row.
inline uint32_t lookup_group_code(const FastScanLookupCache& cache,const FastLookupGroup& group,const uint8_t* row) {
  const uint32_t mask=(1u<<group.bits)-1;
  uint32_t code=0;
  for (uint32_t j=0;j<group.count;++j) {
    const auto& slot=group.slots[j];
    uint32_t part=uint32_t(read_code(row,slot.bit,slot.code_bits));
    if (slot.part!=whole_code_part) part=uint32_t(cache.splits[slot.code_bits]->code[part])>>slot.part_shift;
    code|=(part&mask)<<(j*group.bits);
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
inline bool fast_lookup_supported(const std::vector<TieredQTier>& tiers,const std::vector<std::vector<float>>& levels) {
  const auto splits=additive_splits(levels);
  return std::all_of(tiers.begin(),tiers.end(),[&](const auto& tier){return tier.bits<=lookup_code_bits || splits[tier.bits];});
}
// Largest expected table magnitude of a group: query weights scale with sigma^2 for queries distributed like the data.
inline float lookup_group_range(const FastScanLookupCache& cache,const FastLookupGroup& group,const std::vector<float>& sigma) {
  float range=0;
  for (uint32_t j=0;j<group.count;++j) {
    const auto& slot=group.slots[j]; const auto& book=cache.slot_levels(slot);
    range=std::max(range,sigma[slot.rank]*sigma[slot.rank]*(book.back()-book.front()));
  }
  return range;
}
inline void uniform_lookup_sections(FastScanLookupCache& cache,size_t section_groups) {
  cache.section_ends.clear();
  for (size_t end=section_groups;end<cache.groups.size();end+=section_groups) cache.section_ends.push_back(uint32_t(end));
  if (!cache.groups.empty()) cache.section_ends.push_back(uint32_t(cache.groups.size()));
}
// Byte tables share a scale per section, so models with wide tiers order groups by range: a low part next to its high
// part would otherwise round away. Narrow models keep their stream order, which persisted codes depend on.
inline void order_lookup_groups_by_range(FastScanLookupCache& cache,const std::vector<float>& sigma) {
  std::vector<std::pair<float,FastLookupGroup>> ranked;
  for (const auto& group:cache.groups) ranked.emplace_back(lookup_group_range(cache,group,sigma),group);
  std::stable_sort(ranked.begin(),ranked.end(),[](const auto& a,const auto& b) { return a.first>b.first; });
  for (size_t g=0;g<ranked.size();++g) cache.groups[g]=ranked[g].second;
}
// 7-8 bit tiers appear from about 4 bits/dim, where table rounding limits recall: 32-group sections lost 0.6 pt R@10
// and 4-group ones 0.2 pt on OpenAI-100K, at the cost of more float conversions (#13). Clipping scales to RMS loses more.
constexpr uint32_t fine_section_min_bits=7;
constexpr size_t lookup_fine_scale_groups=4;
static_assert(lookup_fine_scale_groups%lookup_pair_groups==0 && lookup_fine_scale_groups<=lookup_scale_groups);
inline std::shared_ptr<FastScanCache> build_fast_lookup_cache(
    const std::vector<std::vector<float>>& levels, const std::vector<TieredQTier>& tiers,
    FastQueryProjection projection, size_t count, size_t stride, size_t code_bytes,
    size_t num_centroids, const uint8_t* rows, const std::vector<float>& sigma={}) {
  auto cache=std::make_shared<FastScanCache>();
  auto lookup=std::make_shared<FastScanLookupCache>();
  cache->count=count; cache->dimensions=projection.weights;
  cache->blocks=(count+lookup_block_docs-1)/lookup_block_docs;
  lookup->projection=std::move(projection); lookup->levels=levels;
  lookup->splits=additive_splits(levels);
  lookup->groups=fast_lookup_groups(tiers,lookup->splits);
  const bool wide=std::any_of(tiers.begin(),tiers.end(),[](const auto& tier) { return tier.bits>lookup_code_bits; });
  if (wide && !sigma.empty()) order_lookup_groups_by_range(*lookup,sigma);
  const bool fine=std::any_of(tiers.begin(),tiers.end(),[](const auto& tier) { return tier.bits>=fine_section_min_bits; });
  uniform_lookup_sections(*lookup,fine ? lookup_fine_scale_groups : lookup_scale_groups);
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
// std::lround's half-away-from-zero rounding without the libm call (exact: x - trunc(x) is representable).
inline float round_half_away(float x) {
  const float whole=std::trunc(x);
  return std::fabs(x-whole)>=0.5f ? whole+std::copysign(1.0f,x) : whole;
}
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
// One register per group holds its 16 entries; each entry keeps the scalar sequence (0, then + weight * level per
// slot in order), the same scale and the same half-away rounding, so tables match the portable build bit for bit.
__attribute__((target("avx512f,avx512bw")))
inline void fast_lookup_tables_avx512(const FastScanLookupCache& cache,const float* weights,FastScanLookupQuery& query) {
  static_assert(lookup_table_entries==16);
  const auto codes=_mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
  const auto magnitude=_mm512_set1_epi32(INT32_MAX);
  alignas(64) float values[lookup_scale_groups*lookup_table_entries];
  for (size_t section=0,start=0;section<cache.section_ends.size();start=cache.section_ends[section++]) {
    const size_t count=cache.section_ends[section]-start;
    auto peak=_mm512_setzero_ps();
    for (size_t g=0;g<count;++g) {
      const auto& group=cache.groups[start+g];
      const auto mask=_mm512_set1_epi32(int((1u<<group.bits)-1));
      auto value=_mm512_setzero_ps();
      for (uint32_t j=0;j<group.count;++j) {
        const auto& levels=cache.slot_levels(group.slots[j]);
        const auto book=_mm512_maskz_loadu_ps(__mmask16((1u<<levels.size())-1),levels.data());
        const auto index=_mm512_and_si512(_mm512_srlv_epi32(codes,_mm512_set1_epi32(int(j*group.bits))),mask);
        value=_mm512_add_ps(value,_mm512_mul_ps(_mm512_set1_ps(weights[group.slots[j].rank]),_mm512_permutexvar_ps(index,book)));
      }
      // x - x is zero exactly for finite x.
      if (_mm512_cmp_ps_mask(_mm512_sub_ps(value,value),_mm512_setzero_ps(),_CMP_EQ_OQ)!=0xFFFF)
        throw std::runtime_error("TieredQ: nonfinite fast scan table");
      _mm512_store_ps(values+g*lookup_table_entries,value);
      peak=_mm512_max_ps(peak,_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(value),magnitude)));
    }
    const float largest=_mm512_reduce_max_ps(peak);
    const float scale=largest>0 ? std::max(largest/lookup_quant_max,std::numeric_limits<float>::min()) : 1;
    query.scales.push_back(scale);
    const auto divisor=_mm512_set1_ps(scale),half=_mm512_set1_ps(0.5f),one=_mm512_set1_ps(1.0f);
    const auto low=_mm512_set1_ps(-float(lookup_quant_max)),high=_mm512_set1_ps(float(lookup_quant_max));
    for (size_t g=0;g<count;++g) {
      const auto x=_mm512_div_ps(_mm512_load_ps(values+g*lookup_table_entries),divisor);
      const auto whole=_mm512_roundscale_ps(x,_MM_FROUND_TO_ZERO|_MM_FROUND_NO_EXC);
      const auto fraction=_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(_mm512_sub_ps(x,whole)),magnitude));
      const auto away=_mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(one),
          _mm512_andnot_si512(magnitude,_mm512_castps_si512(x))));
      auto rounded=_mm512_mask_add_ps(whole,_mm512_cmp_ps_mask(fraction,half,_CMP_GE_OQ),whole,away);
      rounded=_mm512_min_ps(_mm512_max_ps(rounded,low),high);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(query.tables.data()+(start+g)*lookup_table_entries),
          _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(rounded)));
    }
  }
}
#endif
inline void fast_lookup_tables_scalar(const FastScanLookupCache& cache,const float* output,FastScanLookupQuery& query) {
  std::array<float,lookup_scale_groups*lookup_table_entries> values;
  for (size_t section=0,start=0;section<cache.section_ends.size();start=cache.section_ends[section++]) {
    const size_t count=cache.section_ends[section]-start;
    float peak=0;
    for (size_t g=0;g<count;++g) {
      const auto& group=cache.groups[start+g];
      const uint32_t mask=(1u<<group.bits)-1;
      for (size_t code=0;code<lookup_table_entries;++code) {
        float value=0;
        for (uint32_t j=0;j<group.count;++j)
          value+=output[group.slots[j].rank]*cache.slot_levels(group.slots[j])[(code>>(j*group.bits))&mask];
        if (!std::isfinite(value)) throw std::runtime_error("TieredQ: nonfinite fast scan table");
        values[g*lookup_table_entries+code]=value;
        peak=std::max(peak,std::fabs(value));
      }
    }
    const float scale=peak>0 ? std::max(peak/lookup_quant_max,std::numeric_limits<float>::min()) : 1;
    query.scales.push_back(scale);
    for (size_t i=0;i<count*lookup_table_entries;++i)
      query.tables[start*lookup_table_entries+i]=int8_t(std::clamp(round_half_away(values[i]/scale),-float(lookup_quant_max),float(lookup_quant_max)));
  }
}
// Walks a query group over [first, first + count) one code block at a time, so the group shares each block's cache
// traffic instead of every query streaming the whole range; scan_range(row, begin, documents, out) scores one query.
template<class ScanRange>
inline void scan_lookup_query_group(size_t rows,size_t first,size_t count,float* scores,size_t stride,ScanRange scan_range) {
  for (size_t begin=first,end=first+count;begin<end;) {
    const size_t documents=std::min((begin/lookup_block_docs+1)*lookup_block_docs,end)-begin;
    for (size_t row=0;row<rows;++row) scan_range(row,begin,documents,scores+row*stride+(begin-first));
    begin+=documents;
  }
}
// Byte tables from one query's projection (project_fast_queries layout).
inline FastScanLookupQuery fast_lookup_query(const FastScanLookupCache& cache,const std::vector<float>& output) {
  const size_t m=cache.projection.weights,k=cache.projection.centroids;
  FastScanLookupQuery query;
  query.centroid_scores.assign(output.begin()+m,output.begin()+m+k);
  query.correction_weights.assign(output.begin()+m+k,output.begin()+m+2*k);
  query.tables.assign(cache.pairs()*lookup_pair_groups*lookup_table_entries,0);
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
  if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) {
    fast_lookup_tables_avx512(cache,output.data(),query);
    return query;
  }
#endif
  fast_lookup_tables_scalar(cache,output.data(),query);
  return query;
}
inline FastScanLookupQuery prepare_fast_lookup_query(const FastScanLookupCache& cache,const float* vector,FastQueryBasis source) {
  return fast_lookup_query(cache,project_fast_query(cache.projection,vector,source));
}
} // namespace dv::tieredq_detail
