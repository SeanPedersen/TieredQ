// Portable execution of persisted fast representations; preserves fast storage without exact fallback.
#pragma once
#include "tieredq_scan_lookup_x86.h"
#include "tieredq_scan_codes.h"
#include <bit>

namespace dv::tieredq_detail {
struct FastScanScalarQuery {
  TieredQQuery codes;
  FastScanLookupQuery lookup;
  std::vector<int8_t> weights;
  std::vector<float> centroid_scores,correction_weights;
  float scale=1;
};
inline FastScanScalarQuery fast_scan_query_scalar(const FastScanCache& cache,const TieredQ& codec,const float* vector) {
  FastScanScalarQuery result;
  if (cache.code_lookup()) { result.codes=codec.query_weights(vector); return result; }
  if (cache.lookup) { result.lookup=prepare_fast_lookup_query(*cache.lookup,vector,FastQueryBasis::from(codec)); return result; }
  const auto query=codec.query_weights(vector);
  float peak=0;
  for (size_t rank=0;rank<cache.dimensions;++rank)
    peak=std::max(peak,std::fabs(query.weights[rank]*cache.level_scale[cache.tier_of[rank]]));
  result.scale=peak>0 ? peak/float(fast_scan_quant_max) : 1;
  result.weights.assign(cache.groups*4,0);
  for (size_t rank=0;rank<cache.dimensions;++rank)
    result.weights[rank]=int8_t(std::clamp<long>(std::lround(query.weights[rank]*cache.level_scale[cache.tier_of[rank]]/result.scale),-long(fast_scan_quant_max),long(fast_scan_quant_max)));
  result.centroid_scores=query.centroid_scores;result.correction_weights=query.correction_weights;
  return result;
}
inline void scan_fast_scalar_tile(const FastScanCache& cache,const FastScanScalarQuery* queries,
    size_t rows,size_t first,size_t count,float* scores,size_t stride) {
  for (size_t row=0;row<rows;++row) for (size_t doc=first;doc<first+count;++doc) {
    const auto& q=queries[row];
    if (cache.code_lookup()) {
      scores[row*stride+doc-first]=fast_code_score(cache,q.codes,doc);
      continue;
    }
    float dot=0;
    if (cache.lookup) {
      const auto& lookup=*cache.lookup;
      for (size_t start=0;start<lookup.groups.size();start+=lookup_scale_groups) {
        int sum=0;
        for (size_t group=start;group<std::min(start+lookup_scale_groups,lookup.groups.size());++group) {
          const size_t position=fast_lookup_position(lookup,doc,group);
          const size_t code=(lookup.codes[position]>>(group%lookup_pair_groups*lookup_code_bits))&(lookup_table_entries-1);
          sum+=q.lookup.tables[group*lookup_table_entries+code];
        }
        dot+=float(sum)*q.lookup.scales[start/lookup_scale_groups];
      }
    } else {
      int32_t sum=0;
      for (size_t rank=0;rank<cache.dimensions;++rank) {
        const size_t position=fast_level_position(cache.groups,doc,rank);
        sum+=int32_t(cache.levels[position])*q.weights[rank];
      }
      dot=float(sum)*q.scale;
    }
    const size_t cid=cache.centroid[doc];
    const auto& centers=cache.lookup ? q.lookup.centroid_scores : q.centroid_scores;
    const auto& corrections=cache.lookup ? q.lookup.correction_weights : q.correction_weights;
    float score=centers[cid];
    score+=corrections[cid]*cache.correction_at(doc);
    score+=cache.factor_at(doc)*dot;
    scores[row*stride+doc-first]=score;
  }
}
} // namespace dv::tieredq_detail
