// Lossless packed-code fast scans retain FP32 codebook scores for high-precision models.
// Each 16-document block is decoded to FP32 levels once per query group; register tiles of queries then
// accumulate those levels in coordinate order with separate multiply/add, matching the scalar codec exactly.
#pragma once
#include "tieredq_scan_int8.h"
#include <type_traits>

namespace dv::tieredq_detail {
// Queries sharing one decoded block; bounds per-worker score scratch at group*document tile floats.
constexpr size_t fast_code_query_group=32;
inline float fast_code_score(const FastScanCache& cache,const TieredQQuery& query,size_t doc) {
  float dot=0;
  for (size_t rank=0;rank<cache.dimensions;++rank) {
    dot+=query.weights[rank]*cache.codebooks[cache.tier_of[rank]][fast_code(cache,doc,rank)];
  }
  const size_t cid=cache.centroid[doc];
  float score=query.centroid_scores[cid];
  score+=query.correction_weights[cid]*cache.correction_at(doc);
  score+=cache.factor_at(doc)*dot;
  return score;
}

// Query weights transposed rank-major within tiles of at most `tile` rows: [tile][rank][row].
class CodeScanWeights {
 public:
  template<class QueryAt>
  CodeScanWeights(size_t dimensions,size_t rows,size_t tile,QueryAt query)
      : dimensions_(dimensions),rows_(rows),tile_(tile),values_(dimensions*rows) {
    for (size_t first=0;first<rows;first+=tile) {
      const size_t width=std::min(tile,rows-first);
      float* out=values_.data()+first*dimensions;
      for (size_t rank=0;rank<dimensions;++rank) for (size_t row=0;row<width;++row)
        out[rank*width+row]=query(first+row).weights[rank];
    }
  }
  size_t tiles() const { return (rows_+tile_-1)/tile_; }
  size_t first_row(size_t tile) const { return tile*tile_; }
  size_t rows(size_t tile) const { return std::min(tile_,rows_-first_row(tile)); }
  const float* tile(size_t tile) const { return values_.data()+first_row(tile)*dimensions_; }
 private:
  size_t dimensions_,rows_,tile_;
  std::vector<float> values_;
};

// Decoded block layout: decoded[rank*fast_scan_block_docs+lane].
inline void decode_code_block(const FastScanCache& cache,size_t block,float* decoded) {
  const int8_t* base=cache.levels.data()+block*cache.level_block_bytes();
  uint32_t codes[fast_scan_block_docs];
  for (size_t rank=0;rank<cache.dimensions;++rank) {
    const float* book=cache.codebooks[cache.tier_of[rank]].data();
    read_block_codes(base,cache.code_offset[rank],fast_code_width(cache,rank),codes);
    for (size_t lane=0;lane<fast_scan_block_docs;++lane) decoded[rank*fast_scan_block_docs+lane]=book[codes[lane]];
  }
}

// Expands a runtime tile height into a compile-time register tile.
template<size_t Rows,class Kernel>
inline void dispatch_code_rows(size_t rows,Kernel kernel) {
  if constexpr (Rows>1) if (rows<Rows) return dispatch_code_rows<Rows-1>(rows,kernel);
  kernel(std::integral_constant<size_t,Rows>{});
}

// Shared block loop. `first` is block aligned; `accumulate` writes one query tile (by index) for one decoded block.
template<class Decode,class Accumulate,class QueryAt>
void scan_code_blocks(const FastScanCache& cache,size_t rows,size_t tile,size_t first,size_t count,
    float* scores,size_t stride,QueryAt query,Decode decode,Accumulate accumulate) {
  const CodeScanWeights weights(cache.dimensions,rows,tile,query);
  std::vector<float> decoded(cache.dimensions*fast_scan_block_docs);
  for (size_t doc=first;doc<first+count;doc+=fast_scan_block_docs) {
    decode(doc/fast_scan_block_docs,decoded.data());
    const size_t valid=std::min(fast_scan_block_docs,first+count-doc);
    for (size_t t=0;t<weights.tiles();++t) {
      const size_t row0=weights.first_row(t);
      accumulate(decoded.data(),t,weights.tile(t),weights.rows(t),doc,valid,scores+row0*stride+doc-first,stride,
          [&](size_t row)->const TieredQQuery& {return query(row0+row);});
    }
  }
}
} // namespace dv::tieredq_detail
