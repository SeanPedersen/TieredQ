// Shared fast-cache geometry, metadata and builders for numerical levels or original code IDs.
// Narrow x86 models attach a packed lookup cache; models with any tier wider than 4 bits preserve FP32 codebook
// precision by keeping each document's packed code row as 32-bit words interleaved across 16-document blocks.
#pragma once
#include "tieredq_layout.h"
#include "tieredq/tieredq.h"
#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace dv::tieredq_detail {
constexpr size_t fast_scan_block_docs = 16;
constexpr size_t fast_scan_coordinates_per_group = 4;
constexpr size_t fast_scan_vector_docs = 4;
constexpr size_t fast_scan_vector_bytes = fast_scan_vector_docs * fast_scan_coordinates_per_group;
constexpr size_t fast_scan_group_bytes = fast_scan_block_docs * fast_scan_coordinates_per_group;
constexpr size_t fast_scan_code_bits = 8;
constexpr size_t fast_code_word_bits = 32, fast_code_word_bytes = sizeof(uint32_t);
constexpr size_t fast_scan_quant_max = 127;
constexpr int x86_fast_level_max=63,arm_fast_level_max=127;
constexpr int fast_code_lookup_levels=0; // tagged code IDs, not a numerical quantization range
constexpr uint32_t lookup_code_bits=4; // widest tier of the narrow x86 lookup cache
// Any wider tier scans exact packed codes on every backend.
inline bool fast_code_lookup_supported(const std::vector<TieredQTier>& tiers) {
  return std::any_of(tiers.begin(),tiers.end(),[](const auto& tier){return tier.bits>lookup_code_bits;});
}

struct FastScanLookupCache;
struct FastScanCache {
  size_t count = 0, dimensions = 0, groups = 0, blocks = 0;
  std::shared_ptr<FastScanLookupCache> lookup; // x86 query-specific byte tables for narrow tiers
  int level_max = 127;
  std::vector<int8_t> levels;          // signed numerical levels or interleaved packed code words
  size_t code_bits = 0;                // code lookup: packed code bits per document
  std::vector<uint32_t> code_offset;   // code lookup: bit offset of each coded rank in its document's codes
  std::vector<float> factor, correction;
  std::vector<uint8_t> centroid;
  std::vector<float> level_scale;      // per tier
  std::vector<uint8_t> tier_of;        // per coded rank; 0xFF for group padding
  std::vector<std::vector<float>> codebooks; // per tier, only for FP32 code-ID scans
  bool code_lookup() const { return !lookup && level_max==fast_code_lookup_levels; }
  size_t code_words() const { return (code_bits+fast_code_word_bits-1)/fast_code_word_bits; }
  // Bytes of one 16-document block in `levels`.
  size_t level_block_bytes() const { return (code_lookup() ? code_words() : groups)*fast_scan_group_bytes; }
};

inline size_t fast_level_position(size_t groups,size_t doc,size_t rank) {
  const size_t lane=doc%fast_scan_block_docs;
  return doc/fast_scan_block_docs*groups*fast_scan_group_bytes+rank/fast_scan_coordinates_per_group*fast_scan_group_bytes+
      lane/fast_scan_vector_docs*fast_scan_vector_bytes+lane%fast_scan_vector_docs*fast_scan_coordinates_per_group+
      rank%fast_scan_coordinates_per_group;
}

// Byte offset of word `word` of a document's packed codes.
inline size_t fast_code_word_position(size_t words,size_t doc,size_t word) {
  return (doc/fast_scan_block_docs*words+word)*fast_scan_group_bytes+doc%fast_scan_block_docs*fast_code_word_bytes;
}
inline uint32_t load_code_word(const int8_t* at) {
  uint32_t word;
  std::memcpy(&word,at,sizeof(word));
  if constexpr (std::endian::native!=std::endian::little) {
    word=0;
    for (size_t byte=0;byte<fast_code_word_bytes;++byte)
      word|=uint32_t(uint8_t(at[byte]))<<(byte*CHAR_BIT);
  }
  return word;
}
inline size_t fast_code_width(const FastScanCache& cache,size_t rank) {
  return size_t(std::countr_zero(cache.codebooks[cache.tier_of[rank]].size()));
}
// One rank's codes for the 16 documents of a block starting at `block`.
inline void read_block_codes(const int8_t* block,size_t offset,size_t width,uint32_t* codes) {
  const size_t shift=offset%fast_code_word_bits;
  const uint32_t mask=(uint32_t(1)<<width)-1;
  const int8_t* low=block+offset/fast_code_word_bits*fast_scan_group_bytes;
  if (shift+width<=fast_code_word_bits) {
    for (size_t lane=0;lane<fast_scan_block_docs;++lane) codes[lane]=load_code_word(low+lane*fast_code_word_bytes)>>shift&mask;
    return;
  }
  const int8_t* high=low+fast_scan_group_bytes;
  for (size_t lane=0;lane<fast_scan_block_docs;++lane)
    codes[lane]=(load_code_word(low+lane*fast_code_word_bytes)>>shift|
        load_code_word(high+lane*fast_code_word_bytes)<<(fast_code_word_bits-shift))&mask;
}
inline uint32_t fast_code(const FastScanCache& cache,size_t doc,size_t rank) {
  const size_t offset=cache.code_offset[rank],shift=offset%fast_code_word_bits,width=fast_code_width(cache,rank);
  const int8_t* low=cache.levels.data()+fast_code_word_position(cache.code_words(),doc,offset/fast_code_word_bits);
  uint64_t value=load_code_word(low);
  if (shift+width>fast_code_word_bits) value|=uint64_t(load_code_word(low+fast_scan_group_bytes))<<fast_code_word_bits;
  return uint32_t(value>>shift)&((uint32_t(1)<<width)-1);
}
// Copies a document's packed code row; bits past code_bits stay zero.
inline void store_code_row(FastScanCache& cache,size_t doc,const uint8_t* row) {
  constexpr size_t byte_bits=8;
  for (size_t byte=0;byte<cache.code_words()*fast_code_word_bytes;++byte) {
    const size_t bit=byte*byte_bits;
    uint8_t value=0;
    if (bit<cache.code_bits) value=row[byte]&uint8_t((1u<<std::min(byte_bits,cache.code_bits-bit))-1);
    cache.levels[fast_code_word_position(cache.code_words(),doc,byte/fast_code_word_bytes)+byte%fast_code_word_bytes]=
        std::bit_cast<int8_t>(value);
  }
}

inline uint32_t read_code(const uint8_t* doc, uint32_t offset, uint32_t bits) {
  uint32_t value = uint32_t(doc[offset/8]) >> (offset%8);
  if (offset%8 + bits > 8) value |= uint32_t(doc[offset/8+1]) << (8 - offset%8);
  return value & ((1u<<bits)-1);
}

inline void build_fast_scan_metadata(FastScanCache& cache, size_t padded,
    size_t stride, size_t code_bytes, size_t num_centroids, const uint8_t* rows) {
  cache.factor.assign(padded,0); cache.correction.assign(padded,0); cache.centroid.assign(padded,0);
  const Layout layout{code_bytes};
  for (size_t doc=0;doc<cache.count;++doc) {
    const uint8_t* row=rows+doc*stride;
    if (row[code_bytes]>=num_centroids)
      throw std::runtime_error("TieredQ: invalid centroid id while building fast scan");
    cache.centroid[doc]=row[code_bytes];
    cache.factor[doc]=layout.factor(row); cache.correction[doc]=layout.correction(row);
  }
}

inline std::shared_ptr<FastScanCache> build_fast_scan_cache(
    const std::vector<std::vector<float>>& levels, const std::vector<TieredQTier>& tiers,
    size_t dimensions, size_t count, size_t stride, size_t code_bytes, size_t num_centroids,
    int level_max, const uint8_t* rows) {
  auto cache = std::make_shared<FastScanCache>();
  cache->count = count; cache->dimensions = dimensions; cache->level_max = level_max;
  cache->groups = (dimensions + fast_scan_coordinates_per_group - 1) / fast_scan_coordinates_per_group;
  cache->blocks = (count + fast_scan_block_docs - 1) / fast_scan_block_docs;
  cache->tier_of.assign(dimensions, 0xFF);
  cache->level_scale.resize(tiers.size());
  for (size_t ti=0; ti<tiers.size(); ++ti)
    for (uint32_t j=0; j<tiers[ti].dimensions; ++j) cache->tier_of[tiers[ti].start + j] = uint8_t(ti);
  const size_t padded=cache->blocks*fast_scan_block_docs;
  if (cache->code_lookup()) {
    cache->code_offset.resize(dimensions);
    for (const auto& tier:tiers) {
      cache->codebooks.push_back(levels.at(tier.bits));
      for (uint32_t j=0; j<tier.dimensions; ++j) cache->code_offset[tier.start + j] = tier.bit_offset + j*tier.bits;
      cache->code_bits += size_t(tier.dimensions)*tier.bits;
    }
    cache->levels.assign(cache->blocks * cache->level_block_bytes(), 0);
    build_fast_scan_metadata(*cache,padded,stride,code_bytes,num_centroids,rows);
    for (size_t doc=0; doc<count; ++doc) store_code_row(*cache, doc, rows + doc*stride);
    return cache;
  }
  std::vector<std::vector<int8_t>> quantized(tiers.size());
  for (size_t ti=0; ti<tiers.size(); ++ti) {
    const auto& book = levels.at(tiers[ti].bits);
    float peak = 0;
    for (float value : book) peak = std::max(peak, std::fabs(value));
    const float scale = peak > 0 && level_max ? peak / float(level_max) : 1.0f;
    cache->level_scale[ti] = scale;
    quantized[ti].resize(book.size());
    for (size_t c=0; c<book.size(); ++c)
      quantized[ti][c] = int8_t(std::clamp<long>(std::lround(book[c] / scale), -long(level_max), long(level_max)));
  }
  cache->levels.assign(cache->blocks * cache->level_block_bytes(), 0);
  build_fast_scan_metadata(*cache,padded,stride,code_bytes,num_centroids,rows);
  for (size_t doc=0; doc<count; ++doc) {
    const uint8_t* row = rows + doc*stride;
    for (size_t rank=0; rank<dimensions; ++rank) {
      const size_t ti = cache->tier_of[rank];
      const auto& tier = tiers[ti];
      const uint8_t code = uint8_t(read_code(row, tier.bit_offset + (rank-tier.start)*tier.bits, tier.bits));
      cache->levels[fast_level_position(cache->groups,doc,rank)] = quantized[ti][code];
    }
  }
  return cache;
}
} // namespace dv::tieredq_detail
