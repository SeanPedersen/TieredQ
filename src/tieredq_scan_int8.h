// Shared metadata and int8 level-cache builder for opt-in approximate full scans.
// ARM and wider-code x86 models use level dot products; narrow-code x86 models attach
// a compact byte-lookup cache instead of allocating expanded levels.
#pragma once
#include "tieredq_layout.h"
#include "tieredq/tieredq.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace dv::tieredq_detail {
constexpr size_t fast_scan_block_docs = 16;
constexpr size_t fast_scan_quant_max = 127;
constexpr int x86_fast_level_max=63,arm_fast_level_max=127;

struct FastScanLookupCache;
struct FastScanCache {
  size_t count = 0, dimensions = 0, groups = 0, blocks = 0;
  std::shared_ptr<FastScanLookupCache> lookup; // x86 query-specific byte tables for narrow tiers
  int level_max = 127;
  std::vector<int8_t> levels;          // blocks * groups * 64 bytes
  std::vector<float> factor, correction;
  std::vector<uint8_t> centroid;
  std::vector<float> level_scale;      // per tier
  std::vector<uint8_t> tier_of;        // per coded rank; 0xFF for group padding
};

inline size_t fast_level_position(size_t groups,size_t doc,size_t rank) {
  constexpr size_t coordinates_per_group=4,vector_bytes=16,group_bytes=64;
  const size_t lane=doc%fast_scan_block_docs;
  return doc/fast_scan_block_docs*groups*group_bytes+rank/coordinates_per_group*group_bytes+
      lane/coordinates_per_group*vector_bytes+lane%coordinates_per_group*coordinates_per_group+rank%coordinates_per_group;
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
  cache->groups = (dimensions + 3) / 4;
  cache->blocks = (count + fast_scan_block_docs - 1) / fast_scan_block_docs;
  cache->tier_of.assign(dimensions, 0xFF);
  cache->level_scale.resize(tiers.size());
  std::vector<std::vector<int8_t>> quantized(tiers.size());
  for (size_t ti=0; ti<tiers.size(); ++ti) {
    const auto& tier = tiers[ti];
    const auto& book = levels.at(tier.bits);
    float peak = 0;
    for (float value : book) peak = std::max(peak, std::fabs(value));
    const float scale = peak > 0 ? peak / float(level_max) : 1.0f;
    cache->level_scale[ti] = scale;
    quantized[ti].resize(book.size());
    for (size_t c=0; c<book.size(); ++c)
      quantized[ti][c] = int8_t(std::clamp<long>(std::lround(book[c] / scale), -long(level_max), long(level_max)));
    for (uint32_t j=0; j<tier.dimensions; ++j)
      cache->tier_of[tier.start + j] = uint8_t(ti);
  }
  cache->levels.assign(cache->blocks * cache->groups * 64, 0);
  const size_t padded=cache->blocks*fast_scan_block_docs;
  build_fast_scan_metadata(*cache,padded,stride,code_bytes,num_centroids,rows);
  for (size_t doc=0; doc<count; ++doc) {
    const uint8_t* row = rows + doc*stride;
    for (size_t g=0; g<cache->groups; ++g) for (size_t c=0; c<4; ++c) {
      const size_t rank = g*4 + c;
      if (rank >= dimensions) continue;
      const size_t ti = cache->tier_of[rank];
      const auto& tier = tiers[ti];
      const size_t j = rank - tier.start;
      const uint8_t code = uint8_t(read_code(row, tier.bit_offset + j*tier.bits, tier.bits));
      const size_t pos = fast_level_position(cache->groups,doc,rank);
      cache->levels[pos] = quantized[ti][code];
    }
  }
  return cache;
}
} // namespace dv::tieredq_detail
