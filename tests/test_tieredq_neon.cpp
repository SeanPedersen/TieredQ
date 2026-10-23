// Exhaustive binary16 conversion and NEON final-score order and centroid bounds.
#if defined(__aarch64__) || defined(_M_ARM64)
#include "tieredq_scan_neon.h"
#include <bit>
#include <limits>

namespace {
using namespace dv;
using namespace dv::tieredq_detail;

void require(bool ok,const char* message) {
  if (!ok) throw std::runtime_error(message);
}

void half_patterns() {
  constexpr uint32_t patterns=uint32_t{1}<<std::numeric_limits<uint16_t>::digits;
  std::array<uint8_t,2*neon_block_docs> plane;
  std::array<float,neon_block_docs> decoded;
  for (uint32_t first=0;first<patterns;first+=neon_block_docs) {
    for (size_t i=0;i<neon_block_docs;++i) {
      plane[i]=uint8_t(first+i);
      plane[neon_block_docs+i]=uint8_t((first+i)>>neon_byte_bits);
    }
    const auto values=neon_half_plane(plane.data(),0);
    for (size_t lane=0;lane<values.size();++lane)
      vst1q_f32(decoded.data()+lane*neon_float_lanes,values[lane]);
    for (size_t i=0;i<neon_block_docs;++i) {
      const float expected=half_decode(uint16_t(first+i));
      require(std::isnan(expected) ? std::isnan(decoded[i]) :
          std::bit_cast<uint32_t>(expected)==std::bit_cast<uint32_t>(decoded[i]),"NEON binary16 conversion mismatch");
    }
  }
}

void final_scores() {
  constexpr Layout layout{0};
  constexpr size_t centroids=256;
  constexpr std::array<uint16_t,neon_block_docs> halves={
    0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,
    0x3c00,0xbc00,0x7bff,0xfbff,0x3555,0xb555,0x4000,0xc000};
  std::vector<uint8_t> block(layout.metadata_bytes()*neon_block_docs);
  TieredQQuery q;
  for (size_t cid=0;cid<centroids;++cid) {
    q.centroid_scores.push_back(float(cid)/centroids);
    q.correction_weights.push_back(float(int(cid)-int(centroids/2))/centroids);
  }
  std::array<float,neon_block_docs> totals,scores;
  for (size_t i=0;i<neon_block_docs;++i) {
    block[i]=i%2 ? uint8_t(centroids-1-i) : uint8_t(i);
    totals[i]=1.0f/float(i+1);
    const auto store=[&](size_t offset,uint16_t half) {
      block[offset*neon_block_docs+i]=uint8_t(half);
      block[(offset+1)*neon_block_docs+i]=uint8_t(half>>neon_byte_bits);
    };
    store(layout.factor_offset(),halves[i]);
    store(layout.correction_offset(),halves[neon_block_docs-1-i]);
  }
  NeonBlockScores sums;
  for (size_t lane=0;lane<sums.size();++lane) sums[lane]=vld1q_f32(totals.data()+lane*neon_float_lanes);
  neon_finish(block.data(),q,layout,centroids,sums,scores.data());
  for (size_t i=0;i<neon_block_docs;++i) {
    const auto* doc=block.data()+i;
    float expected=q.centroid_scores[doc[0]];
    expected+=q.correction_weights[doc[0]]*layout.correction(doc,neon_block_docs);
    expected+=layout.factor(doc,neon_block_docs)*totals[i];
    require(std::bit_cast<uint32_t>(scores[i])==std::bit_cast<uint32_t>(expected),"NEON final scoring order mismatch");
  }
  constexpr uint32_t valid_centroids=4;
  q.centroid_scores.resize(valid_centroids);
  q.correction_weights.resize(valid_centroids);
  std::fill_n(block.data(),neon_block_docs,0);
  for (size_t i=0;i<neon_block_docs;++i) {
    block[i]=valid_centroids;
    bool rejected=false;
    try { neon_finish(block.data(),q,layout,valid_centroids,sums,scores.data()); }
    catch (const std::runtime_error&) { rejected=true; }
    require(rejected,"NEON centroid bounds mismatch");
    block[i]=0;
  }
}
}

void tieredq_neon_tests() { half_patterns(); final_scores(); }
#else
void tieredq_neon_tests() {}
#endif
