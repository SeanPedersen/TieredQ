// AArch64 grouped FP32 scoring directly over sixteen-row byte planes.
#pragma once
#include "tieredq_layout.h"
#include "tieredq_scan_groups.h"
#include <arm_neon.h>
#include <array>

namespace dv::tieredq_detail {
constexpr size_t neon_block_docs = 16;

inline float32x4_t neon_lookup(const float* table, uint32x4_t codes, uint32_t bits) {
  const uint8x16_t byte_offsets = {0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3};
  const auto indices=vaddq_u8(vreinterpretq_u8_u32(vmulq_n_u32(codes,0x04040404)),byte_offsets);
  uint8x16x4_t low;
  for (size_t i=0;i<4;++i) low.val[i]=vld1q_u8(reinterpret_cast<const uint8_t*>(table)+i*16);
  auto values=vqtbl4q_u8(low,indices);
  if (bits>short_table_bits) {
    uint8x16x4_t high;
    for (size_t i=0;i<4;++i) high.val[i]=vld1q_u8(reinterpret_cast<const uint8_t*>(table)+64+i*16);
    values=vorrq_u8(values,vqtbl4q_u8(high,vsubq_u8(indices,vdupq_n_u8(64))));
  }
  return vreinterpretq_f32_u8(values);
}

inline size_t scan_neon(const uint8_t* payload,size_t count,size_t stride,
    const TieredQQuery& q,const Layout& layout,uint32_t centroids,float* scores) {
  size_t begin=0;
  for (;count-begin>=neon_block_docs;begin+=neon_block_docs) {
    const auto* block=payload+begin*stride;
    float32x4_t sums[4]={vdupq_n_f32(0),vdupq_n_f32(0),vdupq_n_f32(0),vdupq_n_f32(0)};
    for (const auto& group:q.scan_groups) {
      const size_t byte=group.bit_offset/8;
      const int shift=group.bit_offset%8;
      const auto low=vld1q_u8(block+byte*neon_block_docs);
      uint16x8_t codes[2]={vmovl_u8(vget_low_u8(low)),vmovl_u8(vget_high_u8(low))};
      if (shift+group.bits>8) {
        const auto high=vld1q_u8(block+(byte+1)*neon_block_docs);
        codes[0]=vorrq_u16(codes[0],vshlq_n_u16(vmovl_u8(vget_low_u8(high)),8));
        codes[1]=vorrq_u16(codes[1],vshlq_n_u16(vmovl_u8(vget_high_u8(high)),8));
      }
      for (auto& code:codes) code=vandq_u16(vshlq_u16(code,vdupq_n_s16(-shift)),vdupq_n_u16((1u<<group.bits)-1));
      const uint32x4_t ids[4]={vmovl_u16(vget_low_u16(codes[0])),vmovl_u16(vget_high_u16(codes[0])),
                              vmovl_u16(vget_low_u16(codes[1])),vmovl_u16(vget_high_u16(codes[1]))};
      for (size_t lane=0;lane<4;++lane) {
        float32x4_t values;
        if (group.table!=TieredQScanGroup::wide) values=neon_lookup(q.scan_tables.data()+group.table,ids[lane],group.bits);
        else {
          std::array<uint32_t,4> indices; vst1q_u32(indices.data(),ids[lane]);
          const float* table=q.lut.data()+q.offsets[group.first];
          const float gathered[4]={table[indices[0]],table[indices[1]],table[indices[2]],table[indices[3]]};
          values=vld1q_f32(gathered);
        }
        sums[lane]=vaddq_f32(sums[lane],values);
      }
    }
    float totals[neon_block_docs];
    for (size_t lane=0;lane<4;++lane) vst1q_f32(totals+lane*4,sums[lane]);
    for (size_t i=0;i<neon_block_docs;++i) {
      const auto* doc=block+i;
      const auto cid=doc[layout.centroid_offset()*neon_block_docs];
      if (cid>=centroids) throw std::runtime_error("TieredQ: invalid centroid id");
      float score=q.centroid_scores[cid];
      score+=q.correction_weights[cid]*layout.correction(doc,neon_block_docs);
      score+=layout.factor(doc,neon_block_docs)*totals[i];
      scores[begin+i]=score;
    }
  }
  return begin;
}
} // namespace dv::tieredq_detail
