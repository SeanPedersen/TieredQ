// Fine tiers in fast indexes (#26): codes wider than the model's fast prefix width, whose top prefix bits are an
// additive fast code. The fast scan stores and scores only those prefixes, through a prefix view of the model (its
// tiers clipped to the prefix width, everything else shared); the remaining fine bits of every document sit in one
// packed side row. A search scans to a deeper candidate depth and rescores those candidates exactly from prefix
// codes plus fine bits.
#pragma once
#include "tieredq/tieredq.h"
#include "tieredq_fast_rows.h"
#include "tieredq_io.h"
#include "tieredq_layout.h"
#include <algorithm>
#include <climits>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace tq::tieredq_detail {

// Candidates rescored per query: at least this many, and this multiple of k (#26, 100k Gemini, OpenAI and GloVe:
// k = 10 recall stops changing from 24 candidates; k = 100 still gains up to 1.5 k on OpenAI and GloVe; #31: Gemini
// and OpenAI at 4 and 8 bits/dim keep k = 10 recall from 16).
constexpr size_t fine_min_candidates=24;
constexpr size_t fine_candidates_per_k_numerator=3,fine_candidates_per_k_denominator=2;
// Batch rescoring decodes the distinct candidates of this many queries at a time (bounding its row buffer), in pieces
// of consecutive documents.
constexpr size_t fine_rescore_query_chunk=1024,fine_decode_docs=256;
// Indexes other than the flat TieredQIndex do not serve fine-tier models.
inline TieredQ without_fine_tiers(TieredQ codec,const char* index) {
  if (codec.fast_prefix_bits()) bad(std::string(index)+" does not serve fine-tier models; train them with fine_tiers off");
  return codec;
}
inline size_t fine_candidate_depth(size_t k,size_t documents) {
  return std::min(documents,std::max(fine_min_candidates,
      (k*fine_candidates_per_k_numerator+fine_candidates_per_k_denominator-1)/fine_candidates_per_k_denominator));
}

// Readers of fine rows and rescored prefix rows may load 8 bytes from any field's first byte.
constexpr size_t fine_read_slack=sizeof(uint64_t);
inline uint32_t read_field(const uint8_t* data,uint32_t offset,uint32_t mask) {
  uint64_t window; std::memcpy(&window,data+offset/CHAR_BIT,sizeof(window));
  return uint32_t(window>>(offset%CHAR_BIT))&mask;
}

struct FastFine {
  // One model tier: its prefix field in prefix rows and, for fine tiers, its fine field in fine rows.
  struct Tier { uint32_t start, dimensions, bits, prefix_offset, fine_bits, fine_offset; };
  TieredQ prefix;              // the scan's model view
  std::vector<Tier> tiers;
  size_t fine_bytes = 0;       // per document
  // Prefix-row decoding tables of the prefix view's fast storage geometry, which the model alone determines.
  std::shared_ptr<const FastRowReader> reader;

  static std::vector<Tier> layout(const std::vector<TieredQTier>& model_tiers,uint32_t prefix_width,size_t& fine_bytes) {
    std::vector<Tier> tiers; uint32_t prefix_offset=0,fine_offset=0;
    for (const auto& t:model_tiers) {
      const uint32_t prefix_bits=std::min(t.bits,prefix_width),fine_bits=t.bits-prefix_bits;
      tiers.push_back({t.start,t.dimensions,t.bits,prefix_offset,fine_bits,fine_offset});
      prefix_offset+=t.dimensions*prefix_bits; fine_offset+=t.dimensions*fine_bits;
    }
    fine_bytes=(fine_offset+CHAR_BIT-1)/CHAR_BIT;
    return tiers;
  }
  // Loaded fine rows leave every bit past the last fine field zero.
  void validate_fine_rows(const uint8_t* rows,size_t count) const {
    const auto& last=tiers.back();
    const uint32_t used=(last.fine_offset+last.dimensions*last.fine_bits)%CHAR_BIT;
    if (!used) return;
    for (size_t i=0;i<count;++i) if (rows[(i+1)*fine_bytes-1]>>used) bad("nonzero fine-row padding");
  }
  // Prefix tiers merge equal widths: clipped fine tiers join the prefix-width tier, as fast layouts hold one tier per
  // width.
  // Ranks and prefix fields of neighbouring tiers are contiguous, so a merged tier reads exactly the same codes.
  static std::vector<TieredQTier> prefix_tiers(const std::vector<Tier>& tiers) {
    std::vector<TieredQTier> out;
    for (const auto& t:tiers) {
      const uint32_t bits=t.bits-t.fine_bits;
      if (!out.empty() && out.back().bits==bits) out.back().dimensions+=t.dimensions;
      else out.push_back({t.start,t.dimensions,bits,0,t.prefix_offset});
    }
    return out;
  }

  // Full-width payload rows -> prefix payload rows (same metadata) and fine rows.
  void split(const TieredQ& model,const uint8_t* rows,size_t count,uint8_t* prefix_rows,uint8_t* fine_rows) const {
    const size_t model_stride=model.bytes_per_vector(),model_code=model.code_bytes();
    const size_t prefix_stride=prefix.bytes_per_vector(),prefix_code=prefix.code_bytes();
    const auto& source=model.tiers();
    std::fill_n(prefix_rows,count*prefix_stride,uint8_t(0)); std::fill_n(fine_rows,count*fine_bytes,uint8_t(0));
    for (size_t i=0;i<count;++i) {
      const uint8_t* row=rows+i*model_stride; uint8_t* out=prefix_rows+i*prefix_stride; uint8_t* fine=fine_rows+i*fine_bytes;
      for (size_t ti=0;ti<tiers.size();++ti) {
        const auto& t=tiers[ti];
        for (uint32_t j=0;j<t.dimensions;++j) {
          const uint32_t code=read_wide_code(row,source[ti].bit_offset+j*t.bits,t.bits);
          write_code(out,t.prefix_offset+j*(t.bits-t.fine_bits),t.bits-t.fine_bits,code>>t.fine_bits);
          if (t.fine_bits) write_code(fine,t.fine_offset+j*t.fine_bits,t.fine_bits,code&((1u<<t.fine_bits)-1));
        }
      }
      std::copy(row+model_code,row+model_stride,out+prefix_code);
    }
  }

  // sum_r weights[r] * level(code_r) of one document at full width: the codec's per-row dot before its factor.
  // Both rows need fine_read_slack readable bytes past their end.
  float dot(const std::vector<std::vector<float>>& levels,const uint8_t* prefix_row,const uint8_t* fine_row,
            const float* weights) const {
    float sum=0;
    for (const auto& t:tiers) {
      const uint32_t prefix_bits=t.bits-t.fine_bits,prefix_mask=(1u<<prefix_bits)-1,fine_mask=(1u<<t.fine_bits)-1;
      const float* book=levels[t.bits].data(); const float* w=weights+t.start;
      if (prefix_bits==CHAR_BIT && t.prefix_offset%CHAR_BIT==0) {
        // Byte prefixes (8-bit prefix width).
        const uint8_t* prefix=prefix_row+t.prefix_offset/CHAR_BIT;
        if (!t.fine_bits) { for (uint32_t j=0;j<t.dimensions;++j) sum+=w[j]*book[prefix[j]]; continue; }
        for (uint32_t j=0;j<t.dimensions;++j)
          sum+=w[j]*book[uint32_t(prefix[j])<<t.fine_bits|read_field(fine_row,t.fine_offset+j*t.fine_bits,fine_mask)];
        continue;
      }
      if (!t.fine_bits) {
        for (uint32_t j=0;j<t.dimensions;++j) sum+=w[j]*book[read_field(prefix_row,t.prefix_offset+j*prefix_bits,prefix_mask)];
        continue;
      }
      for (uint32_t j=0;j<t.dimensions;++j)
        sum+=w[j]*book[read_field(prefix_row,t.prefix_offset+j*prefix_bits,prefix_mask)<<t.fine_bits|
                       read_field(fine_row,t.fine_offset+j*t.fine_bits,fine_mask)];
    }
    return sum;
  }
};

} // namespace tq::tieredq_detail
