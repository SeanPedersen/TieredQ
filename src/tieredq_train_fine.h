// Fine tiers of fast-target training (#26): codes of up to widest_fine_code_bits whose top prefix_bits are an
// additive fast code at the automatic cap. The fast scan reads only those prefixes and rescores its candidates
// exactly, so the scan keeps the prefix width's speed. Training keeps the fine allocation whenever it codes a rank wider
// than the prefix: on 100k Gemini-768, OpenAI-1536 and GloVe-200 at 3-8 bits/dim it never cost recall, while a 1 dB
// predicted-gain gate declined cases where it helped (GloVe at 6-8, every dataset at 3-4 bits/dim; #31).
#pragma once
#include "tieredq_lookup_layout.h"
#include "tieredq_train_allocation.h"
#include "tieredq_train_codebook.h"
#include <optional>
#include <vector>

namespace dv::tieredq_detail {

// books[b] for prefix_bits < b <= widest_fine_code_bits refines the additive prefix book by b - prefix_bits bits.
struct FineBooks {
  unsigned prefix_bits;
  std::vector<std::vector<float>> books;
};
inline FineBooks fine_codebooks(unsigned prefix_bits) {
  FineBooks out{prefix_bits,std::vector<std::vector<float>>(widest_fine_code_bits+1)};
  const auto& prefix=additive_wide_codebook(prefix_bits);
  for (unsigned b=prefix_bits+1;b<=widest_fine_code_bits;++b) out.books[b]=refined_prefix_codebook(prefix,b-prefix_bits);
  return out;
}

// The fine-tier allocation of the weighted additive model `narrow`: ranks below wide_dims may reach fine_cap bits
// through the fine books. None when it allocates no code wider than the prefix.
inline std::optional<Allocation> fine_allocation(const AllocationModel& narrow,const FineBooks& books,uint32_t fine_cap,
    uint32_t wide_dims,uint32_t code_bits) {
  DistortionTable error{};
  for (unsigned b=books.prefix_bits+1;b<=fine_cap;++b) error[b]=distortion(books.books[b]);
  auto fine=narrow;
  for (size_t rank=0;rank<fine.cap.size() && rank<wide_dims;++rank) {
    fine.cap[rank]=fine_cap;
    for (unsigned b=books.prefix_bits+1;b<=fine_cap;++b) fine.error[rank][b]=error[b];
  }
  auto bits=additive_wide_allocation(fine,code_bits,books.prefix_bits);
  if (widest_tier(bits)<=books.prefix_bits) return std::nullopt;
  return bits;
}

} // namespace dv::tieredq_detail
