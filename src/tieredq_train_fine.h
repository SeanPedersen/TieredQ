// Fine tiers (#26, #31): codes of up to widest_fine_code_bits whose top prefix_bits are an ordinary code of the scan's
// prefix book. The fast scan reads only those prefixes and rescores its candidates exactly, so it keeps the prefix
// width's speed. Fast targets refine the additive prefix book that the nibble scan needs; full targets (exact fast
// scores) refine the 8-bit Lloyd-Max book that the exact scan reads. Training keeps the fine allocation whenever it codes
// a rank wider than the prefix: on 100k Gemini-768, OpenAI-1536 and GloVe-200 at 3-8 bits/dim it never cost recall,
// while a 1 dB predicted-gain gate declined cases where it helped (GloVe at 6-8, every dataset at 3-4 bits/dim; #31).
#pragma once
#include "tieredq_lookup_layout.h"
#include "tieredq_train_allocation.h"
#include "tieredq_train_codebook.h"
#include <optional>
#include <vector>

namespace tq::tieredq_detail {

// books[b] for prefix_bits < b <= widest_fine_code_bits refines the prefix book by b - prefix_bits bits.
struct FineBooks {
  unsigned prefix_bits;
  std::vector<std::vector<float>> books;
};
inline FineBooks fine_codebooks(const std::vector<float>& prefix,unsigned prefix_bits) {
  FineBooks out{prefix_bits,std::vector<std::vector<float>>(widest_fine_code_bits+1)};
  for (unsigned b=prefix_bits+1;b<=widest_fine_code_bits;++b) out.books[b]=refined_prefix_codebook(prefix,b-prefix_bits);
  return out;
}

// The allocation model `narrow` with ranks below wide_dims allowed up to fine_cap bits through the fine books.
inline AllocationModel with_fine_books(AllocationModel narrow,const FineBooks& books,uint32_t fine_cap,uint32_t wide_dims) {
  DistortionTable error{};
  for (unsigned b=books.prefix_bits+1;b<=fine_cap;++b) error[b]=distortion(books.books[b]);
  for (size_t rank=0;rank<narrow.cap.size() && rank<wide_dims;++rank) {
    narrow.cap[rank]=fine_cap;
    for (unsigned b=books.prefix_bits+1;b<=fine_cap;++b) narrow.error[rank][b]=error[b];
  }
  return narrow;
}

// A fine-tier allocation, or none when it codes no rank wider than the prefix.
inline std::optional<Allocation> fine_or_none(Allocation bits,unsigned prefix_bits) {
  if (widest_tier(bits)<=prefix_bits) return std::nullopt;
  return bits;
}

// Fast targets: the additive model's allocation, whose prefixes keep the nibble lookup layout.
inline std::optional<Allocation> additive_fine_allocation(const AllocationModel& narrow,const FineBooks& books,
    uint32_t fine_cap,uint32_t wide_dims,uint32_t code_bits) {
  return fine_or_none(additive_wide_allocation(with_fine_books(narrow,books,fine_cap,wide_dims),code_bits,books.prefix_bits),
                      books.prefix_bits);
}

// Full targets: dense Lloyd-Max codes, whose prefixes scan through the exact packed-code layout.
inline std::optional<Allocation> lloyd_max_fine_allocation(const AllocationModel& narrow,const FineBooks& books,
    uint32_t fine_cap,uint32_t wide_dims,uint32_t code_bits) {
  return fine_or_none(greedy_allocation(with_fine_books(narrow,books,fine_cap,wide_dims),code_bits,false),books.prefix_bits);
}

} // namespace tq::tieredq_detail
