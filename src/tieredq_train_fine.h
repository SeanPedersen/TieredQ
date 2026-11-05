// Fine tiers of fast-target training (#26): codes of up to widest_fine_code_bits whose top prefix_bits are an
// additive fast code at the automatic cap. The fast scan reads only those prefixes and rescores its candidates
// exactly, so the scan keeps the prefix width's speed. Training allocates with and without fine tiers under the same
// stored bytes and keeps the fine allocation when it is forced or predicted to pay off.
#pragma once
#include "tieredq_lookup_layout.h"
#include "tieredq_train_allocation.h"
#include "tieredq_train_codebook.h"
#include <cmath>
#include <optional>
#include <vector>

namespace dv::tieredq_detail {

// Automatic training adds fine tiers only when they cut the predicted weighted error by at least this much. At 6-8
// bits/dim on 100k Gemini-768 and OpenAI-1536 fine tiers predicted 8.2-9.2 dB (misses@10 -58 to -76%), on GloVe-200
// -0.2 dB (#26).
constexpr double fine_tier_min_gain_db=1.0;

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

inline double predicted_gain_db(double before,double after) {
  constexpr double decibels_per_decade=10;
  return after>0 ? decibels_per_decade*std::log10(before/after) : INFINITY;
}

// The fine-tier allocation of the weighted additive model `narrow` (whose allocation is narrow_bits): ranks below
// wide_dims may reach fine_cap bits through the fine books. Returned when forced or when it predicts at least
// fine_tier_min_gain_db less weighted error than narrow_bits; never when it allocates no code wider than the prefix.
inline std::optional<Allocation> fine_allocation(const AllocationModel& narrow,const Allocation& narrow_bits,
    const FineBooks& books,uint32_t fine_cap,uint32_t wide_dims,uint32_t code_bits,bool forced) {
  DistortionTable error{};
  for (unsigned b=books.prefix_bits+1;b<=fine_cap;++b) error[b]=distortion(books.books[b]);
  auto fine=narrow;
  for (size_t rank=0;rank<fine.cap.size() && rank<wide_dims;++rank) {
    fine.cap[rank]=fine_cap;
    for (unsigned b=books.prefix_bits+1;b<=fine_cap;++b) fine.error[rank][b]=error[b];
  }
  auto bits=additive_wide_allocation(fine,code_bits,books.prefix_bits);
  if (widest_tier(bits)<=books.prefix_bits) return std::nullopt;
  const double gain=predicted_gain_db(predicted_distortion(narrow,narrow_bits),predicted_distortion(fine,bits));
  if (!forced && gain<fine_tier_min_gain_db) return std::nullopt;
  return bits;
}

} // namespace dv::tieredq_detail
