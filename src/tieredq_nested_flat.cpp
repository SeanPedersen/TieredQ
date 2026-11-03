// Flat nested search (#13): a one-list TieredQNestedIVFIndex whose single list holds every document, with the tier
// encoded into RAM at construction.
#include "tieredq/tieredq_nested_flat.h"
#include <memory>
#include <utility>

namespace dv {
namespace {
constexpr uint32_t scan_lists=1;
constexpr size_t scan_nprobe=1;

TieredQNestedIVFIndex nested_with_tier(TieredQ prefix,TieredQRefinement refinement,const float* rows,size_t count,
                                       TieredQTierRows tier_rows) {
  TieredQIVFIndex list(std::move(prefix),scan_lists);
  list.build(rows,count);
  TieredQNestedIVFIndex nested(std::move(list),std::move(refinement),tier_rows);
  const size_t row_bytes=nested.tier_bytes_per_vector();
  std::vector<uint8_t> tier(count*row_bytes);
  nested.encode_tier(rows,0,count,tier.data());
  nested.attach_tier(std::make_shared<TieredQMemoryRows>(std::move(tier),row_bytes));
  return nested;
}
} // namespace

TieredQNestedIndex::TieredQNestedIndex(TieredQ prefix,TieredQRefinement refinement,const float* rows,size_t count,
                                       TieredQTierRows tier_rows)
    : nested_(nested_with_tier(std::move(prefix),std::move(refinement),rows,count,tier_rows)) {}

size_t TieredQNestedIndex::storage_bytes() const {
  return nested_.index().storage_bytes()+nested_.size()*nested_.tier_bytes_per_vector();
}

std::vector<TieredQHit> TieredQNestedIndex::search(const float* query,size_t k,size_t candidates) const {
  return nested_.search(query,k,scan_nprobe,candidates);
}

void TieredQNestedIndex::search_batch(const float* queries,size_t rows,size_t k,size_t candidates,float* scores,
                                      uint32_t* ids) const {
  nested_.search_batch(queries,rows,k,scan_nprobe,candidates,scores,ids);
}
} // namespace dv
