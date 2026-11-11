// Per-model geometry of the grouped FP32 scan: one entry per scan group, in scoring order.
// Window and shift derive only from the tier layout, so they are computed once per model and never in the
// 16-document hot loop. Table offsets index TieredQQuery::scan_tables; wide groups store UINT32_MAX there.
#pragma once
#include <cstdint>

namespace tq::tieredq_detail {
struct ScanGroupPlan {
  uint32_t window, shift, left, first, bits, table;
  bool straddles;
};
}  // namespace tq::tieredq_detail
