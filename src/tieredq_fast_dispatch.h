// Shared fast-kernel selection for query execution and user-visible fallback diagnostics.
#pragma once
#include "tieredq_scan_int8.h"
namespace dv::tieredq_detail {
enum class FastScanBackend { scalar, neon, avx512 };
inline bool native_fast_scan_available() {
#if defined(__aarch64__) || defined(_M_ARM64)
  return true;
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
#else
  return false;
#endif
}
inline FastScanBackend fast_scan_backend(const FastScanCache& cache) {
#if defined(__aarch64__) || defined(_M_ARM64)
  if (!cache.lookup) return FastScanBackend::neon;
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
  if (native_fast_scan_available() && (cache.lookup || cache.level_max<=x86_fast_level_max))
    return FastScanBackend::avx512;
#endif
  return FastScanBackend::scalar;
}
} // namespace dv::tieredq_detail
