// Returns heap a finished batch build freed back to the OS. glibc keeps freed medium blocks in its main heap and in
// one arena per build thread, so without this a built index holds its transient training and encoding buffers as idle
// resident memory (#41: ~20% of storage on VIBE arxiv with 34 build threads). No-op on other C libraries.
#pragma once
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace tq::tieredq_detail {
inline void release_freed_heap() {
#if defined(__GLIBC__)
  malloc_trim(0);
#endif
}
} // namespace tq::tieredq_detail
