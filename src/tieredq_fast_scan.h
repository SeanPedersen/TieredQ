// Fast-scan backend dispatch shared by flat and IVF search: per-backend query preparation and range scanners.
// with_fast_scan hands the caller the prepared queries, the backend's query tile and a scanner over any document
// range [first, first + documents) whose start is aligned to fast_list_alignment.
#pragma once
#include "tieredq/tieredq.h"
#include "tieredq_search.h"
#include "tieredq_scan_int8.h"
#include "tieredq_fast_dispatch.h"
#include "tieredq_scan_fast_scalar.h"
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define DV_TIEREDQ_X86 1
#include "tieredq_scan_int8_avx512.h"
#include "tieredq_scan_int8_avx2.h"
#include "tieredq_range_checkpoint_avx512.h"
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#define DV_TIEREDQ_NEON 1
#include "tieredq_scan_int8_neon.h"
#endif

namespace dv::tieredq_detail {
// Range starts every fast kernel accepts: 16-document int8/code blocks and the 32-document AVX2 lookup chunks.
constexpr size_t fast_list_alignment = 32;

// Code-ID scans consume only query weights and lookup scans only their byte tables; both project query tiles so
// the basis traffic is shared across queries.
template<class Prepare>
auto fast_search_queries(const FastScanCache& cache,const TieredQ& codec,const float* queries,size_t rows,Prepare prepare) {
  using Prepared=decltype(prepare(queries));
  if constexpr (requires (Prepared query) { query.lookup.tables; })
    if (cache.lookup) return prepare_search_tiles<Prepared>(queries,rows,codec.model_dimensions(),query_tile_height(rows,basis_query_tile),
        [&](const float* first,size_t count,Prepared* out) {
          const auto outputs=project_fast_queries(cache.lookup->projection,first,count,FastQueryBasis::from(codec));
          for (size_t row=0;row<count;++row) out[row].lookup=fast_lookup_query(*cache.lookup,outputs[row]);
        });
  if (!cache.code_lookup()) return prepare_search_queries(queries,rows,codec.model_dimensions(),prepare);
  auto weights=codec.model_query_weights(queries,rows);
  std::vector<Prepared> prepared(rows);
  for (size_t row=0;row<rows;++row) prepared[row].codes=std::move(weights[row]);
  return prepared;
}

// Query-to-centroid scores of a prepared query, whichever representation its backend filled.
template<class Prepared>
const std::vector<float>& prepared_centroid_scores(const Prepared& query) {
  if constexpr (requires { query.lookup.centroid_scores; })
    if (!query.lookup.centroid_scores.empty()) return query.lookup.centroid_scores;
  if (!query.centroid_scores.empty()) return query.centroid_scores;
  return query.codes.centroid_scores;
}

// Calls run(prepared, query_tile, scan) for model-space queries, where
// scan(const Prepared* queries, size_t count, size_t first, size_t documents, float* scores, size_t stride).
template<class Run>
void with_fast_scan(const FastScanCache& cache,const TieredQ& codec,const float* queries,size_t rows,Run run) {
  [[maybe_unused]] const auto backend=fast_scan_backend(cache);
#ifdef DV_TIEREDQ_NEON
  if (backend==FastScanBackend::neon) {
    auto prepare=[&](const float* vector) { return fast_scan_query(cache,codec.model_query_weights(vector)); };
    auto scan=[&](const auto* prepared,size_t count,size_t first,size_t documents,float* scores,size_t stride) {
      scan_fast_neon_tile(cache,prepared,count,first,documents,scores,stride);
    };
    run(fast_search_queries(cache,codec,queries,rows,prepare),
        cache.code_lookup() ? fast_code_query_group : fast_scan_query_tile_neon,scan);
    return;
  }
#endif
#ifdef DV_TIEREDQ_X86
  if (backend==FastScanBackend::avx512 || backend==FastScanBackend::avx2) {
    auto prepare=[&](const float* vector) { return fast_scan_query_x86(cache,codec,vector); };
    auto scan=[&](const auto* prepared,size_t count,size_t first,size_t documents,float* scores,size_t stride) {
      if (backend==FastScanBackend::avx512) scan_fast_avx512_tile(cache,prepared,count,first,documents,scores,stride);
      else scan_fast_avx2_tile(cache,prepared,count,first,documents,scores,stride);
    };
    const size_t query_tile=cache.lookup ? 1 : cache.code_lookup() ? fast_code_query_group :
        backend==FastScanBackend::avx512 ? fast_scan_query_tile_x86 : fast_scan_query_tile_avx2;
    run(fast_search_queries(cache,codec,queries,rows,prepare),query_tile,scan);
    return;
  }
#endif
  auto prepare=[&](const float* vector) { return fast_scan_query_scalar(cache,codec,vector); };
  auto scan=[&](const auto* prepared,size_t count,size_t first,size_t documents,float* scores,size_t stride) {
    scan_fast_scalar_tile(cache,prepared,count,first,documents,scores,stride);
  };
  run(fast_search_queries(cache,codec,queries,rows,prepare),size_t(1),scan);
}
} // namespace dv::tieredq_detail
