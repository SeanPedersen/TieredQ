// Bounded-memory fast-search orchestration: scan cache tiles and select before reusing scratch.
// Documents are split into segments when query tiles alone cannot occupy every worker; per-segment
// top-k lists merge with the same score/id order, so results do not depend on the split.
#pragma once
#include "tieredq_search.h"

namespace dv::tieredq_detail {
// Each concurrent worker holds a query_tile x document_tile score scratch. One worker keeps the wide tile;
// parallel searches use a narrow one, which bounds the multi-worker peak and measured faster there.
constexpr size_t fast_search_document_tile = 4096;
constexpr size_t fast_search_parallel_document_tile = 1024;
static_assert(fast_search_document_tile % selection_block == 0);
static_assert(fast_search_parallel_document_tile % selection_block == 0);

// Document segments per query tile: enough to give each worker a task, in whole document tiles.
inline size_t fast_search_segments(size_t query_tiles,size_t documents,size_t workers,size_t document_tile) {
  const size_t document_tiles=(documents+document_tile-1)/document_tile;
  return std::max<size_t>(1,std::min(document_tiles,(workers+query_tiles-1)/query_tiles));
}

template<size_t document_tile,class Prepared,class Scan,class Finish>
void search_fast_document_tiles(const std::vector<Prepared>& queries,size_t documents,size_t k,
    size_t query_tile,size_t available,Scan scan,Finish finish) {
  const size_t tiles=(queries.size()+query_tile-1)/query_tile;
  const size_t segments=fast_search_segments(tiles,documents,available,document_tile);
  const size_t document_tiles=(documents+document_tile-1)/document_tile;
  const size_t segment_tiles=(document_tiles+segments-1)/segments;
  const size_t tasks=tiles*segments,workers=std::min(tasks,available);
  std::vector<std::vector<TieredQHit>> partial(segments*queries.size());
  std::exception_ptr failure;
  std::mutex failure_mutex;
#pragma omp parallel for schedule(static) num_threads(workers) if(workers>1 && !omp_in_parallel())
  for (size_t task=0;task<tasks;++task) {
    try {
      const size_t tile=task/segments,segment=task%segments;
      const size_t first=tile*query_tile,rows=std::min(query_tile,queries.size()-first);
      const size_t begin_document=std::min(documents,segment*segment_tiles*document_tile);
      const size_t end_document=std::min(documents,begin_document+segment_tiles*document_tile);
      const size_t stride=std::min(documents,document_tile);
      std::vector<float> scratch(rows*stride);
      std::vector<TopK> top;
      top.reserve(rows);
      for (size_t row=0;row<rows;++row) top.emplace_back(k);
      for (size_t begin=begin_document;begin<end_document;begin+=stride) {
        const size_t count=std::min(stride,end_document-begin);
        if constexpr (requires {scan(queries.data()+first,rows,begin,count,scratch.data(),stride,top.data());})
          scan(queries.data()+first,rows,begin,count,scratch.data(),stride,top.data());
        else scan(queries.data()+first,rows,begin,count,scratch.data(),stride);
        for (size_t row=0;row<rows;++row) for (size_t block=0;block<count;block+=selection_block)
          top[row].add(scratch.data()+row*stride+block,std::min(selection_block,count-block),begin+block);
      }
      for (size_t row=0;row<rows;++row) partial[(first+row)*segments+segment]=top[row].finish();
    } catch (...) {
      std::lock_guard lock(failure_mutex);
      if (!failure) failure=std::current_exception();
    }
  }
  if (failure) std::rethrow_exception(failure);
  for (size_t row=0;row<queries.size();++row) {
    if (segments==1) { finish(row,std::move(partial[row])); continue; }
    std::vector<TieredQHit> hits;
    hits.reserve(segments*k);
    for (size_t segment=0;segment<segments;++segment)
      hits.insert(hits.end(),partial[row*segments+segment].begin(),partial[row*segments+segment].end());
    std::sort(hits.begin(),hits.end(),better_hit);
    hits.resize(std::min(k,hits.size()));
    finish(row,std::move(hits));
  }
}

template<class Prepared,class Scan,class Finish>
void search_fast_tiles(const std::vector<Prepared>& queries,size_t documents,size_t k,
    size_t query_tile,Scan scan,Finish finish) {
  if (queries.empty()) return;
  const size_t available=omp_in_parallel() ? 1 : std::max<size_t>(1,size_t(omp_get_max_threads()));
  if (available>1)
    search_fast_document_tiles<fast_search_parallel_document_tile>(queries,documents,k,query_tile,available,scan,finish);
  else search_fast_document_tiles<fast_search_document_tile>(queries,documents,k,query_tile,available,scan,finish);
}
} // namespace dv::tieredq_detail
