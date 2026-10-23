// Bounded-memory fast-search orchestration: scan cache tiles and select before reusing scratch.
#pragma once
#include "tieredq_search.h"

namespace dv::tieredq_detail {
constexpr size_t fast_search_document_tile = 4096;
static_assert(fast_search_document_tile % selection_block == 0);

template<class Prepared,class Scan,class Finish>
void search_fast_tiles(const std::vector<Prepared>& queries,size_t documents,size_t k,
    size_t query_tile,Scan scan,Finish finish) {
  const size_t tiles=(queries.size()+query_tile-1)/query_tile;
  if (!tiles) return;
  const size_t workers=std::min(tiles,size_t(omp_get_max_threads()));
  std::exception_ptr failure;
  std::mutex failure_mutex;
#pragma omp parallel for schedule(static) num_threads(workers) if(workers>1 && !omp_in_parallel())
  for (size_t tile=0;tile<tiles;++tile) {
    try {
      const size_t first=tile*query_tile,rows=std::min(query_tile,queries.size()-first);
      const size_t stride=std::min(documents,fast_search_document_tile);
      std::vector<float> scratch(rows*stride);
      std::vector<TopK> top;
      top.reserve(rows);
      for (size_t row=0;row<rows;++row) top.emplace_back(k);
      for (size_t begin=0;begin<documents;begin+=stride) {
        const size_t count=std::min(stride,documents-begin);
        scan(queries.data()+first,rows,begin,count,scratch.data(),stride);
        for (size_t row=0;row<rows;++row) for (size_t block=0;block<count;block+=selection_block)
          top[row].add(scratch.data()+row*stride+block,std::min(selection_block,count-block),begin+block);
      }
      for (size_t row=0;row<rows;++row) finish(first+row,top[row].finish());
    } catch (...) {
      std::lock_guard lock(failure_mutex);
      if (!failure) failure=std::current_exception();
    }
  }
  if (failure) std::rethrow_exception(failure);
}
} // namespace dv::tieredq_detail
