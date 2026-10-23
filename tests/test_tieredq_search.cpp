// Streaming fast search crosses document/query tiles without changing scores, ties or ownership.
#include "tieredq/tieredq.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>
namespace {
void require(bool condition,const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void streaming_search_test(uint32_t code_bits,uint32_t max_bits,bool fast_mode=false) {
  constexpr uint32_t dimensions=8,centroids=4;
  constexpr size_t training_rows=128,documents=4109,query_count=17;
  std::mt19937 random(17);
  std::normal_distribution<float> normal;
  std::vector<float> training(training_rows*dimensions),rows(documents*dimensions),queries(query_count*dimensions);
  for (auto& value:training) value=normal(random);
  for (auto& value:rows) value=normal(random);
  for (auto& value:queries) value=normal(random);
  std::fill_n(queries.data(),dimensions,0);
  for (size_t row=1;row<documents;row+=2) std::copy_n(rows.data(),dimensions,rows.data()+row*dimensions);
  auto codec=dv::TieredQ::train({dimensions,code_bits,centroids,false,max_bits},training.data(),training_rows);
  dv::TieredQIndex index(std::move(codec),fast_mode);
  index.add(rows.data(),documents);
  index.build_fast_scan();
  if (!index.has_fast_scan()) return;
  for (size_t k:{size_t(1),size_t(10),documents+1}) {
    const size_t found=std::min(k,documents);
    std::vector<float> scores(query_count*found);
    std::vector<uint32_t> ids(query_count*found);
    index.search_batch_fast(queries.data(),query_count,k,scores.data(),ids.data());
    for (size_t query=0;query<query_count;++query) {
      const auto hits=index.search_fast(queries.data()+query*dimensions,k);
      for (size_t hit=0;hit<found;++hit) {
        const size_t offset=query*found+hit;
        require(ids[offset]==hits[hit].id && scores[offset]==hits[hit].score,"streamed batch differs from single");
        if (!query) require(ids[offset]==hit,"zero-query ties do not retain lowest ids");
      }
    }
  }
}
} // namespace
void tieredq_search_tests() {
  streaming_search_test(16,4);
  streaming_search_test(64,8);
  streaming_search_test(16,4,true);
  streaming_search_test(64,8,true);
}
