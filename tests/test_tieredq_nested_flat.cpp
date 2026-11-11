// Flat nested search (#13): rescoring every document returns the refinement's exhaustive ranking, batch and single
// search agree for either tier layout, and suffix rescoring of k' candidates beats the prefix ranking alone.
#include "tieredq/tieredq_nested_flat.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
namespace {
using namespace tq;
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
constexpr uint32_t dimensions=48,centroids=8,seed=13;
constexpr size_t rows=1999,queries=24,k=10,candidates=40;
constexpr uint32_t prefix_bits=48,total_bits=192;

std::vector<float> clustered_rows(size_t count,uint32_t noise_seed) {
  constexpr uint32_t center_seed=5;
  constexpr float decay=4,spread=3;
  std::mt19937 center_random(center_seed),random(noise_seed); std::normal_distribution<float> normal;
  std::vector<float> centers(centroids*dimensions);
  for (auto& value:centers) value=spread*normal(center_random);
  std::vector<float> data(count*dimensions);
  for (size_t i=0;i<count;++i) for (uint32_t j=0;j<dimensions;++j)
    data[i*dimensions+j]=centers[(i%centroids)*dimensions+j]+normal(random)*std::exp(-decay*float(j)/dimensions);
  return data;
}
TieredQNestedIndex flat(const std::vector<float>& data,TieredQTierRows tier_rows) {
  auto config=TieredQConfig::target_code_bits(dimensions,prefix_bits);
  config.num_centroids=centroids; config.fine_tiers=false;
  auto prefix=TieredQ::train(config,data.data(),rows,seed);
  auto refinement=TieredQRefinement::train(prefix,total_bits,data.data(),rows,seed);
  return TieredQNestedIndex(std::move(prefix),std::move(refinement),data.data(),rows,tier_rows);
}
std::vector<uint32_t> true_top(const std::vector<float>& data,const float* query) {
  std::vector<float> dots(rows);
  for (size_t i=0;i<rows;++i) dots[i]=std::inner_product(query,query+dimensions,data.data()+i*dimensions,0.0f);
  std::vector<uint32_t> order(rows); std::iota(order.begin(),order.end(),0);
  std::partial_sort(order.begin(),order.begin()+k,order.end(),[&](uint32_t a,uint32_t b) { return dots[a]>dots[b]; });
  order.resize(k);
  return order;
}
size_t hits_in(const std::vector<uint32_t>& truth,const std::vector<TieredQHit>& hits) {
  size_t found=0;
  for (const auto& hit:hits) found+=std::find(truth.begin(),truth.end(),hit.id)!=truth.end();
  return found;
}

void exhaustive_test() {
  const auto data=clustered_rows(rows,seed),probes=clustered_rows(queries,seed+1);
  const auto index=flat(data,TieredQTierRows::suffix);
  const auto& refinement=index.nested().refinement();
  const auto prefix=index.nested().index().codec().encode(data.data(),rows);
  std::vector<uint8_t> suffix(rows*refinement.bytes_per_vector());
  refinement.encode(data.data(),rows,prefix.data(),suffix.data());
  require(index.size()==rows && index.storage_bytes()>=rows*index.nested().tier_bytes_per_vector(),"flat nested shape");
  for (size_t q=0;q<queries;++q) {
    const float* query=probes.data()+q*dimensions;
    std::vector<float> scores(rows);
    refinement.score(prefix.data(),suffix.data(),rows,refinement.query(query),scores.data());
    std::vector<uint32_t> order(rows); std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b) { return scores[a]>scores[b]; });
    const auto hits=index.search(query,k,rows);
    require(hits.size()==k,"flat nested search returned the wrong hit count");
    for (size_t i=0;i<k;++i) require(hits[i].id==order[i],"flat nested hits differ from the exhaustive refinement ranking");
  }
}

void layout_test() {
  const auto data=clustered_rows(rows,seed),probes=clustered_rows(queries,seed+1);
  const auto suffix=flat(data,TieredQTierRows::suffix),both=flat(data,TieredQTierRows::prefix_and_suffix);
  std::vector<float> scores(queries*k); std::vector<uint32_t> ids(queries*k);
  suffix.search_batch(probes.data(),queries,k,candidates,scores.data(),ids.data());
  size_t nested_hits=0,prefix_hits=0;
  for (size_t q=0;q<queries;++q) {
    const float* query=probes.data()+q*dimensions;
    const auto hits=suffix.search(query,k,candidates),other=both.search(query,k,candidates);
    for (size_t i=0;i<k;++i) {
      require(hits[i].id==ids[q*k+i] && hits[i].score==scores[q*k+i],"flat nested batch differs from single search");
      require(hits[i].id==other[i].id && hits[i].score==other[i].score,"flat nested tier layouts disagree");
    }
    const auto truth=true_top(data,query);
    nested_hits+=hits_in(truth,hits);
    prefix_hits+=hits_in(truth,suffix.nested().index().search(query,k,1));
  }
  require(nested_hits>prefix_hits,"flat suffix rescoring does not beat the prefix ranking");
}
} // namespace

void tieredq_nested_flat_tests() {
  exhaustive_test();
  layout_test();
  std::cout<<"PASS flat nested search\n";
}
