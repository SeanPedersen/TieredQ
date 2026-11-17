// Unquantized IVF (#56): probing every list returns the exhaustive ranking for each metric, knn_graph equals a batch
// search of the stored rows with each row's own id removed (so its recall is per-query IVF's), fewer probes lose
// recall monotonically, short results are padded, and invalid input is rejected.
#include "tieredq/tieredq_ivf_flat.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
namespace {
using namespace tq;
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> bool rejects(F&& action) { try { action(); } catch (const std::runtime_error&) { return true; } return false; }
constexpr uint32_t dimensions=24,lists=16;
constexpr size_t rows=1500,k=10;

std::vector<float> gaussian_rows(size_t count,uint32_t seed) {
  std::mt19937 random(seed); std::normal_distribution<float> normal;
  std::vector<float> data(count*dimensions);
  for (auto& value:data) value=normal(random);
  return data;
}
double score(TieredQMetric metric,const float* q,const float* x) {
  double dot=0,qq=0,xx=0,l2=0;
  for (uint32_t j=0;j<dimensions;++j) { dot+=double(q[j])*x[j]; qq+=double(q[j])*q[j]; xx+=double(x[j])*x[j]; l2+=std::pow(double(q[j])-x[j],2); }
  if (metric==TieredQMetric::l2) return -l2;
  if (metric==TieredQMetric::cosine) return dot/std::sqrt(qq*xx);
  return dot;
}
// Exhaustive top k of query against data, skipping row `skip`.
std::vector<uint32_t> exhaustive(TieredQMetric metric,const std::vector<float>& data,const float* query,size_t skip) {
  std::vector<uint32_t> order(rows);
  std::iota(order.begin(),order.end(),0u);
  order.erase(std::remove(order.begin(),order.end(),uint32_t(skip)),order.end());
  std::vector<double> scores(rows);
  for (size_t r=0;r<rows;++r) scores[r]=score(metric,query,data.data()+r*dimensions);
  std::partial_sort(order.begin(),order.begin()+k,order.end(),[&](uint32_t a,uint32_t b) { return scores[a]>scores[b]; });
  order.resize(k);
  return order;
}
TieredQIVFFlatIndex built(TieredQMetric metric,const std::vector<float>& data) {
  TieredQIVFFlatConfig config; config.lists=lists; config.metric=metric;
  TieredQIVFFlatIndex index(dimensions,config);
  index.build(data.data(),rows);
  return index;
}

void exhaustive_test(TieredQMetric metric) {
  const auto data=gaussian_rows(rows,3);
  const auto index=built(metric,data);
  std::vector<float> scores(rows*k); std::vector<uint32_t> ids(rows*k);
  index.knn_graph(k,lists,scores.data(),ids.data());
  size_t matches=0;
  for (size_t r=0;r<rows;++r) {
    const auto truth=exhaustive(metric,data,data.data()+r*dimensions,r);
    for (size_t j=0;j<k;++j) matches+=std::count(truth.begin(),truth.end(),ids[r*k+j]);
    for (size_t j=1;j<k;++j) require(scores[r*k+j-1]>=scores[r*k+j],"knn_graph scores are not sorted");
    const double expected=score(metric,data.data()+r*dimensions,data.data()+size_t(ids[r*k])*dimensions);
    require(std::fabs(expected-scores[r*k])<=1e-3*(1+std::fabs(expected)),"knn_graph score is not the metric's");
  }
  // FP32 sgemm against double references: allow near-ties at the k-th place only.
  require(matches>=rows*k-rows/100,"full-probe knn_graph is not exhaustive");
}

void graph_equals_search_test() {
  const auto data=gaussian_rows(rows,5);
  const auto index=built(TieredQMetric::l2,data);
  constexpr size_t nprobe=3;
  std::vector<float> graph_scores(rows*k),search_scores(rows*(k+1));
  std::vector<uint32_t> graph(rows*k),searched(rows*(k+1));
  index.knn_graph(k,nprobe,graph_scores.data(),graph.data());
  index.search(data.data(),rows,k+1,nprobe,search_scores.data(),searched.data());
  for (size_t r=0;r<rows;++r) {
    std::vector<uint32_t> expected;
    for (size_t j=0;j<k+1 && expected.size()<k;++j) if (searched[r*(k+1)+j]!=r) expected.push_back(searched[r*(k+1)+j]);
    require(std::equal(expected.begin(),expected.end(),graph.begin()+ptrdiff_t(r*k)),"knn_graph differs from search");
  }
}

void probes_test() {
  const auto data=gaussian_rows(rows,7);
  const auto index=built(TieredQMetric::l2,data);
  std::vector<float> scores(rows*k); std::vector<uint32_t> ids(rows*k),exact(rows*k);
  index.knn_graph(k,lists,scores.data(),exact.data());
  size_t previous=0;
  for (size_t nprobe:{1,2,4,8,16}) {
    index.knn_graph(k,nprobe,scores.data(),ids.data());
    size_t hits=0;
    for (size_t r=0;r<rows;++r)
      for (size_t j=0;j<k;++j) hits+=std::count(exact.begin()+ptrdiff_t(r*k),exact.begin()+ptrdiff_t((r+1)*k),ids[r*k+j]);
    require(hits>=previous,"more probes lost recall");
    previous=hits;
  }
  require(previous==rows*k,"probing every list differs from itself");
}

void padding_and_rejects_test() {
  const auto data=gaussian_rows(8,9);
  TieredQIVFFlatConfig config; config.lists=2;
  TieredQIVFFlatIndex index(dimensions,config);
  index.build(data.data(),8);
  require(index.size()==8 && index.lists()==2,"small index shape");
  std::vector<float> scores(8*k); std::vector<uint32_t> ids(8*k);
  index.knn_graph(k,2,scores.data(),ids.data());
  for (size_t r=0;r<8;++r) {
    require(ids[r*k+6]!=std::numeric_limits<uint32_t>::max(),"seven other rows exist");
    require(ids[r*k+7]==std::numeric_limits<uint32_t>::max() && std::isinf(scores[r*k+7]),"short results are not padded");
  }
  const auto labels=index.list_assignment();
  require(labels.size()==8 && *std::max_element(labels.begin(),labels.end())<2,"list assignment");
  require(rejects([&] { TieredQIVFFlatIndex(dimensions,config).build(data.data(),1); }),"fewer rows than lists accepted");
  require(rejects([] { TieredQIVFFlatIndex(0); }),"zero dimensions accepted");
  require(rejects([] { TieredQIVFFlatConfig bad; bad.iterations=0; TieredQIVFFlatIndex(dimensions,bad); }),"zero iterations accepted");
  require(rejects([&] { TieredQIVFFlatIndex(dimensions).search(data.data(),1,k,1,scores.data(),ids.data()); }),"empty index searched");
  require(TieredQIVFFlatIndex::automatic_lists(100000)==256 && TieredQIVFFlatIndex::automatic_lists(1000000)==1024,
          "automatic lists");
}
} // namespace

void tieredq_ivf_flat_tests() {
  for (auto metric:{TieredQMetric::l2,TieredQMetric::inner_product,TieredQMetric::cosine}) exhaustive_test(metric);
  graph_equals_search_test();
  probes_test();
  padding_and_rejects_test();
  std::cout<<"PASS unquantized IVF and knn_graph\n";
}
