// Flat IVF (#56): probing every list returns the exhaustive ranking for each metric, knn_graph equals a batch search
// of the stored rows with each row's own id removed (so its recall is per-query IVF's), fewer probes lose recall
// monotonically, short results are padded, and invalid input is rejected. SQ8 storage built from codes whose decode is
// exact returns exactly what FP32 storage of the decoded rows returns; built from FP32 rows it keeps near-exact recall
// in about a quarter of the bytes.
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

// Codes with power-of-two scales and offsets on a 1/16 grid, so decoding is exact in FP32.
struct SQ8Store { std::vector<int8_t> codes; std::vector<float> scales,offsets,decoded; };
SQ8Store sq8_store(uint32_t seed) {
  std::mt19937 random(seed); std::uniform_int_distribution<int> level(-128,127);
  SQ8Store store{std::vector<int8_t>(rows*dimensions),std::vector<float>(dimensions),std::vector<float>(dimensions),
                 std::vector<float>(rows*dimensions)};
  for (uint32_t j=0;j<dimensions;++j) { store.scales[j]=std::ldexp(1.0f,-int(4+j%3)); store.offsets[j]=float(int(j)-12)/16; }
  for (size_t i=0;i<rows*dimensions;++i) {
    store.codes[i]=int8_t(level(random));
    store.decoded[i]=float(store.codes[i])*store.scales[i%dimensions]+store.offsets[i%dimensions];
  }
  return store;
}

void sq8_codes_equal_fp32_test(TieredQMetric metric) {
  const auto store=sq8_store(11);
  TieredQIVFFlatConfig config; config.lists=lists; config.metric=metric;
  TieredQIVFFlatIndex fp32(dimensions,config);
  fp32.build(store.decoded.data(),rows);
  config.storage=TieredQIVFFlatStorage::sq8;
  TieredQIVFFlatIndex sq8(dimensions,config);
  sq8.build(store.codes.data(),rows,store.scales.data(),store.offsets.data());
  require(sq8.list_assignment()==fp32.list_assignment(),"SQ8 build partitions differently");
  constexpr size_t nprobe=3;
  std::vector<float> expected_scores(rows*k),scores(rows*k);
  std::vector<uint32_t> expected(rows*k),ids(rows*k);
  fp32.knn_graph(k,nprobe,expected_scores.data(),expected.data());
  sq8.knn_graph(k,nprobe,scores.data(),ids.data());
  require(ids==expected && scores==expected_scores,"SQ8 knn_graph differs from FP32 of the decoded rows");
  fp32.search(store.decoded.data(),rows,k,nprobe,expected_scores.data(),expected.data());
  sq8.search(store.decoded.data(),rows,k,nprobe,scores.data(),ids.data());
  require(ids==expected && scores==expected_scores,"SQ8 search differs from FP32 of the decoded rows");
  require(sq8.storage_bytes()*2<fp32.storage_bytes(),"SQ8 storage is not smaller");
}

// Wide rows split each list into several decoded SQ8 tiles; the ranking still matches FP32 of the decoded rows.
void sq8_tiled_lists_test() {
  constexpr uint32_t wide=4096,wide_lists=2; constexpr size_t wide_rows=300;
  std::mt19937 random(17); std::uniform_int_distribution<int> level(-128,127);
  std::vector<int8_t> codes(wide_rows*wide); std::vector<float> scales(wide,1.0f/16),offsets(wide,0.0f),decoded(codes.size());
  for (size_t i=0;i<codes.size();++i) { codes[i]=int8_t(level(random)); decoded[i]=float(codes[i])/16; }
  TieredQIVFFlatConfig config; config.lists=wide_lists;
  TieredQIVFFlatIndex fp32(wide,config);
  fp32.build(decoded.data(),wide_rows);
  config.storage=TieredQIVFFlatStorage::sq8;
  TieredQIVFFlatIndex sq8(wide,config);
  sq8.build(codes.data(),wide_rows,scales.data(),offsets.data());
  std::vector<float> expected_scores(wide_rows*k),scores(wide_rows*k);
  std::vector<uint32_t> expected(wide_rows*k),ids(wide_rows*k);
  fp32.knn_graph(k,1,expected_scores.data(),expected.data());
  sq8.knn_graph(k,1,scores.data(),ids.data());
  size_t matches=0;
  for (size_t r=0;r<wide_rows;++r)
    for (size_t j=0;j<k;++j)
      matches+=std::count(expected.begin()+ptrdiff_t(r*k),expected.begin()+ptrdiff_t((r+1)*k),ids[r*k+j]);
  require(matches>=wide_rows*k-wide_rows/100,"tiled SQ8 lists differ from FP32");
  for (size_t i=0;i<scores.size();++i)
    require(std::fabs(scores[i]-expected_scores[i])<=1e-4f*(1+std::fabs(expected_scores[i])),"tiled SQ8 scores differ");
}

void sq8_from_float_test(TieredQMetric metric) {
  const auto data=gaussian_rows(rows,13);
  TieredQIVFFlatConfig config; config.lists=lists; config.metric=metric; config.storage=TieredQIVFFlatStorage::sq8;
  TieredQIVFFlatIndex index(dimensions,config);
  index.build(data.data(),rows);
  std::vector<float> scores(rows*k); std::vector<uint32_t> ids(rows*k);
  index.knn_graph(k,lists,scores.data(),ids.data());
  size_t matches=0;
  for (size_t r=0;r<rows;++r) {
    const auto truth=exhaustive(metric,data,data.data()+r*dimensions,r);
    for (size_t j=0;j<k;++j) matches+=std::count(truth.begin(),truth.end(),ids[r*k+j]);
    const double expected=score(metric,data.data()+r*dimensions,data.data()+size_t(ids[r*k])*dimensions);
    require(std::fabs(expected-scores[r*k])<=0.05*(1+std::fabs(expected)),"SQ8 score is far from the metric's");
  }
  require(matches>=rows*k*95/100,"SQ8 full-probe recall below 0.95");
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
  const std::vector<int8_t> codes(8*dimensions);
  require(rejects([&] { TieredQIVFFlatIndex(dimensions,config).build(codes.data(),8,nullptr,nullptr); }),"SQ8 build without scales");
  require(TieredQIVFFlatIndex::automatic_lists(100000)==256 && TieredQIVFFlatIndex::automatic_lists(1000000)==1024,
          "automatic lists");
}
// Given centroids replace k-means: their count sets the lists, training is skipped, and each row lands on its nearest
// given centroid (up to binary16 rounding of the centroids).
void given_centroids_test() {
  const auto data=gaussian_rows(rows,11);
  constexpr size_t given_lists=5;
  const std::vector<float> centroids(data.begin(),data.begin()+given_lists*dimensions);
  TieredQIVFFlatConfig config; config.lists=lists;
  TieredQIVFFlatIndex index(dimensions,config);
  index.set_centroids(centroids.data(),given_lists);
  index.build(data.data(),rows);
  require(index.lists()==given_lists && index.build_stats().train_s==0,"given centroids set the lists");
  const auto labels=index.list_assignment();
  size_t nearest_matches=0;
  for (size_t r=0;r<rows;++r) {
    size_t nearest=0; double best=-std::numeric_limits<double>::infinity();
    for (size_t c=0;c<given_lists;++c) {
      const double value=score(TieredQMetric::l2,data.data()+r*dimensions,centroids.data()+c*dimensions);
      if (value>best) { best=value; nearest=c; }
    }
    nearest_matches+=labels[r]==nearest;
  }
  require(nearest_matches>=rows*99/100,"rows go to their nearest given centroid");
  index.clear_centroids();
  index.build(data.data(),rows);
  require(index.lists()==lists,"cleared centroids train k-means again");
  require(rejects([&] { TieredQIVFFlatIndex(dimensions).set_centroids(nullptr,1); }),"null centroids accepted");
}
// A single BLAS thread changes scheduling only: lists and graph equal the default build's.
void blas_threads_cap_test() {
  const auto data=gaussian_rows(rows,3);
  const auto reference=built(TieredQMetric::l2,data);
  TieredQIVFFlatConfig config; config.lists=lists; config.blas_threads=1;
  TieredQIVFFlatIndex capped(dimensions,config);
  capped.build(data.data(),rows);
  require(capped.list_assignment()==reference.list_assignment(),"blas_threads=1 changed the list assignment");
  std::vector<float> scores(rows*k); std::vector<uint32_t> ids(rows*k),reference_ids(rows*k);
  capped.knn_graph(k,lists,scores.data(),ids.data());
  reference.knn_graph(k,lists,scores.data(),reference_ids.data());
  size_t matches=0;
  for (size_t i=0;i<ids.size();++i) matches+=ids[i]==reference_ids[i];
  require(matches>=ids.size()-ids.size()/100,"blas_threads=1 changed the knn graph");
}
} // namespace

void tieredq_ivf_flat_tests() {
  blas_threads_cap_test();
  for (auto metric:{TieredQMetric::l2,TieredQMetric::inner_product,TieredQMetric::cosine}) exhaustive_test(metric);
  graph_equals_search_test();
  probes_test();
  for (auto metric:{TieredQMetric::l2,TieredQMetric::inner_product}) sq8_codes_equal_fp32_test(metric);
  for (auto metric:{TieredQMetric::l2,TieredQMetric::inner_product,TieredQMetric::cosine}) sq8_from_float_test(metric);
  sq8_tiled_lists_test();
  padding_and_rejects_test();
  given_centroids_test();
  std::cout<<"PASS flat IVF (FP32 and SQ8) and knn_graph\n";
}
