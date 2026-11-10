// Coarse training options of TieredQIVFIndex (#48): restricted assignment that probes every super equals exact
// assignment label for label, few probes keep almost every exact label, the iteration count changes the trained
// lists, and invalid options or codec-list indexes are rejected.
#include "tieredq/tieredq.h"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
namespace {
using namespace dv;
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> bool rejects(F&& action) { try { action(); } catch (const std::runtime_error&) { return true; } return false; }
constexpr uint32_t dimensions=32,lists=512;
constexpr size_t rows=6000;

std::vector<float> gaussian_rows(size_t count,uint32_t seed) {
  std::mt19937 random(seed); std::normal_distribution<float> normal;
  std::vector<float> data(count*dimensions);
  for (size_t i=0;i<count;++i) for (uint32_t j=0;j<dimensions;++j) data[i*dimensions+j]=normal(random)*std::exp(-float(j)/dimensions);
  return data;
}
TieredQ codec(TieredQMetric metric,const std::vector<float>& data) {
  auto config=TieredQConfig::target_code_bits(dimensions,2*dimensions);
  config.metric=metric; config.fine_tiers=false;
  return TieredQ::train(config,data.data(),rows,3);
}
// Labels of data on fixed lists (the first rows as centroids), assigned as training says.
std::vector<uint32_t> labels(TieredQMetric metric,const std::vector<float>& data,TieredQCoarseTraining training) {
  TieredQIVFIndex ivf(codec(metric,data),lists,5);
  ivf.set_coarse_training(training);
  ivf.set_lists(gaussian_rows(lists,41).data());
  ivf.build(data.data(),rows);
  return ivf.list_assignment();
}

void restricted_assignment_test(TieredQMetric metric) {
  const auto data=gaussian_rows(rows,11);
  const auto exact=labels(metric,data,{});
  TieredQCoarseTraining every; every.assignment_supers=16; every.assignment_probes=16;
  require(labels(metric,data,every)==exact,"restricted assignment over every super differs from exact assignment");
  TieredQCoarseTraining few; few.assignment_supers=32; few.assignment_probes=12;
  const auto restricted=labels(metric,data,few);
  size_t agree=0; for (size_t row=0;row<rows;++row) agree+=restricted[row]==exact[row];
  require(agree>=rows*95/100,"restricted assignment loses too many exact labels");
  require(agree<rows,"12 of 32 supers never miss a label");
}

void training_options_test() {
  const auto data=gaussian_rows(rows,17);
  const auto model=codec(TieredQMetric::l2,data);
  auto trained=[&](TieredQCoarseTraining training) {
    TieredQIVFIndex ivf(model,lists,5); ivf.set_coarse_training(training); ivf.build(data.data(),rows);
    return ivf.list_assignment();
  };
  TieredQCoarseTraining one; one.iterations=1;
  TieredQCoarseTraining restricted; restricted.assignment_supers=32; restricted.assignment_probes=8;
  const auto full=trained({});
  require(trained(one)!=full,"one Lloyd iteration trains the same lists as twenty");
  require(trained(restricted).size()==rows,"restricted training does not build");
  require(trained({})==full,"coarse training is not deterministic");
  TieredQIVFIndex ivf(model,lists,5);
  require(rejects([&] { TieredQCoarseTraining bad; bad.iterations=0; ivf.set_coarse_training(bad); }),"zero iterations accepted");
  require(rejects([&] { TieredQCoarseTraining bad; bad.assignment_supers=8; bad.assignment_probes=0; ivf.set_coarse_training(bad); }),
          "zero probes accepted");
  require(rejects([&] { TieredQIVFIndex plain(model); plain.set_coarse_training({}); }),"coarse training on codec lists accepted");
}
} // namespace
void tieredq_ivf_training_tests() {
  for (auto metric:{TieredQMetric::inner_product,TieredQMetric::l2}) restricted_assignment_test(metric);
  training_options_test();
  std::cout<<"PASS IVF coarse training\n";
}
