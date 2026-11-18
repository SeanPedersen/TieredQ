// E-means (#45): the resolved recipe (batch, initial step, epoch layout, step count, schedule) equals Faiss's
// reference (faiss/contrib/e_means at fc1095a, values computed with its base.py), training on clustered data reaches
// Lloyd's objective without empty clusters and is deterministic per seed, restricted batch assignment (supers) stays
// within a few percent of exact assignment's objective, TieredQIVFFlatIndex and TieredQIVFIndex train with it, and
// training through a row source equals training on the rows in RAM.
#include "tieredq/tieredq.h"
#include "tieredq/tieredq_ivf_flat.h"
#include "tieredq_ivf_coarse.h"
#include "tieredq_train_emeans.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
namespace {
using namespace tq;
using tieredq_detail::emeans_recipe;
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
bool close_to(double a,double b) { return std::fabs(a-b)<=1e-12*std::fabs(b); }

struct ReferenceRecipe {
  size_t rows,lists,dimensions; double epochs;
  size_t batch; double alpha0; std::vector<size_t> epoch_rows; size_t steps; double middle_alpha;
};
void recipe_test() {
  const std::vector<ReferenceRecipe> reference={
      {262144,1024,1024,2.5,8192,0.37169433629366594,{262144,131072,262144},80,0.2044318849615163},
      {655360,1024,1024,1.0,8192,0.27117820805116016,{655360},80,0.1491480144281381},
      {100000,256,8,3.0,512,0.0878977221077138,{100000,100000,100000},588,0.04834374715924259},
      {1000,16,24,1.5,32,0.22246898148585056,{500,1000},48,0.12235793981721782},
      {4000,40,24,3.0,256,0.4720272001146637,{4000,4000,4000},48,0.25961496006306506}};
  for (const auto& r:reference) {
    const auto recipe=emeans_recipe(r.rows,r.lists,r.dimensions,r.epochs);
    require(recipe.batch_rows==r.batch,"e-means batch law");
    require(close_to(recipe.alpha0,r.alpha0),"e-means step law");
    require(recipe.epoch_rows==r.epoch_rows,"e-means epoch layout");
    require(recipe.total_steps==r.steps,"e-means step count");
    require(close_to(recipe.alpha(r.steps/2),r.middle_alpha),"e-means cosine schedule");
    require(recipe.unused_threshold==(r.dimensions<=2 ? 0.0 : 0.1),"e-means revival threshold");
  }
}

constexpr size_t dimensions=24,clusters=40,rows=4000;
// Gaussian blobs of unequal spread around uniform centres, so poor seeding leaves some blobs without a centroid.
std::vector<float> blobs(uint32_t seed) {
  std::mt19937 random(seed); std::normal_distribution<float> normal; std::uniform_real_distribution<float> uniform(-8,8);
  std::vector<float> centres(clusters*dimensions),data(rows*dimensions);
  for (auto& value:centres) value=uniform(random);
  for (size_t r=0;r<rows;++r) {
    const size_t c=r%clusters; const float spread=0.3f+0.05f*float(c%7);
    for (size_t j=0;j<dimensions;++j) data[r*dimensions+j]=centres[c*dimensions+j]+spread*normal(random);
  }
  return data;
}
// Hierarchical blobs, as embeddings are: groups far apart, each holding clusters / groups nearby blobs. Restricted
// assignment relies on such structure (a super groups nearby centroids); uniform blobs leave supers without meaning.
std::vector<float> grouped_blobs(uint32_t seed,size_t groups) {
  std::mt19937 random(seed); std::normal_distribution<float> normal;
  std::uniform_real_distribution<float> far(-20,20),near(-3,3);
  std::vector<float> group_centres(groups*dimensions),centres(clusters*dimensions),data(rows*dimensions);
  for (auto& value:group_centres) value=far(random);
  for (size_t c=0;c<clusters;++c)
    for (size_t j=0;j<dimensions;++j) centres[c*dimensions+j]=group_centres[(c%groups)*dimensions+j]+near(random);
  for (size_t r=0;r<rows;++r)
    for (size_t j=0;j<dimensions;++j) data[r*dimensions+j]=centres[(r%clusters)*dimensions+j]+0.3f*normal(random);
  return data;
}
// Sum of squared distances to the nearest centroid, and the number of centroids no row is nearest to.
std::pair<double,size_t> objective(const std::vector<float>& data,const std::vector<float>& centroids) {
  const size_t k=centroids.size()/dimensions;
  std::vector<size_t> used(k);
  double total=0;
  for (size_t r=0;r<rows;++r) {
    double best=std::numeric_limits<double>::infinity(); size_t nearest=0;
    for (size_t c=0;c<k;++c) {
      double distance=0;
      for (size_t j=0;j<dimensions;++j) distance+=std::pow(double(data[r*dimensions+j])-centroids[c*dimensions+j],2);
      if (distance<best) { best=distance; nearest=c; }
    }
    total+=best; ++used[nearest];
  }
  size_t empty=0;
  for (const size_t count:used) empty+=count==0;
  return {total,empty};
}
void quality_test() {
  const auto data=blobs(5);
  constexpr uint64_t seed=7;
  const auto emeans=tieredq_detail::train_emeans(data.data(),rows,dimensions,clusters,seed,3.0);
  const auto lloyd=tieredq_detail::train_coarse(data.data(),rows,dimensions,clusters,seed,10).centroids();
  const auto [emeans_objective,emeans_empty]=objective(data,emeans);
  const double lloyd_objective=objective(data,lloyd).first;
  require(emeans_empty==0,"e-means leaves empty clusters");
  require(emeans_objective<=1.02*lloyd_objective,"e-means misses Lloyd's objective");
  require(tieredq_detail::train_emeans(data.data(),rows,dimensions,clusters,seed,3.0)==emeans,"e-means is not deterministic");
  require(tieredq_detail::train_emeans(data.data(),rows,dimensions,clusters,seed+1,3.0)!=emeans,"e-means ignores the seed");
}
void restricted_test() {
  constexpr uint64_t seed=3;
  constexpr size_t supers=8,probes=3;
  const auto data=grouped_blobs(13,supers);
  const auto exact=tieredq_detail::train_emeans(data.data(),rows,dimensions,clusters,seed,3.0);
  const auto restricted=tieredq_detail::train_emeans(data.data(),rows,dimensions,clusters,seed,3.0,supers,probes);
  const auto [exact_objective,exact_empty]=objective(data,exact);
  const auto [restricted_objective,restricted_empty]=objective(data,restricted);
  require(restricted_empty==0,"restricted e-means leaves empty clusters");
  require(restricted_objective<=1.03*exact_objective,"restricted e-means misses exact e-means' objective");
}
// Training through a row source (as builds stream e-means batches) equals training on the same rows in RAM, and
// e-means reads every row it needs through the source.
void row_source_test() {
  const auto data=blobs(21);
  constexpr uint64_t seed=11;
  size_t reads=0;
  const tieredq_detail::TrainingRowSource source=[&](const uint32_t* positions,size_t n,float* out) {
    reads+=n;
    for (size_t i=0;i<n;++i) std::copy_n(data.data()+size_t(positions[i])*dimensions,dimensions,out+i*dimensions);
  };
  for (const double epochs:{0.0,2.5}) {
    const TieredQCoarseTraining training{4,8,3,64,epochs};
    reads=0;
    const auto streamed=tieredq_detail::train_coarse_rows(source,rows,dimensions,clusters,seed,training).centroids();
    const auto in_ram=tieredq_detail::train_coarse(data.data(),rows,dimensions,clusters,seed,training).centroids();
    require(streamed==in_ram,"training through a row source differs from training on the rows");
    require(reads>=rows,"training skipped rows of its source");
  }
}
void coarse_index_test() {
  const auto data=blobs(17);
  constexpr uint32_t ivf_lists=32;
  auto codec_config=TieredQConfig::target_code_bits(dimensions,2*dimensions);
  codec_config.metric=TieredQMetric::l2; codec_config.fine_tiers=false;
  TieredQIVFIndex index(TieredQ::train(codec_config,data.data(),rows,3),ivf_lists,5);
  index.set_coarse_training(TieredQCoarseTraining{4,8,3,64,2.5});
  index.build(data.data(),rows);
  const auto labels=index.list_assignment();
  std::vector<size_t> sizes(ivf_lists);
  for (const uint32_t l:labels) ++sizes[l];
  require(*std::min_element(sizes.begin(),sizes.end())>0,"e-means coarse lists are empty");
  bool rejected=false;
  try { TieredQCoarseTraining{4,0,16,64,-1}.validate(); } catch (const std::runtime_error&) { rejected=true; }
  require(rejected,"negative e-means epochs accepted");
}
void index_test() {
  const auto data=blobs(9);
  TieredQIVFFlatConfig config; config.lists=16; config.trainer=TieredQIVFFlatTrainer::emeans; config.epochs=2;
  config.sample_rows_per_list=128;
  TieredQIVFFlatIndex index(dimensions,config);
  index.build(data.data(),rows);
  require(index.lists()==16 && index.build_stats().train_s>0,"e-means lists");
  constexpr size_t k=5;
  std::vector<float> scores(rows*k); std::vector<uint32_t> ids(rows*k);
  index.knn_graph(k,16,scores.data(),ids.data());
  for (size_t r=0;r<rows;++r) require(ids[r*k]!=r && ids[r*k]<rows,"full-probe graph over e-means lists");
  config.epochs=0;
  bool rejected=false;
  try { TieredQIVFFlatIndex bad_index(dimensions,config); } catch (const std::runtime_error&) { rejected=true; }
  require(rejected,"zero epochs accepted");
}
} // namespace

void tieredq_emeans_tests() {
  recipe_test();
  quality_test();
  restricted_test();
  index_test();
  coarse_index_test();
  row_source_test();
  std::cout<<"PASS e-means recipe, quality, restricted assignment, row sources and IVF lists\n";
}
