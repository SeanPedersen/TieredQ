// Native metrics: L2 scores estimate negative squared distances and rank like exact L2, cosine is scale invariant,
// every search path and persistence agree, decode inverts the lift, and the allocation power follows the metric.
#include "tieredq/tieredq.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
namespace {
using namespace dv;
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
constexpr uint32_t dimensions=24,centroids=8;
constexpr size_t rows=1500,queries=20,k=10,min_overlap=8;
// Offset, anisotropic rows with varied norms, so L2, inner-product and cosine rankings differ.
std::vector<float> rows_with_norms(size_t count,uint32_t seed) {
  std::mt19937 random(seed); std::normal_distribution<float> normal; std::uniform_real_distribution<float> norm(0.2f,3.0f);
  std::vector<float> data(count*dimensions);
  for (size_t i=0;i<count;++i) {
    const float scale=norm(random);
    for (uint32_t j=0;j<dimensions;++j) data[i*dimensions+j]=2.0f+scale*normal(random)*std::exp(-float(j)/dimensions);
  }
  return data;
}
TieredQ train(TieredQMetric metric,uint32_t bits_per_dim,TieredQIndexMode mode,double power=0) {
  const auto sample=rows_with_norms(rows,3);
  auto config=TieredQConfig::target_code_bits(dimensions,dimensions*bits_per_dim);
  config.num_centroids=centroids; config.metric=metric; config.target_mode=mode; config.allocation_power=power;
  return TieredQ::train(config,sample.data(),rows,7);
}
double similarity(TieredQMetric metric,const float* q,const float* x) {
  double dot=0,qq=0,xx=0,distance=0;
  for (uint32_t j=0;j<dimensions;++j) { dot+=double(q[j])*x[j]; qq+=double(q[j])*q[j]; xx+=double(x[j])*x[j]; distance+=(double(q[j])-x[j])*(double(q[j])-x[j]); }
  if (metric==TieredQMetric::l2) return -distance;
  if (metric==TieredQMetric::cosine) return dot/std::sqrt(qq*xx);
  return dot;
}
std::vector<uint32_t> exact_top(TieredQMetric metric,const std::vector<float>& data,const float* q) {
  std::vector<uint32_t> ids(data.size()/dimensions); std::iota(ids.begin(),ids.end(),0);
  std::partial_sort(ids.begin(),ids.begin()+k,ids.end(),[&](uint32_t a,uint32_t b) {
    return similarity(metric,q,data.data()+a*dimensions)>similarity(metric,q,data.data()+b*dimensions); });
  ids.resize(k); return ids;
}
size_t overlap(const std::vector<uint32_t>& exact,const std::vector<TieredQHit>& hits) {
  size_t found=0; for (const auto& hit:hits) found+=std::count(exact.begin(),exact.end(),hit.id); return found;
}
void same_hits(const std::vector<TieredQHit>& a,const std::vector<TieredQHit>& b,const char* message) {
  require(a.size()==b.size(),message);
  for (size_t i=0;i<a.size();++i) require(a[i].id==b[i].id && a[i].score==b[i].score,message);
}
// Every search path of one index returns the same hits; batch rows match single searches.
void path_parity(const TieredQIndex& index,const std::vector<float>& probes) {
  std::vector<float> scores(queries*k),fast_scores(queries*k); std::vector<uint32_t> ids(queries*k),fast_ids(queries*k);
  index.search_batch(probes.data(),queries,k,scores.data(),ids.data());
  index.search_batch_fast(probes.data(),queries,k,fast_scores.data(),fast_ids.data());
  for (size_t q=0;q<queries;++q) {
    const auto single=index.search(probes.data()+q*dimensions,k);
    same_hits(single,index.search_fast(probes.data()+q*dimensions,k),"search and search_fast differ");
    for (size_t i=0;i<k;++i) {
      require(ids[q*k+i]==single[i].id && scores[q*k+i]==single[i].score,"batch search differs from search");
      require(fast_ids[q*k+i]==single[i].id && fast_scores[q*k+i]==single[i].score,"fast batch search differs from search");
    }
  }
}
// L2 scores track -|q - x|^2 and rank like exact L2 in both modes; scores survive save/load.
void l2_test() {
  const auto data=rows_with_norms(rows,11),probes=rows_with_norms(queries,13);
  for (auto mode:{TieredQIndexMode::fast,TieredQIndexMode::full}) {
    const auto codec=train(TieredQMetric::l2,4,mode);
    require(codec.model_dimensions()==dimensions+1 && codec.config().dimensions==dimensions,"L2 model dimensions");
    TieredQIndex index(codec,mode); index.add(data.data(),rows);
    size_t found=0; double error=0,scale=0;
    for (size_t q=0;q<queries;++q) {
      const float* probe=probes.data()+q*dimensions;
      const auto hits=index.search(probe,k);
      found+=overlap(exact_top(TieredQMetric::l2,data,probe),hits);
      for (const auto& hit:hits) {
        const double exact=similarity(TieredQMetric::l2,probe,data.data()+hit.id*dimensions);
        error+=std::fabs(hit.score-exact); scale+=std::fabs(exact);
      }
    }
    require(found>=min_overlap*queries,"L2 search disagrees with exact L2");
    require(error<0.05*scale,"L2 scores do not estimate negative squared distances");
    path_parity(index,probes);
    const auto path=std::filesystem::temp_directory_path()/("tieredq-l2-"+std::to_string(getpid()));
    index.save(path.string());
    const auto loaded=TieredQIndex::load(path.string());
    std::filesystem::remove(path);
    require(loaded.codec().config().metric==TieredQMetric::l2 && loaded.codec().model_id()==codec.model_id(),"reload changes the L2 model");
    for (size_t q=0;q<queries;++q) same_hits(loaded.search(probes.data()+q*dimensions,k),index.search(probes.data()+q*dimensions,k),"reload changes L2 scores");
  }
}
// Decoding inverts the lift; lifting queries once matches query().
void l2_codec_test() {
  const auto codec=TieredQ::deserialize(train(TieredQMetric::l2,8,TieredQIndexMode::full).serialize());
  const auto data=rows_with_norms(64,17);
  std::vector<float> decoded(data.size());
  codec.decode(codec.encode(data.data(),64).data(),64,decoded.data());
  double error=0,energy=0;
  for (size_t i=0;i<data.size();++i) { error+=(decoded[i]-data[i])*(decoded[i]-data[i]); energy+=data[i]*data[i]; }
  require(error<1e-3*energy,"decode does not invert the L2 lift");
  std::vector<float> lifted(codec.model_dimensions());
  codec.lift_queries(data.data(),1,lifted.data());
  const auto direct=codec.query(data.data()),model=codec.model_query(lifted.data());
  require(direct.lut==model.lut && direct.centroid_scores==model.centroid_scores,"lifted query differs from query()");
}
// Cosine ranks like exact cosine and ignores row and query norms.
void cosine_test() {
  const auto data=rows_with_norms(rows,19),probes=rows_with_norms(queries,23);
  const auto codec=train(TieredQMetric::cosine,4,TieredQIndexMode::fast);
  TieredQIndex index(codec),scaled_index(codec);
  auto scaled=data; for (auto& v:scaled) v*=4.0f; // powers of two scale exactly, so normalized rows are identical
  index.add(data.data(),rows); scaled_index.add(scaled.data(),rows);
  size_t found=0;
  for (size_t q=0;q<queries;++q) {
    const float* probe=probes.data()+q*dimensions;
    const auto hits=index.search(probe,k);
    found+=overlap(exact_top(TieredQMetric::cosine,data,probe),hits);
    std::vector<float> doubled(probe,probe+dimensions); for (auto& v:doubled) v*=2;
    same_hits(scaled_index.search(doubled.data(),k),hits,"cosine depends on vector norms");
    for (const auto& hit:hits) require(hit.score<1.05f,"cosine score exceeds one");
  }
  require(found>=min_overlap*queries,"cosine search disagrees with exact cosine");
  path_parity(index,probes);
  require(TieredQ::deserialize(codec.serialize()).model_id()==codec.model_id(),"cosine model does not round-trip");
}
// train_sampled trains on the named rows in place: the same model as train() on those rows gathered, in every metric.
void sampled_training_test() {
  const auto data=rows_with_norms(rows,31);
  std::vector<uint32_t> ids;
  for (uint32_t i=0;i<rows;i+=2) ids.push_back(uint32_t(rows-1-i));
  std::vector<float> gathered;
  for (uint32_t id:ids) gathered.insert(gathered.end(),data.begin()+size_t(id)*dimensions,data.begin()+size_t(id+1)*dimensions);
  for (auto metric:{TieredQMetric::inner_product,TieredQMetric::cosine,TieredQMetric::l2}) {
    auto config=TieredQConfig::target_code_bits(dimensions,dimensions*2);
    config.num_centroids=centroids; config.metric=metric;
    const auto expected=TieredQ::train(config,gathered.data(),ids.size(),7);
    const auto sampled=TieredQ::train_sampled(config,data.data(),rows,ids.data(),ids.size(),7);
    require(sampled.model_id()==expected.model_id() && sampled.encode(data.data(),rows)==expected.encode(data.data(),rows),
            "sampled training differs from training on the gathered rows");
  }
  auto invalid=ids; invalid.back()=uint32_t(rows);
  bool rejected=false;
  try { TieredQ::train_sampled(TieredQConfig::target_code_bits(dimensions,dimensions),data.data(),rows,invalid.data(),invalid.size()); }
  catch (const std::exception&) { rejected=true; }
  require(rejected,"out-of-range training sample id accepted");
}
// Cosine training reads normalized rows on access; it equals inner-product training on rows normalized beforehand.
void cosine_training_test() {
  const auto sample=rows_with_norms(rows,3),data=rows_with_norms(rows,29);
  auto normalized=[](const std::vector<float>& input) {
    std::vector<float> out(input.size());
    for (size_t i=0;i<input.size()/dimensions;++i) {
      double square=0; for (uint32_t j=0;j<dimensions;++j) square+=double(input[i*dimensions+j])*input[i*dimensions+j];
      const double norm=std::sqrt(square);
      for (uint32_t j=0;j<dimensions;++j) out[i*dimensions+j]=float(input[i*dimensions+j]/norm);
    }
    return out;
  };
  const auto unit_sample=normalized(sample);
  auto config=TieredQConfig::target_code_bits(dimensions,dimensions*2);
  config.num_centroids=centroids;
  const auto inner=TieredQ::train(config,unit_sample.data(),rows,7);
  config.metric=TieredQMetric::cosine;
  const auto cosine=TieredQ::train(config,sample.data(),rows,7);
  require(cosine.encode(data.data(),rows)==inner.encode(normalized(data).data(),rows),
          "cosine training differs from inner-product training on normalized rows");
}
// The default power follows the metric in both modes (1 for L2, 1.5 otherwise from 1 bit per dimension up, lower
// below it); inner-product models keep format 3.
void power_and_format_test() {
  require(train(TieredQMetric::l2,2,TieredQIndexMode::fast).model_id()==train(TieredQMetric::l2,2,TieredQIndexMode::fast,1.0).model_id(),
          "L2 fast default is not power 1");
  require(train(TieredQMetric::inner_product,2,TieredQIndexMode::fast).model_id()==
          train(TieredQMetric::inner_product,2,TieredQIndexMode::fast,1.5).model_id(),"inner-product fast default is not power 1.5");
  require(train(TieredQMetric::inner_product,2,TieredQIndexMode::full).model_id()==
          train(TieredQMetric::inner_product,2,TieredQIndexMode::full,1.5).model_id(),"full inner-product default is not power 1.5");
  require(train(TieredQMetric::l2,2,TieredQIndexMode::full).model_id()==
          train(TieredQMetric::l2,2,TieredQIndexMode::full,1.0).model_id(),"full L2 default is not power 1");
  // Below 1 bit per dimension the inner-product default falls to 1 at 0.5 bits per dimension (1.25 at 0.75).
  auto sub_bit=[](uint32_t code_bits,double power) {
    const auto sample=rows_with_norms(rows,3);
    auto config=TieredQConfig::target_code_bits(dimensions,code_bits);
    config.num_centroids=centroids; config.allocation_power=power;
    return TieredQ::train(config,sample.data(),rows,7).model_id();
  };
  require(sub_bit(dimensions/4,0)==sub_bit(dimensions/4,1.0) && sub_bit(dimensions/2,0)==sub_bit(dimensions/2,1.0),
          "inner-product default is not power 1 at 0.5 bits per dimension or below");
  require(sub_bit(3*dimensions/4,0)==sub_bit(3*dimensions/4,1.25),"inner-product default is not power 1.25 at 0.75 bits");
  auto format=[](const TieredQ& codec) { const auto data=codec.serialize(); uint32_t v; std::memcpy(&v,data.data()+4*sizeof(uint32_t),sizeof v); return v; };
  require(format(train(TieredQMetric::inner_product,2,TieredQIndexMode::fast))==TieredQ::model_format,"inner-product model format changed");
  require(format(train(TieredQMetric::l2,2,TieredQIndexMode::fast))==TieredQ::metric_model_format,"L2 model format");
  for (double power:{-1.0,double(NAN)}) {
    auto config=TieredQConfig::target_code_bits(dimensions,dimensions); config.allocation_power=power;
    bool rejected=false; try { config.validate(); } catch (const std::runtime_error&) { rejected=true; }
    require(rejected,"invalid allocation power accepted");
  }
}
} // namespace
void tieredq_metric_tests() {
  l2_test();
  l2_codec_test();
  cosine_test();
  cosine_training_test();
  sampled_training_test();
  power_and_format_test();
  std::cout<<"PASS native metrics\n";
}
