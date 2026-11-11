// Nested refinement (#23): prefix plus suffix ranks like a flat model of the total budget and clearly better than
// the prefix alone, in every metric and for prefixes below and at 1 bit/dim; serialization round-trips bit for bit;
// mismatched training inputs and prefixes are rejected.
#include "tieredq/tieredq_refinement.h"
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
constexpr uint32_t dimensions=48,centroids=8,seed=7;
constexpr size_t rows=3001,queries=32,k=10;
// Clustered rows whose noise decays over dimensions, so models spread bits over several tiers.
std::vector<float> clustered_rows(size_t count,uint32_t noise_seed) {
  constexpr uint32_t center_seed=5;
  constexpr float decay=4;
  std::mt19937 center_random(center_seed),random(noise_seed); std::normal_distribution<float> normal;
  std::vector<float> centers(centroids*dimensions);
  for (auto& value:centers) value=3.0f*normal(center_random);
  std::vector<float> data(count*dimensions);
  for (size_t i=0;i<count;++i) for (uint32_t j=0;j<dimensions;++j)
    data[i*dimensions+j]=centers[(i%centroids)*dimensions+j]+normal(random)*std::exp(-decay*float(j)/dimensions);
  return data;
}
TieredQ train(const std::vector<float>& sample,TieredQMetric metric,uint32_t code_bits,uint64_t model_seed=seed) {
  auto config=TieredQConfig::target_code_bits(dimensions,code_bits);
  config.num_centroids=centroids; config.metric=metric; config.fine_tiers=false;
  return TieredQ::train(config,sample.data(),rows,model_seed);
}
float exact(TieredQMetric metric,const float* q,const float* x) {
  double dot=0,qq=0,xx=0,distance=0;
  for (uint32_t j=0;j<dimensions;++j) {
    dot+=double(q[j])*x[j]; qq+=double(q[j])*q[j]; xx+=double(x[j])*x[j]; distance+=(double(q[j])-x[j])*(q[j]-x[j]);
  }
  if (metric==TieredQMetric::cosine) return float(dot/std::sqrt(qq*xx));
  return float(metric==TieredQMetric::l2 ? -distance : dot);
}
// Mean recall@k over queries of score lists (rows per query) against the metric's exact ranking.
double recall(TieredQMetric metric,const std::vector<float>& data,const std::vector<float>& probes,
              const std::vector<std::vector<float>>& scores) {
  auto top=[](const std::vector<float>& s) {
    std::vector<uint32_t> ids(s.size()); std::iota(ids.begin(),ids.end(),0);
    std::partial_sort(ids.begin(),ids.begin()+k,ids.end(),[&](uint32_t a,uint32_t b) { return s[a]>s[b]; });
    ids.resize(k); return ids;
  };
  size_t hits=0;
  for (size_t q=0;q<queries;++q) {
    std::vector<float> truth(rows);
    for (size_t i=0;i<rows;++i) truth[i]=exact(metric,probes.data()+q*dimensions,data.data()+i*dimensions);
    const auto expected=top(truth),found=top(scores[q]);
    for (uint32_t id:found) hits+=std::count(expected.begin(),expected.end(),id);
  }
  return double(hits)/double(queries*k);
}
std::vector<std::vector<float>> flat_scores(const TieredQ& model,const std::vector<float>& data,const std::vector<float>& probes) {
  const auto codes=model.encode(data.data(),rows);
  std::vector<std::vector<float>> out(queries,std::vector<float>(rows));
  for (size_t q=0;q<queries;++q) model.score(codes.data(),rows,model.query(probes.data()+q*dimensions),out[q].data());
  return out;
}
std::vector<std::vector<float>> refined_scores(const TieredQRefinement& refinement,const std::vector<uint8_t>& prefix_codes,
                                               const std::vector<uint8_t>& suffix_codes,const std::vector<float>& probes,
                                               TieredQKernel kernel=TieredQKernel::automatic) {
  std::vector<std::vector<float>> out(queries,std::vector<float>(rows));
  for (size_t q=0;q<queries;++q)
    refinement.score(prefix_codes.data(),suffix_codes.data(),rows,refinement.query(probes.data()+q*dimensions),out[q].data(),kernel);
  return out;
}
// The AVX-512 kernel (where the host has it) sums the scalar kernel's terms in another order; a query built from the
// prefix's own query weights scores exactly as one that projects them itself.
void kernel_parity(const TieredQRefinement& refinement,const std::vector<uint8_t>& prefix_codes,
                   const std::vector<uint8_t>& suffix_codes,const std::vector<float>& probes) {
  const auto scalar=refined_scores(refinement,prefix_codes,suffix_codes,probes,TieredQKernel::scalar);
  const auto automatic=refined_scores(refinement,prefix_codes,suffix_codes,probes);
  constexpr float tolerance=1e-4f;
  for (size_t q=0;q<queries;++q) {
    float scale=0;
    for (float value:scalar[q]) scale=std::max(scale,std::fabs(value));
    for (size_t i=0;i<rows;++i) require(std::fabs(scalar[q][i]-automatic[q][i])<=tolerance*scale,"refinement kernels disagree");
  }
  const float* probe=probes.data();
  const auto shared=refinement.query(probe,refinement.prefix().query_weights(probe)),own=refinement.query(probe);
  std::vector<float> a(rows),b(rows);
  refinement.score(prefix_codes.data(),suffix_codes.data(),rows,shared,a.data());
  refinement.score(prefix_codes.data(),suffix_codes.data(),rows,own,b.data());
  require(a==b,"refinement scores differently from shared prefix query weights");
  // Batched queries (full and partial tiles) equal per-row ones exactly.
  constexpr size_t ragged_rows=13;
  for (size_t count:{queries,ragged_rows}) {
    const auto batch=refinement.query(probes.data(),count,refinement.prefix().query_weights(probes.data(),count));
    require(batch.size()==count,"refinement batch size");
    for (size_t q=0;q<count;++q) {
      const auto row=refinement.query(probes.data()+q*dimensions);
      require(batch[q].weights==row.weights && batch[q].extra_mean==row.extra_mean &&
              batch[q].prefix.centroid_scores==row.prefix.centroid_scores,"batched refinement query differs from per-row");
    }
  }
}
// Batched queries across several suffix basis blocks (16 panels of 16 ranks each) equal per-row ones exactly.
void wide_batch_test() {
  constexpr uint32_t wide=640,wide_seed=17;
  constexpr size_t wide_rows=2000,batch_rows=21,block_ranks=256;
  constexpr float decay=1;
  std::mt19937 random(wide_seed); std::normal_distribution<float> normal;
  std::vector<float> data(wide_rows*wide);
  for (size_t i=0;i<wide_rows;++i) for (uint32_t j=0;j<wide;++j) data[i*wide+j]=normal(random)*std::exp(-decay*float(j)/wide);
  auto config=TieredQConfig::target_code_bits(wide,wide/4);
  config.num_centroids=centroids; config.fine_tiers=false;
  const auto prefix=TieredQ::train(config,data.data(),wide_rows,seed);
  const auto refinement=TieredQRefinement::train(prefix,4*wide,data.data(),wide_rows,seed);
  require(refinement.positions()-prefix.coded_dimensions()>block_ranks,"wide refinement fits one basis block");
  const auto batch=refinement.query(data.data(),batch_rows,prefix.query_weights(data.data(),batch_rows));
  for (size_t q=0;q<batch_rows;++q) {
    const auto row=refinement.query(data.data()+q*wide);
    require(batch[q].weights==row.weights && batch[q].extra_mean==row.extra_mean,"wide batched refinement query differs");
  }
  // A batch needs one prefix query per row.
  bool rejected=false;
  try { refinement.query(data.data(),batch_rows,prefix.query_weights(data.data(),batch_rows-1)); }
  catch (const std::exception&) { rejected=true; }
  require(rejected,"refinement batch accepted a missing prefix query");
}
void quality_test(TieredQMetric metric,uint32_t prefix_bits,uint32_t total_bits) {
  const auto sample=clustered_rows(rows,3),data=clustered_rows(rows,11),probes=clustered_rows(queries,13);
  const auto prefix=train(sample,metric,prefix_bits),flat=train(sample,metric,total_bits);
  const auto refinement=TieredQRefinement::train(prefix,total_bits,sample.data(),rows,seed);
  require(refinement.suffix_bits()>0 && refinement.bytes_per_vector()==refinement.suffix_code_bytes()+4,"suffix shape");
  require(refinement.positions()>=prefix.coded_dimensions(),"refinement drops prefix ranks");
  const auto prefix_codes=prefix.encode(data.data(),rows);
  std::vector<uint8_t> suffix_codes(rows*refinement.bytes_per_vector());
  refinement.encode(data.data(),rows,prefix_codes.data(),suffix_codes.data());
  kernel_parity(refinement,prefix_codes,suffix_codes,probes);
  const double alone=recall(metric,data,probes,flat_scores(prefix,data,probes));
  const double nested=recall(metric,data,probes,refined_scores(refinement,prefix_codes,suffix_codes,probes));
  const double full=recall(metric,data,probes,flat_scores(flat,data,probes));
  std::cout<<"refinement metric "<<int(metric)<<" "<<prefix_bits<<"->"<<total_bits<<": prefix "<<alone<<" nested "
           <<nested<<" flat "<<full<<" suffix bits "<<refinement.suffix_bits()<<"\n";
  constexpr double kept_misses=0.5,flat_margin=0.03;
  require(1-nested<=kept_misses*(1-alone),"refinement does not halve its prefix's misses");
  require(nested>=full-flat_margin,"refinement ranks clearly worse than the flat model of its budget");
}
// Wide suffix fields cross byte boundaries; restored encoders and every available scoring kernel agree.
void fine_suffix_test() {
  const auto sample=clustered_rows(rows,3),data=clustered_rows(rows,11),probes=clustered_rows(queries,13);
  const auto prefix=train(sample,TieredQMetric::inner_product,2*dimensions);
  const auto baseline=TieredQRefinement::train(prefix,8*dimensions,sample.data(),rows,seed);
  require(baseline.max_code_bits()<=8,"default refinement changed its width cap");
  for (uint32_t cap:{10u,12u}) {
    const auto refinement=TieredQRefinement::train(prefix,8*dimensions,sample.data(),rows,seed,true,cap);
    require(refinement.max_code_bits()>8 && refinement.max_code_bits()<=cap,"fine refinement did not go wide");
    const auto bytes=refinement.serialize();
    const auto restored=TieredQRefinement::deserialize(bytes,prefix);
    require(restored.serialize()==bytes,"wide refinement persistence changed");
    const auto prefix_codes=prefix.encode(data.data(),rows);
    std::vector<uint8_t> codes(rows*refinement.bytes_per_vector()),again(codes.size());
    refinement.encode(data.data(),rows,prefix_codes.data(),codes.data());
    restored.encode(data.data(),rows,prefix_codes.data(),again.data());
    require(codes==again,"wide refinement persistence changed encoding");
    kernel_parity(refinement,prefix_codes,codes,probes);
    require(refined_scores(refinement,prefix_codes,codes,probes)==refined_scores(restored,prefix_codes,codes,probes),
            "wide refinement persistence changed scores");
    auto old_format=bytes;
    old_format[4]=1;
    bool rejected=false;
    try { TieredQRefinement::deserialize(old_format,prefix); } catch (const std::exception&) { rejected=true; }
    require(rejected,"legacy refinement format accepted wide codes");
  }
  // A tiny prefix forces suffix fields themselves past eight bits, including three-byte packed fields.
  auto tiny_config=TieredQConfig::target_code_bits(dimensions,1);
  tiny_config.num_centroids=centroids; tiny_config.fine_tiers=false;
  tiny_config.target_mode=TieredQIndexMode::full;
  const auto tiny=TieredQ::train(tiny_config,sample.data(),rows,seed);
  const auto wide=TieredQRefinement::train(tiny,8*dimensions,sample.data(),rows,seed,true);
  uint32_t prefix_width=0;
  for (const auto& tier:tiny.tiers()) prefix_width=std::max(prefix_width,tier.bits);
  require(wide.max_code_bits()>prefix_width+8,"wide suffix field regression did not exercise wide fields");
  const auto prefix_codes=tiny.encode(data.data(),rows);
  std::vector<uint8_t> codes(rows*wide.bytes_per_vector());
  wide.encode(data.data(),rows,prefix_codes.data(),codes.data());
  kernel_parity(wide,prefix_codes,codes,probes);
}
void persistence_test() {
  const auto sample=clustered_rows(rows,3),data=clustered_rows(rows,11),probes=clustered_rows(queries,13);
  const auto prefix=train(sample,TieredQMetric::inner_product,dimensions);
  const auto refinement=TieredQRefinement::train(prefix,4*dimensions,sample.data(),rows,seed);
  const auto restored=TieredQRefinement::deserialize(refinement.serialize(),TieredQ::deserialize(prefix.serialize()));
  require(restored.serialize()==refinement.serialize(),"refinement serialization does not round-trip");
  const auto prefix_codes=prefix.encode(data.data(),rows);
  std::vector<uint8_t> a(rows*refinement.bytes_per_vector()),b(a.size());
  refinement.encode(data.data(),rows,prefix_codes.data(),a.data());
  restored.encode(data.data(),rows,prefix_codes.data(),b.data());
  require(a==b,"restored refinement encodes differently");
  require(refined_scores(refinement,prefix_codes,a,probes)==refined_scores(restored,prefix_codes,a,probes),
          "restored refinement scores differently");
  // A different prefix, a budget no larger than the prefix, and other training inputs are rejected.
  const auto other=train(sample,TieredQMetric::inner_product,2*dimensions);
  bool rejected=false;
  try { TieredQRefinement::deserialize(refinement.serialize(),other); } catch (const std::exception&) { rejected=true; }
  require(rejected,"refinement accepted a different prefix");
  rejected=false;
  try { TieredQRefinement::train(prefix,prefix.config().code_bits,sample.data(),rows,seed); } catch (const std::exception&) { rejected=true; }
  require(rejected,"refinement accepted a budget no larger than its prefix");
  rejected=false;
  try { TieredQRefinement::train(prefix,4*dimensions,sample.data(),rows,seed+1); } catch (const std::exception&) { rejected=true; }
  require(rejected,"refinement accepted a different training seed");
}
} // namespace

void tieredq_refinement_tests() {
  for (auto metric:{TieredQMetric::inner_product,TieredQMetric::cosine,TieredQMetric::l2}) {
    quality_test(metric,dimensions,4*dimensions);
    quality_test(metric,dimensions/2,3*dimensions);
  }
  fine_suffix_test();
  persistence_test();
  wide_batch_test();
  std::cout<<"PASS refinement\n";
}
