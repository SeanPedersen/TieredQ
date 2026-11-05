// IVF over the codec's centroids or a coarse quantizer (beyond the codec's 256-centroid limit): a full probe returns
// the flat fast index's hits bit for bit in every fast layout and metric, partial probes stay inside the probed
// lists, unfilled slots are marked, and both persistence paths round-trip.
#include "tieredq/tieredq.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <unistd.h>
namespace {
using namespace dv;
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
constexpr uint32_t dimensions=48,centroids=8;
constexpr size_t rows=3001,queries=24,k=10;
// Rows around one fixed set of cluster centers; seed varies only the noise, so samples, data and queries agree.
// Noise decays over dimensions at `decay`; a steep spectrum spreads bits over more tiers.
float decay=1;
std::vector<float> clustered_rows(size_t count,uint32_t seed) {
  constexpr uint32_t center_seed=5;
  std::mt19937 center_random(center_seed),random(seed); std::normal_distribution<float> normal;
  std::vector<float> centers(centroids*dimensions);
  for (auto& value:centers) value=3.0f*normal(center_random);
  std::vector<float> data(count*dimensions);
  for (size_t i=0;i<count;++i) for (uint32_t j=0;j<dimensions;++j)
    data[i*dimensions+j]=centers[(i%centroids)*dimensions+j]+normal(random)*std::exp(-decay*float(j)/dimensions);
  return data;
}
TieredQ train(TieredQMetric metric,uint32_t code_bits) {
  const auto sample=clustered_rows(rows,3);
  auto config=TieredQConfig::target_code_bits(dimensions,code_bits);
  config.num_centroids=centroids; config.metric=metric; config.fine_tiers=false;
  return TieredQ::train(config,sample.data(),rows,7);
}
struct Results { std::vector<float> scores; std::vector<uint32_t> ids; };
Results run(const TieredQIVFIndex& index,const std::vector<float>& probes,size_t nprobe) {
  Results out{std::vector<float>(queries*k),std::vector<uint32_t>(queries*k)};
  index.search_batch(probes.data(),queries,k,nprobe,out.scores.data(),out.ids.data());
  return out;
}
void same(const Results& a,const Results& b,const char* message) {
  require(a.ids==b.ids,message);
  for (size_t i=0;i<a.scores.size();++i) require(a.scores[i]==b.scores[i] || (std::isinf(a.scores[i]) && std::isinf(b.scores[i])),message);
}
// Full probe equals the flat fast index; single and batched searches agree.
void full_probe_test(TieredQMetric metric,uint32_t code_bits,uint32_t lists=0) {
  const auto codec=train(metric,code_bits);
  const auto data=clustered_rows(rows,11),probes=clustered_rows(queries,13);
  TieredQIndex flat(codec,TieredQIndexMode::fast); flat.add(data.data(),rows);
  TieredQIVFIndex ivf(codec,lists); ivf.build(data.data(),rows);
  const size_t list_count=lists ? lists : centroids;
  require(ivf.size()==rows && ivf.lists()==list_count,"IVF shape");
  size_t listed=0; for (size_t list=0;list<ivf.lists();++list) listed+=ivf.list_size(list);
  require(listed==rows,"IVF lists do not partition the rows");
  const auto assignment=ivf.list_assignment();
  std::vector<size_t> counted(ivf.lists(),0);
  for (uint32_t list:assignment) ++counted[list];
  for (size_t list=0;list<ivf.lists();++list) require(counted[list]==ivf.list_size(list),"IVF list assignment differs from list sizes");
  Results expected{std::vector<float>(queries*k),std::vector<uint32_t>(queries*k)};
  flat.search_batch(probes.data(),queries,k,expected.scores.data(),expected.ids.data());
  const auto full=run(ivf,probes,list_count);
  same(full,expected,"full-probe IVF differs from the flat fast index");
  for (size_t q=0;q<queries;++q) {
    const auto hits=ivf.search(probes.data()+q*dimensions,k,list_count);
    require(hits.size()==k,"single IVF search size");
    for (size_t i=0;i<k;++i) require(hits[i].id==full.ids[q*k+i] && hits[i].score==full.scores[q*k+i],"single and batched IVF differ");
  }
  // Caller-prepared query weights scan the same lists; lookup layouts round their own projection differently.
  const auto weights=codec.query_weights(probes.data(),queries);
  Results prepared{std::vector<float>(queries*k),std::vector<uint32_t>(queries*k)};
  ivf.search_batch(probes.data(),weights.data(),queries,k,list_count,prepared.scores.data(),prepared.ids.data());
  size_t shared=0;
  for (size_t q=0;q<queries;++q) for (size_t i=0;i<k;++i)
    shared+=std::count(full.ids.begin()+q*k,full.ids.begin()+(q+1)*k,prepared.ids[q*k+i]);
  require(shared>=queries*k*9/10,"IVF search with prepared weights differs");
  const auto single=ivf.search(probes.data(),weights[0],k,list_count);
  for (size_t i=0;i<k;++i) require(single[i].id==prepared.ids[i] && single[i].score==prepared.scores[i],"single and batched prepared IVF differ");
  bool rejected=false;
  try { auto bad_weights=weights[0]; bad_weights.weights.pop_back(); ivf.search(probes.data(),bad_weights,k,list_count); }
  catch (const std::exception&) { rejected=true; }
  require(rejected,"IVF accepted mis-shaped query weights");
}
// Statistical pruning only drops documents, so every returned score is the unpruned score of that row; a huge
// confidence prunes nothing extra and matches the exact bound, and an invalid one is rejected.
void statistical_pruning_test() {
  const auto codec=train(TieredQMetric::inner_product,4*dimensions);
  const auto data=clustered_rows(rows,11),probes=clustered_rows(queries,13);
  TieredQIVFIndex ivf(codec); ivf.build(data.data(),rows);
  const auto exact=run(ivf,probes,centroids);
  for (uint32_t prefix:{1u,3u,4u}) for (double confidence:{0.5,4.0,1e30}) {
    Results pruned{std::vector<float>(queries*k),std::vector<uint32_t>(queries*k)};
    ivf.search_batch(probes.data(),queries,k,centroids,pruned.scores.data(),pruned.ids.data(),{confidence,prefix});
    std::vector<float> unpruned(rows,0);
    for (size_t q=0;q<queries;++q) {
      // Scores of every row for this query through the exact full probe (k = all rows).
      const auto all=ivf.search(probes.data()+q*dimensions,rows,centroids);
      for (const auto& hit:all) unpruned[hit.id]=hit.score;
      for (size_t i=0;i<k;++i) {
        const uint32_t id=pruned.ids[q*k+i];
        require(id!=UINT32_MAX && pruned.scores[q*k+i]==unpruned[id],"statistical pruning changed a score");
        require(i==0 || pruned.scores[q*k+i]<=pruned.scores[q*k+i-1],"statistical pruning broke the order");
      }
    }
    if (confidence>1e29 && prefix==4) same(pruned,exact,"unbounded confidence differs from the exact bound");
  }
  bool rejected=false;
  try { std::vector<float> s(k); std::vector<uint32_t> i(k); ivf.search_batch(probes.data(),1,k,1,s.data(),i.data(),{-1.0,4}); }
  catch (const std::exception&) { rejected=true; }
  require(rejected,"negative confidence accepted");
}
// One probed list: every hit lies in it, and slots beyond its rows are marked unfilled.
void partial_probe_test() {
  const auto codec=train(TieredQMetric::inner_product,2*dimensions);
  const auto data=clustered_rows(rows,17),probes=clustered_rows(queries,19);
  TieredQIVFIndex ivf(codec); ivf.build(data.data(),rows);
  TieredQIndex flat(codec,TieredQIndexMode::fast); flat.add(data.data(),rows);
  const auto encoded=codec.encode(data.data(),rows);
  auto list_of=[&](uint32_t id) { return encoded[size_t(id)*codec.bytes_per_vector()+codec.code_bytes()]; };
  const auto assignment=ivf.list_assignment();
  for (uint32_t id=0;id<rows;++id) require(assignment[id]==list_of(id),"codec-list assignment differs from centroid ids");
  const auto one=run(ivf,probes,1);
  for (size_t q=0;q<queries;++q) {
    std::set<uint32_t> lists;
    for (size_t i=0;i<k;++i) if (one.ids[q*k+i]!=UINT32_MAX) lists.insert(list_of(one.ids[q*k+i]));
    require(lists.size()==1,"one-list probe returned rows of several lists");
  }
  // Tiny index: more rows than k overall, fewer than k in the probed list.
  const size_t few=2*k;
  TieredQIVFIndex tiny(codec); tiny.build(data.data(),few);
  std::vector<float> scores(k); std::vector<uint32_t> ids(k);
  tiny.search_batch(probes.data(),1,k,1,scores.data(),ids.data());
  const size_t filled=size_t(std::count_if(ids.begin(),ids.end(),[](uint32_t id) { return id!=UINT32_MAX; }));
  require(filled>=1 && filled<k && std::isinf(scores[k-1]) && scores[k-1]<0,"unfilled slots are not marked");
  require(tiny.search(probes.data(),k,1).size()==filled,"single search returns marked slots");
}
void persistence_test(uint32_t lists) {
  const auto codec=train(TieredQMetric::cosine,dimensions);
  const auto data=clustered_rows(rows,23),probes=clustered_rows(queries,29);
  TieredQIVFIndex ivf(codec,lists); ivf.build(data.data(),rows);
  const auto expected=run(ivf,probes,3);
  same(run(TieredQIVFIndex::deserialize(ivf.serialize()),probes,3),expected,"IVF bytes do not round-trip");
  const auto path=(std::filesystem::temp_directory_path()/("tieredq-ivf-"+std::to_string(::getpid())+".tq")).string();
  ivf.save(path);
  const auto loaded=TieredQIVFIndex::load(path);
  std::filesystem::remove(path);
  same(run(loaded,probes,3),expected,"IVF file does not round-trip");
  require(loaded.storage_bytes()>0 && loaded.size()==rows,"loaded IVF shape");
  TieredQIVFIndex empty(codec,lists);
  require(TieredQIVFIndex::deserialize(empty.serialize()).size()==0,"empty IVF does not round-trip");
}
// train_lists fixes the partition: builds of other data assign to the same lists, a saved trained index keeps them,
// and indexes without coarse lists reject it.
void fixed_lists_test(TieredQMetric metric) {
  constexpr uint32_t lists=20;
  const auto codec=train(metric,2*dimensions);
  const auto sample=clustered_rows(rows,31),first=clustered_rows(rows,37),second=clustered_rows(rows,41);
  TieredQIVFIndex fixed(codec,lists); fixed.train_lists(sample.data(),rows);
  const auto trained=TieredQIVFIndex::deserialize(fixed.serialize());
  fixed.build(first.data(),rows); fixed.build(second.data(),rows);
  TieredQIVFIndex fresh(codec,lists); fresh.train_lists(sample.data(),rows); fresh.build(second.data(),rows);
  require(fixed.list_assignment()==fresh.list_assignment(),"fixed lists change between builds");
  auto reloaded=trained; reloaded.build(second.data(),rows);
  require(reloaded.list_assignment()==fresh.list_assignment(),"saved trained lists are not kept");
  TieredQIVFIndex learned(codec,lists); learned.build(second.data(),rows);
  require(learned.list_assignment()!=fresh.list_assignment(),"lists trained on the build rows equal the fixed ones");
  TieredQIndex flat(codec,TieredQIndexMode::fast); flat.add(second.data(),rows);
  const auto probes=clustered_rows(queries,43);
  Results expected{std::vector<float>(queries*k),std::vector<uint32_t>(queries*k)};
  flat.search_batch(probes.data(),queries,k,expected.scores.data(),expected.ids.data());
  same(run(fresh,probes,lists),expected,"full probe over fixed lists differs from the flat index");
  bool rejected=false;
  try { TieredQIVFIndex plain(codec); plain.train_lists(sample.data(),rows); } catch (const std::runtime_error&) { rejected=true; }
  require(rejected,"train_lists without coarse lists accepted");
  rejected=false;
  try { TieredQIVFIndex few(codec,lists); few.train_lists(sample.data(),lists-1); } catch (const std::runtime_error&) { rejected=true; }
  require(rejected,"train_lists with fewer rows than lists accepted");
}
// Builds of more documents than one build chunk still equal the flat index under a full probe.
void multi_chunk_build_test(uint32_t lists) {
  constexpr size_t many=20000;
  const auto codec=train(TieredQMetric::inner_product,2*dimensions);
  const auto data=clustered_rows(many,47),probes=clustered_rows(queries,53);
  TieredQIndex flat(codec,TieredQIndexMode::fast); flat.add(data.data(),many);
  Results expected{std::vector<float>(queries*k),std::vector<uint32_t>(queries*k)};
  flat.search_batch(probes.data(),queries,k,expected.scores.data(),expected.ids.data());
  TieredQIVFIndex ivf(codec,lists); ivf.build(data.data(),many);
  same(run(ivf,probes,ivf.lists()),expected,"multi-chunk IVF build differs from the flat index");
}
} // namespace
void tieredq_ivf_tests() {
  // 1 bit/dim takes the nibble lookup layout on x86, 4 bits int8 levels or codes, 6 bits wide code lookups.
  for (uint32_t bits:{1u,2u,4u,6u})
    for (auto metric:{TieredQMetric::inner_product,TieredQMetric::cosine,TieredQMetric::l2}) full_probe_test(metric,bits*dimensions);
  // Coarse lists: more than the codec's centroids, and more than one byte of list ids.
  for (uint32_t lists:{20u,300u})
    for (auto metric:{TieredQMetric::inner_product,TieredQMetric::l2}) full_probe_test(metric,2*dimensions,lists);
  {
    // More than four tiers take the AVX-512 range checkpoint, which prunes across lists with the running threshold;
    // the full probe must still equal the flat index's unpruned batch scan.
    constexpr float steep=8;
    decay=steep;
    require(train(TieredQMetric::inner_product,4*dimensions).tiers().size()>4,"steep model lacks checkpoint tiers");
    for (auto metric:{TieredQMetric::inner_product,TieredQMetric::l2}) full_probe_test(metric,4*dimensions);
    full_probe_test(TieredQMetric::inner_product,4*dimensions,20);
    statistical_pruning_test();
    decay=1;
  }
  partial_probe_test();
  persistence_test(0);
  persistence_test(20);
  for (auto metric:{TieredQMetric::inner_product,TieredQMetric::l2}) fixed_lists_test(metric);
  for (uint32_t lists:{0u,20u}) multi_chunk_build_test(lists);
  std::cout<<"PASS IVF\n";
}
