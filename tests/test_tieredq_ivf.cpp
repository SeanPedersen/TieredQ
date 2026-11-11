// IVF over the codec's centroids or a coarse quantizer (beyond the codec's 256-centroid limit): a full probe returns
// the flat fast index's hits bit for bit in every fast layout and metric, partial probes stay inside the probed
// lists, unfilled slots are marked, and both persistence paths round-trip. Ids are Elias-Fano lists; version 2 files
// with raw ids still load. Lists are packed (version 6); padded lists of earlier versions load and keep their hits.
#include "tieredq/tieredq.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>
namespace {
using namespace tq;
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> bool rejects(F&& action) { try { action(); } catch (const std::runtime_error&) { return true; } return false; }
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
// Scores match exactly; ids match except among rows whose score ties a neighbour's or the k-th score (a tie with a
// row ranked beyond k), where scan order decides.
void same_up_to_ties(const Results& a,const Results& b,const char* message) {
  for (size_t q=0;q<queries;++q) {
    const auto score=[&](const Results& r,size_t i) { return r.scores[q*k+i]; };
    for (size_t i=0;i<k;++i) {
      const bool both_unfilled=std::isinf(score(a,i)) && std::isinf(score(b,i));
      require(score(a,i)==score(b,i) || both_unfilled,(std::string(message)+" (scores)").c_str());
      const bool tied=(i>0 && score(a,i)==score(a,i-1)) || (i+1<k && score(a,i)==score(a,i+1)) || score(a,i)==score(a,k-1);
      require(tied || a.ids[q*k+i]==b.ids[q*k+i],(std::string(message)+" (ids)").c_str());
    }
  }
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
  require(loaded.storage_bytes()>loaded.codec().memory_bytes() && loaded.size()==rows,"loaded IVF shape");
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
  same_up_to_ties(run(fresh,probes,lists),expected,"full probe over fixed lists differs from the flat index");
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
// Probing ranks lists by the rule that assigned the rows: a stored row used as the query, probing one list, scans its
// own list. L2 coarse lists live in input space; L2 codec lists come from the encoder's model-space centroids and are
// not covered.
void probe_order_test(TieredQMetric metric,uint32_t lists) {
  const auto codec=train(metric,4*dimensions);
  const auto data=clustered_rows(rows,61);
  TieredQIVFIndex ivf(codec,lists); ivf.build(data.data(),rows);
  const auto assignment=ivf.list_assignment();
  constexpr size_t probes=200;
  size_t own=0;
  for (size_t row=0;row<probes;++row) {
    const auto hits=ivf.search(data.data()+row*dimensions,k,1);
    require(!hits.empty(),"single-list probe found nothing");
    own+=assignment[hits.front().id]==assignment[row];
  }
  if (own!=probes) throw std::runtime_error("a stored row does not probe its own list first (metric "+
      std::to_string(int(metric))+", lists "+std::to_string(lists)+": "+std::to_string(own)+"/"+std::to_string(probes)+")");
}
// L2 coarse lists are plain L2 k-means on the input vectors: every row sits in the list of its nearest input-space
// centroid (the list mean, up to the last k-means step), and a saved index round-trips with the same hits.
void l2_input_space_test() {
  constexpr uint32_t lists=20;
  const auto codec=train(TieredQMetric::l2,4*dimensions);
  const auto data=clustered_rows(rows,67),probes=clustered_rows(queries,71);
  TieredQIVFIndex ivf(codec,lists); ivf.build(data.data(),rows);
  const auto assignment=ivf.list_assignment();
  std::vector<double> sums(size_t(lists)*dimensions,0); std::vector<size_t> counts(lists,0);
  for (size_t row=0;row<rows;++row) {
    ++counts[assignment[row]];
    for (uint32_t j=0;j<dimensions;++j) sums[assignment[row]*dimensions+j]+=data[row*dimensions+j];
  }
  auto distance=[&](size_t row,uint32_t list) {
    double total=0;
    for (uint32_t j=0;j<dimensions;++j) { const double e=data[row*dimensions+j]-sums[list*dimensions+j]/double(counts[list]); total+=e*e; }
    return total;
  };
  size_t nearest=0;
  for (size_t row=0;row<rows;++row) {
    uint32_t best=0;
    for (uint32_t list=1;list<lists;++list) if (counts[list] && distance(row,list)<distance(row,best)) best=list;
    nearest+=best==assignment[row];
  }
  require(nearest>=rows*98/100,"L2 coarse lists are not input-space nearest-mean lists");
  const auto expected=run(ivf,probes,3);
  same(run(TieredQIVFIndex::deserialize(ivf.serialize()),probes,3),expected,"input-space L2 IVF does not round-trip");
}
// An L2 index saved as version 3 (model-space coarse lists) loads and returns the hits that version returned.
void l2_v3_file_test() {
  const auto fixtures=std::filesystem::path(__FILE__).parent_path()/"fixtures";
  const auto loaded=TieredQIVFIndex::load((fixtures/"tieredq_ivf_l2_v3.tq").string());
  constexpr size_t fixture_dimensions=16,fixture_queries=8,fixture_k=10,fixture_nprobe=3;
  auto read=[&](const char* name,auto& out) {
    std::ifstream in(fixtures/name,std::ios::binary);
    in.read(reinterpret_cast<char*>(out.data()),std::streamsize(out.size()*sizeof(out[0])));
    require(bool(in),"short L2 version 3 fixture");
  };
  std::vector<float> fixture_probes(fixture_queries*fixture_dimensions); read("tieredq_ivf_l2_v3.queries",fixture_probes);
  std::vector<uint32_t> expected(fixture_queries*fixture_k); read("tieredq_ivf_l2_v3.hits",expected);
  std::vector<uint32_t> found;
  for (size_t q=0;q<fixture_queries;++q)
    for (const auto& hit:loaded.search(fixture_probes.data()+q*fixture_dimensions,fixture_k,fixture_nprobe)) found.push_back(hit.id);
  require(found==expected,"L2 version 3 index changed its hits");
  const auto again=TieredQIVFIndex::deserialize(loaded.serialize());
  std::vector<uint32_t> reloaded;
  for (size_t q=0;q<fixture_queries;++q)
    for (const auto& hit:again.search(fixture_probes.data()+q*fixture_dimensions,fixture_k,fixture_nprobe)) reloaded.push_back(hit.id);
  require(reloaded==expected,"resaved L2 version 3 index changed its hits");
}
// Indexes saved with padded lists (payload versions 4 and 5, tests/fixtures/tieredq_ivf_padded.md) load and return
// the ids they returned. Rebuilt in place from the same rows, their lists stay fixed, so the packed rebuild (#53) has
// the same partition and codes: it returns the same ids and scores, stores fewer documents, saves as version 6 and
// round-trips.
// The IVF payload version: after the model's length (u64) and bytes.
uint32_t payload_version(const std::vector<uint8_t>& bytes) {
  uint64_t model=0; uint32_t version=0;
  std::memcpy(&model,bytes.data(),sizeof(model));
  std::memcpy(&version,bytes.data()+sizeof(model)+model,sizeof(version));
  return version;
}
void padded_fixture_test() {
  constexpr uint32_t packed_version=6;
  const auto fixtures=std::filesystem::path(__FILE__).parent_path()/"fixtures";
  constexpr size_t fixture_rows=1200,fixture_dimensions=32,fixture_queries=16,fixture_k=10,fixture_nprobe=6;
  auto read=[&](const std::string& name,auto& out) {
    std::ifstream in(fixtures/name,std::ios::binary);
    in.read(reinterpret_cast<char*>(out.data()),std::streamsize(out.size()*sizeof(out[0])));
    require(bool(in),"short padded IVF fixture");
  };
  std::vector<float> data(fixture_rows*fixture_dimensions),probes(fixture_queries*fixture_dimensions);
  read("tieredq_ivf_padded.rows",data); read("tieredq_ivf_padded.queries",probes);
  auto search=[&](const TieredQIVFIndex& index) {
    Results out{std::vector<float>(fixture_queries*fixture_k),std::vector<uint32_t>(fixture_queries*fixture_k)};
    index.search_batch(probes.data(),fixture_queries,fixture_k,fixture_nprobe,out.scores.data(),out.ids.data());
    return out;
  };
  for (const std::string name:{"ip2","l2x6"}) {
    const auto loaded=TieredQIVFIndex::load((fixtures/("tieredq_ivf_padded_"+name+".tq")).string());
    std::vector<uint32_t> expected(fixture_queries*fixture_k); read("tieredq_ivf_padded_"+name+".ids",expected);
    const auto padded=search(loaded);
    require(padded.ids==expected,"padded IVF fixture changed its hits");
    require(TieredQIVFIndex::deserialize(loaded.serialize()).serialize()==loaded.serialize(),"padded IVF does not resave as is");
    auto packed=loaded;
    packed.build(data.data(),fixture_rows);
    require(packed.list_assignment()==loaded.list_assignment(),"rebuild moved rows between lists");
    same(search(packed),padded,"packed lists change ids or scores");
    const auto bytes=packed.serialize();
    require(payload_version(bytes)==packed_version && payload_version(loaded.serialize())<packed_version,
            "packed lists do not save as version 6");
    const auto reloaded=TieredQIVFIndex::deserialize(bytes);
    same(search(reloaded),padded,"packed IVF does not round-trip");
    // Both loaded, so neither holds the encoder state a build leaves in its codecs (storage_bytes counts it).
    require(reloaded.storage_bytes()<loaded.storage_bytes(),"packed lists do not shrink the storage");
    require(bytes.size()<loaded.serialize().size(),"packed lists do not shrink the payload");
  }
}
// set_lists fixes the partition to given centroids: every row lands in its nearest given centroid's list (L2: input
// distance; inner product: q.c - |c|^2/2, as list ranking), builds keep those lists, and bad input is rejected.
void set_lists_test(TieredQMetric metric) {
  constexpr uint32_t lists=20;
  const auto codec=train(metric,2*dimensions);
  const auto data=clustered_rows(rows,73);
  const std::vector<float> centroids(data.begin(),data.begin()+size_t(lists)*dimensions);
  TieredQIVFIndex ivf(codec,lists); ivf.set_lists(centroids.data()); ivf.build(data.data(),rows);
  const auto assignment=ivf.list_assignment();
  auto closeness=[&](size_t row,uint32_t list) {
    double dot=0,norm=0;
    for (uint32_t j=0;j<dimensions;++j) { const double c=centroids[list*dimensions+j]; dot+=data[row*dimensions+j]*c; norm+=c*c; }
    return dot-norm/2;
  };
  size_t nearest=0;
  for (size_t row=0;row<rows;++row) {
    uint32_t best=0;
    for (uint32_t list=1;list<lists;++list) if (closeness(row,list)>closeness(row,best)) best=list;
    nearest+=best==assignment[row];
  }
  // Binary16 centroids may flip near-ties.
  require(nearest>=rows*99/100,"set_lists rows are not in their nearest given list");
  TieredQIVFIndex again(codec,lists); again.set_lists(centroids.data()); again.build(data.data(),rows);
  require(again.list_assignment()==assignment,"set_lists partitions differ between builds");
  std::vector<float> bad_centroids=centroids; bad_centroids[3]=std::nanf("");
  require(rejects([&] { TieredQIVFIndex index(codec,lists); index.set_lists(bad_centroids.data()); }),"nonfinite centroids accepted");
  require(rejects([&] { TieredQIVFIndex plain(codec); plain.set_lists(centroids.data()); }),"set_lists without coarse lists accepted");
}
// A build pulling rows through a loader equals the build over the same rows in one array: same lists, same serialized
// index; the loader only sees in-range ids. Covers codec and coarse lists, both metrics and several build chunks.
void loader_build_test(TieredQMetric metric,uint32_t lists,size_t count) {
  const auto codec=train(metric,2*dimensions);
  const auto data=clustered_rows(count,59);
  TieredQIVFIndex direct(codec,lists); direct.build(data.data(),count);
  TieredQIVFIndex loaded(codec,lists);
  size_t loaded_rows=0;
  loaded.build([&](const uint32_t* ids,size_t n,float* out) {
    for (size_t i=0;i<n;++i) {
      require(ids[i]<count,"loader asked for an id outside the build");
      std::copy_n(data.data()+size_t(ids[i])*dimensions,dimensions,out+i*dimensions);
    }
    loaded_rows+=n;
  },count);
  require(loaded_rows>=count,"loader build skipped rows");
  require(loaded.list_assignment()==direct.list_assignment(),"loader build changes the lists");
  require(loaded.serialize()==direct.serialize(),"loader build differs from the array build");
}
// Random partitions, including empty lists, one-id lists and a lone list (zero-width low parts): every id decodes at
// its position in order and at random, the words round-trip, and corrupt or mis-sized streams are rejected.
void elias_fano_test() {
  using tieredq_detail::EliasFanoLists;
  std::mt19937 random(41);
  for (uint32_t lists:{1u,3u,50u}) for (uint32_t universe:{0u,1u,5u,1000u,5000u}) {
    std::vector<uint32_t> list_of(universe),sizes(lists,0);
    std::vector<std::vector<uint32_t>> expected(lists);
    for (uint32_t id=0;id<universe;++id) {
      constexpr uint32_t crowded=7;
      const uint32_t list=uint32_t(random()%lists);
      list_of[id]=list%crowded ? list : 0;  // skews sizes and leaves lists empty
      ++sizes[list_of[id]];expected[list_of[id]].push_back(id);
    }
    const auto encoded=EliasFanoLists::encode(list_of,lists);
    const auto decoded=EliasFanoLists::decode(sizes,encoded.words());
    std::vector<uint32_t> at,positions;
    for (uint32_t list=0;list<lists;++list) {
      std::vector<uint32_t> listed;
      decoded.for_each(list,[&](size_t position,uint32_t id) { require(position==listed.size(),"Elias-Fano order"); listed.push_back(id); });
      require(listed==expected[list],"Elias-Fano lists do not round-trip");
      for (uint32_t position=0;position<listed.size();++position) { at.push_back(list); positions.push_back(position); }
    }
    std::vector<size_t> order(at.size());
    std::iota(order.begin(),order.end(),0); std::shuffle(order.begin(),order.end(),random);
    std::vector<uint32_t> shuffled_at,shuffled_positions,ids(at.size());
    for (size_t i:order) { shuffled_at.push_back(at[i]); shuffled_positions.push_back(positions[i]); }
    decoded.ids(shuffled_at.data(),shuffled_positions.data(),ids.size(),ids.data());
    for (size_t i=0;i<ids.size();++i) require(ids[i]==expected[shuffled_at[i]][shuffled_positions[i]],"Elias-Fano random access differs");
    auto longer=encoded.words(); longer.push_back(0);
    require(rejects([&] { EliasFanoLists::decode(sizes,longer); }),"mis-sized Elias-Fano stream accepted");
    if (universe<2) continue;
    // The first list's first low bit or, for zero-width low parts, its first high one: either changes an id.
    auto corrupt=encoded.words(); corrupt[0]^=1;
    require(rejects([&] { EliasFanoLists::decode(sizes,corrupt); }),"corrupt Elias-Fano stream accepted");
  }
}
// A version 2 file (raw ids; tests/fixtures/tieredq_ivf_v2.md) loads with the stored document of every id that the
// pre-#34 build reported, and saving it again (version 3) keeps every hit in a smaller file.
void raw_ids_file_test() {
  const auto fixtures=std::filesystem::path(__FILE__).parent_path()/"fixtures";
  const auto fixture=(fixtures/"tieredq_ivf_v2.tq").string();
  const auto loaded=TieredQIVFIndex::load(fixture);
  constexpr size_t fixture_rows=600,fixture_lists=20;
  require(loaded.size()==fixture_rows && loaded.lists()==fixture_lists,"version 2 IVF shape");
  std::vector<uint32_t> documents(fixture_rows);
  std::ifstream(fixtures/"tieredq_ivf_v2.documents",std::ios::binary).read(reinterpret_cast<char*>(documents.data()),
      std::streamsize(documents.size()*sizeof(uint32_t)));
  require(loaded.id_documents()==documents,"version 2 IVF ids differ from the stored documents");
  const size_t d=loaded.codec().config().dimensions;
  std::vector<float> probes(queries*d);
  for (size_t i=0;i<probes.size();++i) probes[i]=std::sin(0.1f*float(i));
  const auto expected=run(loaded,probes,fixture_lists);
  const auto path=(std::filesystem::temp_directory_path()/("tieredq-ivf-v3-"+std::to_string(::getpid())+".tq")).string();
  loaded.save(path);
  const auto saved_bytes=std::filesystem::file_size(path);
  const auto reloaded=TieredQIVFIndex::load(path);
  std::filesystem::remove(path);
  same(run(reloaded,probes,fixture_lists),expected,"version 3 IVF resave differs");
  require(reloaded.id_documents()==documents,"version 3 IVF resave moves ids");
  require(saved_bytes<std::filesystem::file_size(fixture),"Elias-Fano ids do not shrink the file");
}
} // namespace
void tieredq_ivf_tests() {
  elias_fano_test();
  raw_ids_file_test();
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
  for (auto metric:{TieredQMetric::inner_product,TieredQMetric::cosine}) for (uint32_t lists:{0u,20u}) probe_order_test(metric,lists);
  probe_order_test(TieredQMetric::l2,20);
  l2_input_space_test();
  for (auto metric:{TieredQMetric::inner_product,TieredQMetric::l2}) set_lists_test(metric);
  l2_v3_file_test();
  padded_fixture_test();
  for (auto metric:{TieredQMetric::inner_product,TieredQMetric::l2}) for (uint32_t lists:{0u,20u}) for (size_t count:{size_t(rows),size_t(20000)})
    loader_build_test(metric,lists,count);
  std::cout<<"PASS IVF\n";
}
