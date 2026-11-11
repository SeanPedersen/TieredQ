// Nested IVF serving (#23): IVF lists give back exactly the payload rows the codec encodes, in every fast layout;
// prefix scan plus suffix rescoring returns the refinement's own ranking of the scanned candidates and beats the
// prefix scan alone; batch, file-backed and deserialized indexes answer identically with either tier layout; an exact
// SQ8 stage ranks the candidates exactly in every metric; mismatches are rejected.
#include "tieredq/tieredq_nested.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unistd.h>
#include <vector>
namespace {
using namespace dv;
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
constexpr uint32_t dimensions=48,centroids=8,seed=11;
constexpr size_t rows=2003,queries=24,k=10,candidates=40,nprobe=3;
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
TieredQ train(const std::vector<float>& data,uint32_t code_bits,TieredQMetric metric,
              TieredQIndexMode mode=TieredQIndexMode::fast) {
  auto config=TieredQConfig::target_code_bits(dimensions,code_bits);
  config.num_centroids=centroids; config.metric=metric; config.target_mode=mode; config.fine_tiers=false;
  return TieredQ::train(config,data.data(),rows,seed);
}
TieredQIVFIndex built(const TieredQ& codec,const std::vector<float>& data) {
  TieredQIVFIndex index(codec);
  index.build(data.data(),rows);
  return index;
}

// Narrow, additive and wide models cover the nibble lookup (with split codes) and the packed code-word layouts.
void document_rows_test(uint32_t code_bits,TieredQIndexMode mode) {
  const auto data=clustered_rows(rows,seed);
  const auto codec=train(data,code_bits,TieredQMetric::inner_product,mode);
  const auto index=built(codec,data);
  const auto expected=codec.encode(data.data(),rows);
  const auto documents=index.id_documents();
  std::vector<uint8_t> recovered(expected.size());
  index.document_rows(documents.data(),rows,recovered.data());
  require(recovered==expected,"IVF document rows differ from the encoded payload");
}

TieredQNestedIVFIndex nested(const std::vector<float>& data,TieredQMetric metric,TieredQTierRows tier_rows) {
  const auto prefix=train(data,prefix_bits,metric);
  auto refinement=TieredQRefinement::train(prefix,total_bits,data.data(),rows,seed);
  TieredQNestedIVFIndex index(built(prefix,data),std::move(refinement),tier_rows);
  std::vector<uint8_t> tier(rows*index.tier_bytes_per_vector());
  index.encode_tier(data.data(),0,rows,tier.data());
  index.attach_tier(std::make_shared<TieredQMemoryRows>(std::move(tier),index.tier_bytes_per_vector()));
  return index;
}
float exact(TieredQMetric metric,const float* q,const float* x) {
  double dot=0,qq=0,xx=0,distance=0;
  for (uint32_t j=0;j<dimensions;++j) {
    dot+=double(q[j])*x[j]; qq+=double(q[j])*q[j]; xx+=double(x[j])*x[j]; distance+=(double(q[j])-x[j])*(q[j]-x[j]);
  }
  if (metric==TieredQMetric::cosine) return float(dot/std::sqrt(qq*xx));
  return float(metric==TieredQMetric::l2 ? -distance : dot);
}
size_t true_hits(TieredQMetric metric,const std::vector<float>& data,const float* query,const std::vector<TieredQHit>& hits) {
  std::vector<float> truth(rows);
  for (size_t i=0;i<rows;++i) truth[i]=exact(metric,query,data.data()+i*dimensions);
  std::vector<uint32_t> order(rows); std::iota(order.begin(),order.end(),0);
  std::partial_sort(order.begin(),order.begin()+k,order.end(),[&](uint32_t a,uint32_t b) { return truth[a]>truth[b]; });
  size_t found=0;
  for (const auto& hit:hits) found+=std::count(order.begin(),order.begin()+k,hit.id);
  return found;
}

// The nested hits are the refinement's top k of the prefix scan's candidates, scored on the codec's own payload.
void search_test(TieredQMetric metric,TieredQTierRows tier_rows) {
  const auto data=clustered_rows(rows,seed),probes=clustered_rows(queries,seed+1);
  const auto index=nested(data,metric,tier_rows);
  const auto& refinement=index.refinement();
  const auto payload=index.index().codec().encode(data.data(),rows);
  std::vector<uint8_t> suffix(rows*refinement.bytes_per_vector());
  refinement.encode(data.data(),rows,payload.data(),suffix.data());
  const size_t prefix_stride=index.index().codec().bytes_per_vector(),suffix_stride=refinement.bytes_per_vector();
  size_t nested_hits=0,prefix_hits=0;
  for (size_t q=0;q<queries;++q) {
    const float* query=probes.data()+q*dimensions;
    const auto hits=index.search(query,k,nprobe,candidates);
    const auto scanned=index.index().search(query,candidates,nprobe);
    std::vector<uint8_t> prefix_rows,suffix_rows;
    for (const auto& hit:scanned) {
      prefix_rows.insert(prefix_rows.end(),payload.begin()+hit.id*prefix_stride,payload.begin()+(hit.id+1)*prefix_stride);
      suffix_rows.insert(suffix_rows.end(),suffix.begin()+hit.id*suffix_stride,suffix.begin()+(hit.id+1)*suffix_stride);
    }
    std::vector<float> scores(scanned.size());
    refinement.score(prefix_rows.data(),suffix_rows.data(),scanned.size(),refinement.query(query),scores.data());
    std::vector<uint32_t> order(scanned.size()); std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b) { return scores[a]>scores[b]; });
    require(hits.size()==std::min(k,scanned.size()),"nested search returned the wrong hit count");
    for (size_t i=0;i<hits.size();++i) require(hits[i].id==scanned[order[i]].id,"nested hits differ from refinement ranking");
    nested_hits+=true_hits(metric,data,query,hits);
    prefix_hits+=true_hits(metric,data,query,index.index().search(query,k,nprobe));
  }
  require(nested_hits>prefix_hits,"suffix rescoring does not beat the prefix scan");
}

std::vector<std::vector<TieredQHit>> answers(const TieredQNestedIVFIndex& index,const std::vector<float>& probes) {
  std::vector<std::vector<TieredQHit>> out;
  for (size_t q=0;q<queries;++q) out.push_back(index.search(probes.data()+q*dimensions,k,nprobe,candidates));
  return out;
}
bool same(const std::vector<std::vector<TieredQHit>>& a,const std::vector<std::vector<TieredQHit>>& b) {
  if (a.size()!=b.size()) return false;
  for (size_t q=0;q<a.size();++q) {
    if (a[q].size()!=b[q].size()) return false;
    for (size_t i=0;i<a[q].size();++i) if (a[q][i].id!=b[q][i].id || a[q][i].score!=b[q][i].score) return false;
  }
  return true;
}

void batch_file_and_serialization_test(TieredQTierRows tier_rows) {
  const auto data=clustered_rows(rows,seed),probes=clustered_rows(queries,seed+1);
  auto index=nested(data,TieredQMetric::l2,tier_rows);
  const auto expected=answers(index,probes);

  std::vector<float> scores(queries*k); std::vector<uint32_t> ids(queries*k);
  index.search_batch(probes.data(),queries,k,nprobe,candidates,scores.data(),ids.data());
  for (size_t q=0;q<queries;++q) for (size_t i=0;i<k;++i)
    require(ids[q*k+i]==expected[q][i].id && scores[q*k+i]==expected[q][i].score,"batch differs from single search");

  const auto path=(std::filesystem::temp_directory_path()/("tieredq_nested_"+std::to_string(::getpid())+".rows")).string();
  constexpr size_t chunk=500;
  {
    TieredQRowFileWriter writer(path,index.tier_bytes_per_vector());
    std::vector<uint8_t> tier(chunk*index.tier_bytes_per_vector());
    for (size_t first=0;first<rows;first+=chunk) {
      const size_t n=std::min(chunk,rows-first);
      index.encode_tier(data.data()+first*dimensions,first,n,tier.data());
      writer.append(tier.data(),n);
    }
    writer.finish();
  }
  auto loaded=TieredQNestedIVFIndex::deserialize(index.serialize());
  loaded.attach_tier(std::make_shared<TieredQFileRows>(path));
  require(loaded.tier_rows()==tier_rows,"deserialized tier layout differs");
  require(same(answers(loaded,probes),expected),"file-backed deserialized index differs");
  loaded.attach_tier(std::make_shared<TieredQFileRows>(path,TieredQFileReads::direct));
  require(same(answers(loaded,probes),expected),"direct-read file tier differs");
  constexpr size_t queue_depth=16;
  for (auto reads:{TieredQFileReads::cached,TieredQFileReads::direct}) {
    loaded.attach_tier(std::make_shared<TieredQFileRows>(path,reads,queue_depth));
    require(same(answers(loaded,probes),expected),"concurrent file tier reads differ");
  }

  bool rejected=false;
  try { index.search(probes.data(),k,nprobe,k-1); } catch (const std::runtime_error&) { rejected=true; }
  require(rejected,"fewer candidates than k accepted");
  rejected=false;
  try { index.attach_tier(std::make_shared<TieredQMemoryRows>(std::vector<uint8_t>(rows),1)); }
  catch (const std::runtime_error&) { rejected=true; }
  require(rejected,"mismatched tier rows accepted");
  {
    TieredQRowFileWriter unfinished(path,index.tier_bytes_per_vector());
    std::vector<uint8_t> row(index.tier_bytes_per_vector());
    unfinished.append(row.data(),1);
  }
  rejected=false;
  try { TieredQFileRows rows_file(path); } catch (const std::runtime_error&) { rejected=true; }
  require(rejected,"unfinished row file accepted");
  std::filesystem::remove(path);
}

// Two-step search over an external tier: prefix candidates, then rescoring with caller-gathered tier rows (zeros in
// padded slots) answers as search_batch does; out-of-range ids are rejected.
void external_tier_test(TieredQTierRows tier_rows) {
  const auto data=clustered_rows(rows,seed),probes=clustered_rows(queries,seed+1);
  const auto index=nested(data,TieredQMetric::inner_product,tier_rows);
  std::vector<float> scores(queries*k); std::vector<uint32_t> ids(queries*k);
  index.search_batch(probes.data(),queries,k,nprobe,candidates,scores.data(),ids.data());

  const size_t stride=index.tier_bytes_per_vector();
  std::vector<uint8_t> tier(rows*stride);
  index.encode_tier(data.data(),0,rows,tier.data());
  std::vector<float> found_scores(queries*candidates); std::vector<uint32_t> found(queries*candidates);
  index.index().search_batch(probes.data(),queries,candidates,nprobe,found_scores.data(),found.data());
  std::vector<uint8_t> gathered(found.size()*stride,0);
  for (size_t i=0;i<found.size();++i)
    if (found[i]!=UINT32_MAX) std::copy_n(tier.data()+size_t(found[i])*stride,stride,gathered.data()+i*stride);
  std::vector<float> rescored(queries*k); std::vector<uint32_t> rescored_ids(queries*k);
  index.rescore_batch(probes.data(),queries,found.data(),candidates,gathered.data(),k,rescored.data(),rescored_ids.data());
  require(rescored==scores && rescored_ids==ids,"external tier rescoring differs from search_batch");

  found[0]=uint32_t(rows);
  bool rejected=false;
  try { index.rescore_batch(probes.data(),1,found.data(),candidates,gathered.data(),k,rescored.data(),rescored_ids.data()); }
  catch (const std::runtime_error&) { rejected=true; }
  require(rejected,"out-of-range candidate id accepted");
}

// SQ8 codes of data (per-dimension min/max, 255 steps) with their scales and offsets: decoded = code * scale + offset.
struct SQ8 { std::vector<uint8_t> codes; std::vector<float> scales,offsets; };
SQ8 sq8(const std::vector<float>& data) {
  SQ8 out; out.codes.resize(rows*dimensions); out.scales.resize(dimensions); out.offsets.resize(dimensions);
  for (uint32_t j=0;j<dimensions;++j) {
    float low=data[j],high=data[j];
    for (size_t i=0;i<rows;++i) { low=std::min(low,data[i*dimensions+j]); high=std::max(high,data[i*dimensions+j]); }
    out.scales[j]=std::max(high-low,1e-6f)/255; out.offsets[j]=low+128*out.scales[j];
    for (size_t i=0;i<rows;++i)
      out.codes[i*dimensions+j]=uint8_t(int8_t(std::lround((data[i*dimensions+j]-out.offsets[j])/out.scales[j])));
  }
  return out;
}

// With an exact stage over every candidate, the hits are the scanned candidates' exact top k on the SQ8 rows; batch
// and file-backed rows answer identically, and a narrower final stage still returns k hits.
void exact_stage_test(TieredQMetric metric) {
  const auto data=clustered_rows(rows,seed),probes=clustered_rows(queries,seed+1);
  auto index=nested(data,metric,TieredQTierRows::suffix);
  const auto store=sq8(data);
  auto exact_rows=std::make_shared<TieredQSQ8Rows>(std::make_shared<TieredQMemoryRows>(store.codes,dimensions),
                                                   store.scales,store.offsets);
  std::vector<float> decoded(rows*dimensions);
  std::vector<uint32_t> all(rows); std::iota(all.begin(),all.end(),0);
  exact_rows->decode(all.data(),rows,decoded.data());
  index.attach_exact(exact_rows,candidates);
  require(index.has_exact() && index.finals()==candidates,"exact stage not attached");
  for (size_t q=0;q<queries;++q) {
    const float* query=probes.data()+q*dimensions;
    const auto hits=index.search(query,k,nprobe,candidates);
    auto scanned=index.index().search(query,candidates,nprobe);
    for (auto& hit:scanned) hit.score=exact(metric,query,decoded.data()+size_t(hit.id)*dimensions);
    std::stable_sort(scanned.begin(),scanned.end(),[](const TieredQHit& a,const TieredQHit& b) {
      return a.score!=b.score ? a.score>b.score : a.id<b.id;
    });
    require(hits.size()==std::min(k,scanned.size()),"exact stage returned the wrong hit count");
    for (size_t i=0;i<hits.size();++i) {
      require(hits[i].id==scanned[i].id,"exact stage hits differ from the exact ranking of the candidates");
      require(std::abs(hits[i].score-scanned[i].score)<=1e-4f*std::max(1.f,std::abs(scanned[i].score)),
              "exact stage scores differ from the exact metric");
    }
  }
  const auto single=answers(index,probes);
  std::vector<float> scores(queries*k); std::vector<uint32_t> ids(queries*k);
  index.search_batch(probes.data(),queries,k,nprobe,candidates,scores.data(),ids.data());
  for (size_t q=0;q<queries;++q) for (size_t i=0;i<single[q].size();++i)
    require(ids[q*k+i]==single[q][i].id && scores[q*k+i]==single[q][i].score,"exact stage batch differs from search");

  const auto path=(std::filesystem::temp_directory_path()/("tieredq_exact_"+std::to_string(::getpid())+".rows")).string();
  {
    TieredQRowFileWriter writer(path,dimensions);
    writer.append(store.codes.data(),rows);
    writer.finish();
  }
  index.attach_exact(std::make_shared<TieredQSQ8Rows>(std::make_shared<TieredQFileRows>(path,TieredQFileReads::cached,4),
                                                      store.scales,store.offsets),candidates);
  require(same(answers(index,probes),single),"file-backed exact rows answer differently");
  std::filesystem::remove(path);

  index.attach_exact(exact_rows,k/2);
  for (size_t q=0;q<queries;++q)
    require(index.search(probes.data()+q*dimensions,k,nprobe,candidates).size()==k,"a narrow final stage lost hits");
  index.attach_exact(nullptr,0);
  require(!index.has_exact(),"exact stage not detached");

  bool rejected=false;
  try { TieredQSQ8Rows(std::make_shared<TieredQMemoryRows>(store.codes,dimensions),std::vector<float>(dimensions-1,1),
                       std::vector<float>(dimensions-1,0)); }
  catch (const std::runtime_error&) { rejected=true; }
  require(rejected,"SQ8 rows with the wrong scale count accepted");
}

void mismatched_prefix_test() {
  const auto data=clustered_rows(rows,seed);
  const auto prefix=train(data,prefix_bits,TieredQMetric::inner_product);
  const auto other=train(data,prefix_bits*2,TieredQMetric::inner_product);
  auto refinement=TieredQRefinement::train(other,total_bits,data.data(),rows,seed);
  bool rejected=false;
  try { TieredQNestedIVFIndex index(built(prefix,data),std::move(refinement)); } catch (const std::runtime_error&) { rejected=true; }
  require(rejected,"refinement of another prefix accepted");
}
} // namespace

void tieredq_nested_tests() {
  constexpr uint32_t narrow_bits=48,additive_bits=144,wide_bits=288;
  document_rows_test(narrow_bits,TieredQIndexMode::fast);
  document_rows_test(additive_bits,TieredQIndexMode::fast);
  document_rows_test(wide_bits,TieredQIndexMode::fast);
  document_rows_test(wide_bits,TieredQIndexMode::full);
  for (auto tier_rows:{TieredQTierRows::prefix_and_suffix,TieredQTierRows::suffix}) {
    for (auto metric:{TieredQMetric::inner_product,TieredQMetric::cosine,TieredQMetric::l2}) search_test(metric,tier_rows);
    batch_file_and_serialization_test(tier_rows);
    external_tier_test(tier_rows);
  }
  for (auto metric:{TieredQMetric::inner_product,TieredQMetric::cosine,TieredQMetric::l2}) exact_stage_test(metric);
  mismatched_prefix_test();
  std::cout<<"nested IVF tests passed\n";
}
