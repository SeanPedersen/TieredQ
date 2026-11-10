// Two-level coarse router of TieredQIVFIndex (#47): probing every super with a head as large as the lists re-ranks
// every list exactly, so router searches equal exact-ranking searches bit for bit; the automatic probe count keeps
// recall; batches of at least TieredQRouting::batch_queries take the exact GEMM path; the router is rebuilt with the
// lists, persists in payload version 5 through both paths, and is rejected where it cannot apply.
#include "tieredq/tieredq.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>
namespace {
using namespace dv;
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> bool rejects(F&& action) { try { action(); } catch (const std::runtime_error&) { return true; } return false; }
constexpr uint32_t dimensions=32,lists=512;
constexpr size_t rows=6000,k=10,nprobe=24;
constexpr uint32_t lists_per_super=16;  // 32 supers over the 512 lists

// Gaussian rows with a decaying spectrum: a diffuse cloud, so neighbouring lists are close and routing matters.
std::vector<float> gaussian_rows(size_t count,uint32_t seed) {
  std::mt19937 random(seed); std::normal_distribution<float> normal;
  std::vector<float> data(count*dimensions);
  for (size_t i=0;i<count;++i) for (uint32_t j=0;j<dimensions;++j) data[i*dimensions+j]=normal(random)*std::exp(-float(j)/dimensions);
  return data;
}
TieredQIVFIndex built(TieredQMetric metric,const std::vector<float>& data) {
  auto config=TieredQConfig::target_code_bits(dimensions,2*dimensions);
  config.metric=metric; config.fine_tiers=false; config.max_scan_bits=4;
  TieredQIVFIndex ivf(TieredQ::train(config,data.data(),rows,3),lists,5);
  ivf.build(data.data(),rows);
  return ivf;
}
std::vector<TieredQHit> routed(const TieredQIVFIndex& ivf,const float* query,TieredQRouting::Mode mode,uint32_t super_probes=0) {
  return ivf.search(query,k,nprobe,{},{mode,super_probes});
}
bool same_hits(const std::vector<TieredQHit>& a,const std::vector<TieredQHit>& b) {
  return std::equal(a.begin(),a.end(),b.begin(),b.end(),[](const TieredQHit& x,const TieredQHit& y) {
    return x.id==y.id && x.score==y.score;
  });
}

// Every super probed and a head covering every list: the router's order is the exact order.
void exhaustive_router_test(TieredQMetric metric) {
  const auto data=gaussian_rows(rows,11),queries=gaussian_rows(40,13);
  auto ivf=built(metric,data);
  TieredQRouterConfig config; config.lists_per_super=lists_per_super; config.rerank=lists; config.min_super_probes=lists;
  ivf.build_router(config);
  require(ivf.has_router() && ivf.router_bytes()>0,"router not built");
  for (size_t q=0;q<40;++q) {
    const float* query=queries.data()+q*dimensions;
    require(same_hits(routed(ivf,query,TieredQRouting::Mode::router),routed(ivf,query,TieredQRouting::Mode::exact)),
            "exhaustive router differs from the exact ranking");
  }
}

// probe_lists names the lists a search scans: every hit lies in them, for codec and coarse lists, routed or exact; an
// exhaustive router returns the exact order.
void probe_lists_test(TieredQMetric metric,uint32_t list_count) {
  const auto data=gaussian_rows(rows,43),queries=gaussian_rows(20,47);
  auto config=TieredQConfig::target_code_bits(dimensions,2*dimensions);
  config.metric=metric; config.fine_tiers=false; config.max_scan_bits=4;
  TieredQIVFIndex ivf(TieredQ::train(config,data.data(),rows,3),list_count,5);
  ivf.build(data.data(),rows);
  const size_t probes=std::min<size_t>(nprobe,ivf.lists()/4);
  if (list_count) {
    TieredQRouterConfig router; router.lists_per_super=lists_per_super; router.rerank=list_count; router.min_super_probes=list_count;
    ivf.build_router(router);
  }
  const auto assignment=ivf.list_assignment();
  for (size_t q=0;q<20;++q) {
    const float* query=queries.data()+q*dimensions;
    for (auto mode:{TieredQRouting::Mode::automatic,TieredQRouting::Mode::exact}) {
      const auto lists=ivf.probe_lists(query,probes,{mode});
      require(lists.size()==probes,"probe_lists size");
      const std::set<uint32_t> probed(lists.begin(),lists.end());
      for (const auto& hit:ivf.search(query,k,probes,{},{mode})) require(probed.count(assignment[hit.id]),"hit outside probe_lists");
    }
    if (list_count) require(ivf.probe_lists(query,probes)==ivf.probe_lists(query,probes,{TieredQRouting::Mode::exact}),
                            "exhaustive router order differs from exact probe_lists");
  }
}

// Default re-rank and automatic super probes: hits stay close to the exact ranking's, and fewer supers lose recall.
void automatic_router_test() {
  const auto data=gaussian_rows(rows,17),queries=gaussian_rows(100,19);
  auto ivf=built(TieredQMetric::l2,data);
  TieredQRouterConfig config; config.lists_per_super=lists_per_super; config.min_super_probes=4; config.rerank=64;
  ivf.build_router(config);
  auto overlap=[&](uint32_t super_probes) {
    size_t shared=0;
    for (size_t q=0;q<100;++q) {
      const float* query=queries.data()+q*dimensions;
      const auto exact=routed(ivf,query,TieredQRouting::Mode::exact),router=routed(ivf,query,TieredQRouting::Mode::router,super_probes);
      std::set<uint32_t> ids; for (const auto& hit:exact) ids.insert(hit.id);
      for (const auto& hit:router) shared+=ids.count(hit.id);
    }
    return double(shared)/double(100*k);
  };
  const double automatic=overlap(0),one=overlap(1);
  require(automatic>=0.95,"automatic router loses too many exact hits");
  require(one<automatic,"probing one super does not lose recall");
}

// Automatic mode: single searches route, batches of batch_queries rows take the exact GEMM ranking.
void batch_threshold_test() {
  const auto data=gaussian_rows(rows,23);
  const size_t batch=TieredQRouting::batch_queries;
  const auto queries=gaussian_rows(batch,29);
  auto ivf=built(TieredQMetric::inner_product,data);
  TieredQRouterConfig config; config.lists_per_super=lists_per_super; config.min_super_probes=1; config.expansion=1; config.rerank=0;
  ivf.build_router(config);
  std::vector<float> scores(batch*k),exact_scores(batch*k); std::vector<uint32_t> ids(batch*k),exact_ids(batch*k);
  ivf.search_batch(queries.data(),batch,k,nprobe,scores.data(),ids.data());
  ivf.search_batch(queries.data(),batch,k,nprobe,exact_scores.data(),exact_ids.data(),{},{TieredQRouting::Mode::exact});
  require(ids==exact_ids && scores==exact_scores,"large automatic batches do not take the exact ranking");
  bool differs=false;
  for (size_t q=0;q<batch && !differs;++q)
    differs=!same_hits(ivf.search(queries.data()+q*dimensions,k,nprobe),routed(ivf,queries.data()+q*dimensions,TieredQRouting::Mode::exact));
  require(differs,"two-super routing never changes a single search");
}

// build_router before build defers; set_lists drops the router until the next build; persistence keeps it.
void lifecycle_test() {
  const auto data=gaussian_rows(rows,31),queries=gaussian_rows(8,37);
  auto config=TieredQConfig::target_code_bits(dimensions,2*dimensions);
  config.metric=TieredQMetric::l2; config.fine_tiers=false; config.max_scan_bits=4;
  const auto codec=TieredQ::train(config,data.data(),rows,3);
  TieredQRouterConfig router; router.lists_per_super=lists_per_super;
  TieredQIVFIndex ivf(codec,lists,5);
  ivf.build_router(router);
  require(!ivf.has_router(),"router built before the lists");
  require(rejects([&] { routed(ivf,queries.data(),TieredQRouting::Mode::router); }),"router search before build accepted");
  ivf.build(data.data(),rows);
  require(ivf.has_router(),"build does not build the configured router");
  const auto unrouted=TieredQIVFIndex::deserialize(built(TieredQMetric::l2,data).serialize());
  require(!unrouted.has_router(),"router appeared without build_router");
  std::vector<std::vector<TieredQHit>> expected;
  for (size_t q=0;q<8;++q) expected.push_back(routed(ivf,queries.data()+q*dimensions,TieredQRouting::Mode::router));
  auto check=[&](const TieredQIVFIndex& index,const char* message) {
    require(index.has_router(),message);
    for (size_t q=0;q<8;++q) require(same_hits(routed(index,queries.data()+q*dimensions,TieredQRouting::Mode::router),expected[q]),message);
  };
  check(TieredQIVFIndex::deserialize(ivf.serialize()),"router bytes do not round-trip");
  const auto path=(std::filesystem::temp_directory_path()/("tieredq-ivf-router-"+std::to_string(::getpid())+".tq")).string();
  ivf.save(path);
  const auto loaded=TieredQIVFIndex::load(path);
  std::filesystem::remove(path);
  check(loaded,"router file does not round-trip");
  // An unbuilt index keeps its router config through persistence and builds the router at its first build.
  TieredQIVFIndex pending(codec,lists,5); pending.build_router(router);
  auto restored=TieredQIVFIndex::deserialize(pending.serialize());
  require(!restored.has_router(),"unbuilt router restored");
  restored.build(data.data(),rows);
  require(restored.has_router(),"restored router config does not build");
  std::vector<float> centroids(size_t(lists)*dimensions);
  std::copy_n(data.begin(),centroids.size(),centroids.begin());
  ivf.set_lists(centroids.data());
  require(!ivf.has_router(),"set_lists keeps a router of the previous lists");
  ivf.build(data.data(),rows);
  require(ivf.has_router(),"rebuild does not rebuild the router");
  require(rejects([&] { TieredQIVFIndex plain(codec); plain.build_router(router); }),"router on codec lists accepted");
  require(rejects([&] { TieredQRouterConfig bad=router; bad.expansion=0; ivf.build_router(bad); }),"invalid router config accepted");
  require(rejects([&] { auto copy=built(TieredQMetric::l2,data); routed(copy,queries.data(),TieredQRouting::Mode::router); }),
          "router search without a router accepted");
}
} // namespace
void tieredq_ivf_router_tests() {
  for (auto metric:{TieredQMetric::inner_product,TieredQMetric::cosine,TieredQMetric::l2}) exhaustive_router_test(metric);
  for (auto metric:{TieredQMetric::inner_product,TieredQMetric::l2}) for (uint32_t list_count:{0u,lists}) probe_lists_test(metric,list_count);
  automatic_router_test();
  batch_threshold_test();
  lifecycle_test();
  std::cout<<"PASS IVF router\n";
}
