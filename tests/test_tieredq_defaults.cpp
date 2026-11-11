// Low-RAM defaults (#52): fast list training and the router switch on at their list thresholds and stay overridable,
// the automatic list count follows rows / 135, loaded indexes keep the router they were saved with, a prefix's
// training-only scan cap does not change its refinement, and TieredQNestedIVFIndex::build trains the default nested
// configuration identically from float rows and from an SQ8 store, whose write_tier() matches encode_tier().
#include "tieredq/tieredq_nested.h"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>
namespace {
using namespace tq;
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
constexpr uint32_t dimensions=16,seed=7;
constexpr size_t rows=TieredQRouterConfig::automatic_lists+3000,queries=8,k=10,nprobe=64;

std::vector<float> gaussian_rows(size_t count,uint32_t noise_seed) {
  std::mt19937 random(noise_seed); std::normal_distribution<float> normal;
  std::vector<float> data(count*dimensions);
  for (size_t i=0;i<count;++i) for (uint32_t j=0;j<dimensions;++j) data[i*dimensions+j]=normal(random)*std::exp(-float(j)/dimensions);
  return data;
}
TieredQ codec(const std::vector<float>& data,uint32_t max_scan_bits=0) {
  auto config=TieredQConfig::target_code_bits(dimensions,2*dimensions);
  config.fine_tiers=false; config.max_scan_bits=max_scan_bits;
  return TieredQ::train(config,data.data(),4096,seed);
}
bool same(const TieredQCoarseTraining& a,const TieredQCoarseTraining& b) {
  return a.iterations==b.iterations && a.assignment_supers==b.assignment_supers && a.assignment_probes==b.assignment_probes &&
         a.sample_rows_per_list==b.sample_rows_per_list;
}
std::vector<TieredQHit> hits(const TieredQIVFIndex& index,const std::vector<float>& probes,TieredQRouting routing={}) {
  std::vector<TieredQHit> out;
  for (size_t q=0;q<queries;++q) {
    const auto found=index.search(probes.data()+q*dimensions,k,nprobe,{},routing);
    out.insert(out.end(),found.begin(),found.end());
  }
  return out;
}
bool equal(const std::vector<TieredQHit>& a,const std::vector<TieredQHit>& b) {
  if (a.size()!=b.size()) return false;
  for (size_t i=0;i<a.size();++i) if (a[i].id!=b[i].id || a[i].score!=b[i].score) return false;
  return true;
}

void automatic_lists_test() {
  require(TieredQIVFIndex::automatic_lists(30000)==0,"small sets keep the codec's lists");
  require(TieredQIVFIndex::automatic_lists(200000)==256,"200K rows get 256 lists");
  require(TieredQIVFIndex::automatic_lists(1000000)==2048,"1M rows get 2K lists");
  require(TieredQIVFIndex::automatic_lists(1344643)==2048,"1.34M rows get 2K lists");
  require(TieredQIVFIndex::automatic_lists(8840823)==65536,"8.84M rows get 64K lists");
}

void scan_cap_test() {
  auto config=TieredQConfig::target_code_bits(dimensions,2*dimensions); config.fine_tiers=false;
  require(TieredQIVFIndex::automatic_scan_bits(config)==4,"2-bit IVF codecs do not scan on nibbles");
  config.max_scan_bits=6;
  require(TieredQIVFIndex::automatic_scan_bits(config)==6,"an explicit scan cap is overridden");
  config.max_scan_bits=0; config.fine_tiers=true;
  require(TieredQIVFIndex::automatic_scan_bits(config)==0,"fine tiers lose their 6-bit cap");
  config=TieredQConfig::target_code_bits(dimensions,4*dimensions); config.fine_tiers=false;
  require(TieredQIVFIndex::automatic_scan_bits(config)==0,"4-bit IVF codecs get the 4-bit cap");
}

void coarse_training_threshold_test() {
  const TieredQCoarseTraining exact{},fast{4,256,16,40};
  require(same(TieredQCoarseTraining::automatic(TieredQCoarseTraining::fast_lists-1),exact),"few lists train exactly");
  require(same(TieredQCoarseTraining::automatic(TieredQCoarseTraining::fast_lists),fast),"many lists train fast");
  const auto data=gaussian_rows(4096,3);
  TieredQIVFIndex many(codec(data),TieredQCoarseTraining::fast_lists);
  require(same(many.coarse_training(),fast),"a many-list index trains fast without being told");
  const TieredQCoarseTraining chosen{9,0,16,50};
  many.set_coarse_training(chosen);
  require(same(many.coarse_training(),chosen),"an explicit coarse training overrides the automatic one");
  TieredQIVFIndex few(codec(data),TieredQCoarseTraining::fast_lists/2);
  require(same(few.coarse_training(),exact),"a few-list index trains exactly");
  bool rejected=false;
  try { TieredQCoarseTraining{4,0,16,0}.validate(); } catch (const std::runtime_error&) { rejected=true; }
  require(rejected,"a list-training sample of no rows per list is accepted");
}

// At the router threshold the index routes without being told, exactly as an explicit build_router() at the defaults,
// keeps it through a round trip, and opts out through remove_router(); below it no router appears.
void router_threshold_test() {
  const auto data=gaussian_rows(rows,5),probes=gaussian_rows(queries,9);
  const auto model=codec(data);
  TieredQIVFIndex automatic(model,TieredQRouterConfig::automatic_lists,seed);
  automatic.build(data.data(),rows);
  require(automatic.has_router(),"a many-list index builds no router by default");
  TieredQIVFIndex explicit_router(model,TieredQRouterConfig::automatic_lists,seed);
  explicit_router.build_router();
  explicit_router.build(data.data(),rows);
  require(equal(hits(automatic,probes),hits(explicit_router,probes)),"the automatic router differs from build_router()");
  const auto loaded=TieredQIVFIndex::deserialize(automatic.serialize());
  require(loaded.has_router() && equal(hits(loaded,probes),hits(automatic,probes)),"the automatic router does not round-trip");

  TieredQIVFIndex removed(model,TieredQRouterConfig::automatic_lists,seed);
  removed.remove_router();
  removed.build(data.data(),rows);
  require(!removed.has_router(),"remove_router() before build still routes");
  require(equal(hits(removed,probes),hits(automatic,probes,{TieredQRouting::Mode::exact})),"an unrouted index ranks unlike exact routing");
  auto reloaded=TieredQIVFIndex::deserialize(removed.serialize());
  reloaded.build(data.data(),rows);
  require(!reloaded.has_router(),"a loaded index without a router gains one on rebuild");

  TieredQIVFIndex few(model,TieredQRouterConfig::automatic_lists/2,seed);
  few.build(data.data(),rows);
  require(!few.has_router(),"a few-list index builds a router by default");
}

// The prefix's max_scan_bits is training-only, so a refinement trained on the prefix equals one trained on its copy.
void refinement_scan_cap_test() {
  const auto data=gaussian_rows(4096,13);
  const auto capped=codec(data,4);
  const auto loaded=TieredQ::deserialize(capped.serialize());
  const auto direct=TieredQRefinement::train(capped,8*dimensions,data.data(),4096,seed,true);
  const auto copied=TieredQRefinement::train(loaded,8*dimensions,data.data(),4096,seed,true);
  require(direct.serialize()==copied.serialize(),"the prefix's scan cap changes its refinement");
  require(direct.suffix_bits()+capped.config().code_bits>=7*dimensions,"the capped prefix's 8-bit refinement is narrowed");
}

struct Store { std::vector<int8_t> codes; std::vector<float> scales,offsets,decoded; };
Store sq8_store(size_t count) {
  std::mt19937 random(17); std::uniform_int_distribution<int> code(-128,127); std::uniform_real_distribution<float> unit(0.5f,1.5f);
  Store store; store.codes.resize(count*dimensions); store.scales.resize(dimensions); store.offsets.resize(dimensions);
  for (auto& value:store.codes) value=int8_t(code(random));
  for (uint32_t j=0;j<dimensions;++j) { store.scales[j]=0.01f*unit(random)*std::exp(-float(j)/dimensions); store.offsets[j]=unit(random)-1; }
  store.decoded.resize(count*dimensions);
  for (size_t i=0;i<count;++i) for (uint32_t j=0;j<dimensions;++j)
    store.decoded[i*dimensions+j]=float(store.codes[i*dimensions+j])*store.scales[j]+store.offsets[j];
  return store;
}
TieredQIVFIndex::RowLoader float_loader(const std::vector<float>& data) {
  return [&data](const uint32_t* ids,size_t count,float* out) {
    for (size_t i=0;i<count;++i) std::copy_n(data.data()+size_t(ids[i])*dimensions,dimensions,out+i*dimensions);
  };
}

// Default nested builds: 1-bit prefix, 8-bit fine suffix tier stored suffix-only, 40 candidates; an SQ8 store builds
// the same bytes as its decoded rows and writes the same tier as encode_tier().
void nested_build_test(const std::filesystem::path& dir) {
  constexpr size_t count=6000;
  const auto store=sq8_store(count);
  const auto loader=sq8_row_loader(store.codes.data(),count,store.scales,store.offsets);
  std::vector<uint32_t> ids(count); std::iota(ids.begin(),ids.end(),0u);
  std::vector<float> loaded(count*dimensions); loader(ids.data(),count,loaded.data());
  require(loaded==store.decoded,"the SQ8 loader decodes unlike codes * scales + offsets");

  TieredQNestedConfig config; config.dimensions=dimensions;
  auto index=TieredQNestedIVFIndex::build(config,float_loader(store.decoded),count);
  require(index.tier_rows()==TieredQTierRows::suffix,"the default tier is not suffix-only");
  require(index.index().lists()==256 && !index.index().has_router(),"6,000 rows do not keep the codec's lists");
  const size_t prefix=index.index().codec().config().code_bits,total=prefix+index.refinement().suffix_bits();
  require(prefix>=dimensions && prefix<2*dimensions,"the default prefix is not 1 bit per dimension");
  require(total>=7*dimensions && total<=9*dimensions,"the default total is not 8 bits per dimension");
  require(index.index().codec().tiers().front().bits<=4,"the 2-bit prefix scans more than 4 bits");
  const auto streamed=TieredQNestedIVFIndex::build(config,loader,count);
  require(streamed.serialize()==index.serialize(),"an SQ8 build differs from the float build");

  std::vector<uint8_t> tier(count*index.tier_bytes_per_vector());
  index.encode_tier(store.decoded.data(),0,count,tier.data());
  const auto path=(dir/"defaults_tier.rows").string();
  streamed.write_tier(path,loader);
  const TieredQFileRows file(path);
  std::vector<uint8_t> written(tier.size()); file.read(ids.data(),count,written.data());
  require(written==tier,"write_tier() from SQ8 differs from encode_tier()");
  index.attach_tier(std::make_shared<TieredQMemoryRows>(std::move(tier),index.tier_bytes_per_vector()));
  const auto probes=gaussian_rows(queries,19);
  for (size_t q=0;q<queries;++q) {
    const float* query=probes.data()+q*dimensions;
    const auto defaulted=index.search(query,k,8),given=index.search(query,k,8,TieredQNestedIVFIndex::default_candidates);
    require(defaulted.size()==given.size(),"default candidates are not 40");
    for (size_t i=0;i<given.size();++i) require(defaulted[i].id==given[i].id && defaulted[i].score==given[i].score,"default candidates are not 40");
  }
  require(TieredQNestedIVFIndex::candidates_for(20)==40,"k up to 20 does not rescore 40 candidates");
  require(TieredQNestedIVFIndex::candidates_for(100)==200,"k above 20 does not rescore 2k candidates");
  const auto round_trip=TieredQNestedIVFIndex::deserialize(index.serialize());
  require(round_trip.serialize()==index.serialize() && round_trip.tier_rows()==TieredQTierRows::suffix,"a default nested index does not round-trip");
}

// Many lists: the nested build routes and trains fast by default; explicit settings win.
void nested_many_lists_test() {
  const auto data=gaussian_rows(rows,23);
  TieredQNestedConfig config; config.dimensions=dimensions; config.lists=TieredQRouterConfig::automatic_lists;
  const auto automatic=TieredQNestedIVFIndex::build(config,float_loader(data),rows);
  require(automatic.index().has_router(),"a many-list nested build has no router");
  require(same(automatic.index().coarse_training(),TieredQCoarseTraining::automatic(config.lists)),"a many-list nested build trains exactly");
  config.router=false; config.coarse_training=TieredQCoarseTraining{2,0,16,8};
  const auto chosen=TieredQNestedIVFIndex::build(config,float_loader(data),rows);
  require(!chosen.index().has_router(),"router=false still routes");
  require(same(chosen.index().coarse_training(),*config.coarse_training),"an explicit coarse training is ignored");
  config.lists=64; config.router=true; config.coarse_training.reset();
  require(TieredQNestedIVFIndex::build(config,float_loader(data),rows).index().has_router(),"router=true below the threshold does not route");
}
} // namespace

void tieredq_defaults_tests() {
  const auto dir=std::filesystem::temp_directory_path()/("tieredq_defaults_"+std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  try {
    automatic_lists_test();
    scan_cap_test();
    coarse_training_threshold_test();
    router_threshold_test();
    refinement_scan_cap_test();
    nested_build_test(dir);
    nested_many_lists_test();
  } catch (...) { std::filesystem::remove_all(dir); throw; }
  std::filesystem::remove_all(dir);
}
