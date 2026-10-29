// Additive 3-bit tiers in fast mode: split derivation, padded allocation, scoring parity, persistence and
// compatibility of models that keep the Lloyd-Max 3-bit codebook.
#include "tieredq/tieredq.h"
#include "tieredq_io.h"
#include "tieredq_lookup_layout.h"
#include "tieredq_train_allocation.h"
#include "tieredq_train_codebook.h"
#include "tieredq_train_codebook_data.h"
#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
namespace {
using namespace dv;
using namespace dv::tieredq_detail;
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
constexpr uint32_t dimensions=32,centroids=4;
constexpr size_t training_rows=1024,rows=600,queries=24,k=10;
constexpr size_t min_overlap=8;
// ARM fast caches store one int8 level per coordinate instead of x86 nibble lookups.
#if defined(__aarch64__) || defined(_M_ARM64)
constexpr bool nibble_layout=false;
#else
constexpr bool nibble_layout=true;
#endif
// Mildly decaying spectrum: 2.25-3.5 bits/dim allocations mix 2-, 3- and 4-bit tiers.
std::vector<float> spectrum_rows(size_t count,uint32_t seed) {
  std::mt19937 random(seed); std::normal_distribution<float> normal;
  std::vector<float> data(count*dimensions);
  for (size_t i=0;i<count;++i) for (uint32_t j=0;j<dimensions;++j) data[i*dimensions+j]=normal(random)*std::exp(-float(j)/dimensions);
  return data;
}
// The persisted 3-bit table, read from the serialized model.
std::vector<float> persisted_three_levels(const TieredQ& codec) {
  const auto data=codec.serialize(); size_t pos=0;
  auto u32=[&] { uint32_t v=0; for (int i=0;i<4;++i) v|=uint32_t(data.at(pos++))<<(8*i); return v; };
  auto u64=[&] { const uint64_t lo=u32(); return lo|uint64_t(u32())<<32; };
  auto floats=[&] { std::vector<float> v(u64()); for (auto& x:v) x=std::bit_cast<float>(u32()); return v; };
  for (int i=0;i<4;++i) u32();
  const uint32_t format=u32(); u32(); u32();
  const uint32_t tiers=u32(); pos+=size_t(tiers)*5*sizeof(uint32_t);
  floats();
  if (format==TieredQ::model_format) pos+=u64()*sizeof(uint16_t); else floats();
  floats(); floats();
  require(u32()==9,"unexpected codebook count");
  std::vector<float> book;
  for (unsigned b=0;b<=split_code_bits;++b) book=floats();
  return book;
}
bool has_three(const TieredQ& codec) {
  return std::any_of(codec.tiers().begin(),codec.tiers().end(),[](const auto& tier){return tier.bits==split_code_bits;});
}
unsigned widest(const TieredQ& codec) {
  unsigned bits=0; for (const auto& tier:codec.tiers()) bits=std::max(bits,tier.bits); return bits;
}
TieredQ train(uint32_t code_bits,TieredQIndexMode target) {
  const auto sample=spectrum_rows(training_rows,3);
  TieredQConfig config{dimensions,code_bits,centroids,true}; config.target_mode=target;
  return TieredQ::train(config,sample.data(),training_rows);
}
size_t stored_code_bits(const TieredQIndex& index) { return (index.bytes_per_vector()-TieredQ::metadata_bytes)*CHAR_BIT; }
size_t mean_overlap(const TieredQIndex& fast,const TieredQIndex& full,const std::vector<float>& data) {
  size_t overlap=0;
  for (size_t q=0;q<queries;++q) {
    const auto a=fast.search(data.data()+q*dimensions,k),b=full.search(data.data()+q*dimensions,k);
    for (const auto& hit:a) overlap+=std::any_of(b.begin(),b.end(),[&](const auto& other){return other.id==hit.id;});
  }
  return overlap/queries;
}
void require_same(const std::vector<TieredQHit>& a,const std::vector<TieredQHit>& b,const char* message) {
  require(a.size()==b.size(),message);
  for (size_t i=0;i<a.size();++i) require(a[i].id==b[i].id && a[i].score==b[i].score,message);
}

void split_derivation_test() {
  const auto& book=additive_three_codebook();
  const auto split=additive_three_split(book);
  require(split.has_value(),"additive codebook does not split");
  std::array<bool,split_levels> seen{};
  for (size_t code=0;code<split_levels;++code) {
    const uint8_t parts=split->code[code],low=parts&((1u<<split_low_bits)-1),high=parts>>split_low_bits;
    require(parts<split_levels && !seen[parts],"split codes are not a permutation");
    seen[parts]=true;
    require(split->part_levels[0][low]+split->part_levels[1][high]==book[code],"split parts are not exact");
  }
  require(std::is_sorted(split->part_levels[0].begin(),split->part_levels[0].end()),"low part levels unsorted");
  require(!additive_three_split(gaussian_codebooks()[split_code_bits]),"Lloyd-Max 3-bit codebook splits");
  require(!additive_three_split(gaussian_codebooks()[2]),"4-level codebook splits");
  auto perturbed=book; perturbed[5]=std::nextafter(perturbed[5],INFINITY);
  require(!additive_three_split(perturbed),"inexact additive codebook splits");
  // Split parts share streams: 5x4 + ceil((3+5)/2) + ceil((2+5)/4) groups, against 5 + 5 + 2 + 1 unsplit.
  const LookupWidthCounts widths{0,2,3,5,5};
  require(lookup_group_count(widths,true)==5+4+2 && lookup_group_count(widths,false)==5+5+2+1,"group count");
  // 199 two-bit + one 3-bit coordinate pad the 1-bit stream to a full nibble: 404 bits, stored as 408.
  Allocation bits(200,2); bits[0]=3;
  require(lookup_stored_bits(bits,true)==408 && lookup_stored_bits(Allocation(200,2),true)==400,"padded storage");
}
// The 3-bit allocation spends the whole budget, never stores more than code_bits or the 2/4-bit allocation, and
// must predict less distortion. On a flat spectrum at 2 bits/dim, one padded 3-bit coordinate falls back.
void padded_allocation_test() {
  constexpr int ranks=200;
  AllocationModel model{std::vector<double>(ranks),std::vector<uint32_t>(ranks,max_code_bits),{}};
  model.error[0]=1;
  for (unsigned b=1;b<=max_code_bits;++b) model.error[b]=distortion(gaussian_codebooks()[b]);
  model.error[split_code_bits]=distortion(additive_three_codebook());
  for (int rank=0;rank<ranks;++rank) model.variance[rank]=std::exp(-rank/double(ranks));
  size_t additive=0;
  for (uint32_t code_bits=ranks;code_bits<=4*ranks;code_bits+=ranks/8) for (uint32_t odd:{0u,1u,3u}) {
    const uint32_t budget=code_bits-odd;
    const auto dense=greedy_allocation(model,budget,false);
    const auto fallback=has_three_bit_tier(dense) ? greedy_allocation(model,budget,true) : dense;
    if (widest_tier(dense)>lookup_code_bits) continue;
    const auto chosen=fast_lookup_allocation(model,budget,dense);
    require(chosen.additive_three==has_three_bit_tier(chosen.bits),"invalid padded allocation");
    if (!chosen.additive_three) { require(chosen.bits==fallback,"2/4-bit fallback changed"); continue; }
    const bool nibble_fallback=widest_tier(fallback)<=lookup_code_bits;
    require(widest_tier(chosen.bits)<=lookup_code_bits,"3-bit allocation left the nibble layout");
    require(chosen.bits==greedy_allocation(model,budget,false),"3-bit allocation trimmed its budget");
    require(lookup_stored_bits(chosen.bits,true)<=std::max(budget,nibble_fallback ? lookup_stored_bits(fallback,false) : 0),"padded allocation exceeds the budget");
    require(predicted_distortion(model,chosen.bits)<predicted_distortion(model,fallback),"3-bit tiers kept without gain");
    ++additive;
  }
  require(additive>=ranks/16,"padded allocation rarely keeps 3-bit tiers");
  std::fill(model.variance.begin(),model.variance.end(),1.0);
  const auto flat=fast_lookup_allocation(model,2*ranks,greedy_allocation(model,2*ranks,false));
  require(!flat.additive_three && flat.bits==Allocation(ranks,2),"2 bits/dim does not fall back to 2-bit tiers");
}
// Fast targets with every tier <= 4 bits use additive 3-bit tiers; training predicts their padded storage.
void fast_target_allocation_test() {
  size_t with_three=0;
  for (uint32_t code_bits:{72u,80u,88u,96u,112u}) {
    const auto fast=train(code_bits,TieredQIndexMode::fast),full=train(code_bits,TieredQIndexMode::full);
    require(widest(fast)<=lookup_code_bits,"fast allocation left the nibble layout");
    require(fast.config().code_bits<=code_bits,"fast allocation exceeds the budget");
    TieredQIndex index(fast);
    Allocation bits;
    for (const auto& tier:fast.tiers()) bits.insert(bits.end(),tier.dimensions,tier.bits);
    require(!nibble_layout || stored_code_bits(index)==lookup_stored_bits(bits,true),"training mispredicts padded fast storage");
    require(persisted_three_levels(full)==gaussian_codebooks()[split_code_bits],"full target changed the 3-bit codebook");
    if (!has_three(fast)) continue;
    ++with_three;
    require(persisted_three_levels(fast)==additive_three_codebook(),"fast 3-bit tiers do not use the additive codebook");
    require(fast.config().code_bits==code_bits,"3-bit allocation trimmed its budget");
  }
  require(with_three>=3,"fast target allocates too few 3-bit tiers");
}
// Additive tiers: fast scores track exact full-mode scores; append, save and load keep fast scores exact.
void additive_index_test() {
  const auto codec=train(80,TieredQIndexMode::fast);
  require(has_three(codec),"parity model has no 3-bit tier");
  const auto data=spectrum_rows(rows,11);
  constexpr size_t first=rows/3;
  TieredQIndex fast(codec),split(codec),full(codec,TieredQIndexMode::full);
  fast.add(data.data(),rows); full.add(data.data(),rows);
  split.add(data.data(),first); split.add(data.data()+first*dimensions,rows-first);
  require(mean_overlap(fast,full,data)>=min_overlap,"additive fast scan disagrees with full mode");
  for (size_t q=0;q<queries;++q) require_same(split.search(data.data()+q*dimensions,k),fast.search(data.data()+q*dimensions,k),"append changes additive fast scores");
  const auto path=std::filesystem::temp_directory_path()/("tieredq-additive-"+std::to_string(getpid()));
  fast.save(path.string());
  auto loaded=TieredQIndex::load(path.string());
  require(loaded.bytes_per_vector()==fast.bytes_per_vector(),"reload changes additive storage");
  for (size_t q=0;q<queries;++q) require_same(loaded.search(data.data()+q*dimensions,k),fast.search(data.data()+q*dimensions,k),"reload changes additive fast scores");
  loaded.add(data.data(),first); fast.add(data.data(),first);
  for (size_t q=0;q<queries;++q) require_same(loaded.search(data.data()+q*dimensions,k),fast.search(data.data()+q*dimensions,k),"append after reload changes additive scores");
  std::filesystem::remove(path);
}
// Models that keep the Lloyd-Max 3-bit codebook (full-target or pre-additive) still serve fast mode with
// one padded nibble per 3-bit code.
void lloyd_max_fast_test() {
  const auto codec=TieredQ::deserialize(train(80,TieredQIndexMode::full).serialize());
  require(has_three(codec),"full-target model has no 3-bit tier");
  LookupWidthCounts widths{};
  for (const auto& tier:codec.tiers()) widths[tier.bits]+=tier.dimensions;
  const size_t groups=lookup_group_count(widths,false);
  const auto data=spectrum_rows(rows,13);
  TieredQIndex fast(codec),full(codec,TieredQIndexMode::full);
  fast.add(data.data(),rows); full.add(data.data(),rows);
  require(!nibble_layout || widest(codec)>lookup_code_bits ||
      fast.bytes_per_vector()==lookup_group_bytes(groups)+TieredQ::metadata_bytes,"Lloyd-Max 3-bit codes are not padded");
  require(mean_overlap(fast,full,data)>=min_overlap,"Lloyd-Max fast scan disagrees with full mode");
  const auto path=std::filesystem::temp_directory_path()/("tieredq-lloyd-max-"+std::to_string(getpid()));
  fast.save(path.string());
  const auto loaded=TieredQIndex::load(path.string());
  for (size_t q=0;q<queries;++q) require_same(loaded.search(data.data()+q*dimensions,k),fast.search(data.data()+q*dimensions,k),"reload changes Lloyd-Max fast scores");
  std::filesystem::remove(path);
}
} // namespace
void tieredq_additive_tests() {
  split_derivation_test();
  padded_allocation_test();
  fast_target_allocation_test();
  additive_index_test();
  lloyd_max_fast_test();
  std::cout<<"PASS additive 3-bit tiers\n";
}
