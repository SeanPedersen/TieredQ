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
std::vector<float> persisted_levels(const TieredQ& codec,unsigned bits=split_code_bits) {
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
  for (unsigned b=0;b<=bits;++b) book=floats();
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
constexpr double max_wide_distortion_ratio=1.5;
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
// Exact allocation stays within the padded storage of the 2/4-bit rule and never predicts more distortion than its
// 2/4-bit fallback or than a dense greedy allocation whose padded size also fits; mildly decaying spectra keep
// additive 3-bit tiers. On a flat spectrum at 2 bits/dim uniform 2-bit tiers win.
// Wide canonical books: every sorted level equals its parts' sum, parts fit nibble streams and codes are a bijection.
void wide_split_test() {
  for (uint32_t bits=lookup_code_bits+1;bits<=widest_code_bits;++bits) {
    const auto& book=additive_wide_codebook(bits);
    require(book.size()==size_t(1)<<bits && std::is_sorted(book.begin(),book.end()),"wide book is not sorted");
    std::vector<std::vector<float>> levels(widest_code_bits+1);
    levels[bits]=book;
    const auto split=additive_splits(levels)[bits];
    require(split.has_value(),"canonical wide book does not split");
    uint32_t total=0;
    for (uint32_t part:split->part_bits) { require(part==1 || part==2 || part==lookup_code_bits,"part wider than a stream"); total+=part; }
    require(total==bits,"parts do not cover the code");
    std::vector<bool> seen(book.size(),false);
    for (size_t code=0;code<book.size();++code) {
      uint32_t packed=split->code[code]; float sum=0;
      require(!seen[packed],"split codes collide"); seen[packed]=true;
      for (size_t part=0;part<split->part_bits.size();++part) {
        sum+=split->part_levels[part][packed&((1u<<split->part_bits[part])-1)];
        packed>>=split->part_bits[part];
      }
      require(sum==book[code],"split does not reproduce the level");
    }
    require(distortion(book)<distortion(gaussian_codebooks()[bits])*max_wide_distortion_ratio,"wide book distortion regressed");
    levels[bits]=gaussian_codebooks()[bits];
    require(!additive_splits(levels)[bits],"Lloyd-Max wide book splits");
  }
}
// Fast-target wide models use additive books and approximate nibble scores close to the codec's exact ones; both
// survive persistence unchanged.
void wide_index_test() {
  constexpr uint32_t dimensions=64,wide_bits=4*dimensions,centroids=4;
  constexpr size_t training_rows=512,documents=2048,k=10;
  constexpr double min_overlap=0.8;
  std::mt19937 random(47); std::normal_distribution<float> normal;
  std::vector<float> training(training_rows*dimensions),rows(documents*dimensions),query(dimensions);
  for (auto* values:{&training,&rows,&query}) for (size_t i=0;i<values->size();++i)
    (*values)[i]=normal(random)*std::exp(-float(i%dimensions)/8);
  TieredQConfig config{dimensions,wide_bits,centroids,true}; config.fine_tiers=TieredQFineTiers::off;
  const auto codec=TieredQ::train(config,training.data(),training_rows);
  require(codec.tiers().front().bits>lookup_code_bits,"wide test needs a wide tier");
  require(persisted_levels(codec,codec.tiers().front().bits)==additive_wide_codebook(codec.tiers().front().bits),"fast wide tier is not additive");
  TieredQIndex fast(codec,TieredQIndexMode::fast),full(codec,TieredQIndexMode::full);
  fast.add(rows.data(),documents); full.add(rows.data(),documents);
  const auto approximate=fast.search(query.data(),k),exact=full.search(query.data(),k);
  size_t overlap=0;
  for (const auto& hit:approximate) overlap+=std::any_of(exact.begin(),exact.end(),[&](const auto& other){return other.id==hit.id;});
  require(double(overlap)>=min_overlap*k,"nibble scores lost the exact top hits");
  const auto path=std::filesystem::temp_directory_path()/("tieredq-wide-"+std::to_string(getpid()));
  fast.save(path.string());
  const auto loaded=TieredQIndex::load(path.string());
  std::filesystem::remove(path);
  require_same(fast.search(query.data(),k),loaded.search(query.data(),k),"wide nibble index changed on reload");
  // Single-query and IVF searches prune by default; the exact bound keeps the unpruned hits.
  std::vector<float> pair(2*dimensions);
  std::copy(query.begin(),query.end(),pair.begin()); std::copy(rows.begin(),rows.begin()+dimensions,pair.begin()+dimensions);
  std::vector<float> scores(2*k); std::vector<uint32_t> ids(2*k);
  fast.search_batch_fast(pair.data(),2,k,scores.data(),ids.data());
  const auto single=fast.search(query.data(),k);
  for (size_t hit=0;hit<k;++hit) require(single[hit].id==ids[hit] && single[hit].score==scores[hit],"pruned flat search changed hits");
  TieredQIVFIndex ivf(codec,centroids); ivf.build(rows.data(),documents);
  const auto unpruned=ivf.search(query.data(),k,centroids,TieredQPruning{0,uint32_t(codec.tiers().size())});
  require_same(ivf.search(query.data(),k,centroids),unpruned,"pruned IVF search changed hits");
}
void padded_allocation_test() {
  constexpr int ranks=200;
  DistortionTable error{1};
  for (unsigned b=1;b<=widest_code_bits;++b) error[b]=distortion(gaussian_codebooks()[b]);
  error[split_code_bits]=distortion(additive_three_codebook());
  AllocationModel model{std::vector<double>(ranks),std::vector<uint32_t>(ranks,widest_code_bits),std::vector(ranks,error)};
  for (int rank=0;rank<ranks;++rank) model.variance[rank]=std::exp(-rank/double(ranks));
  size_t additive=0;
  for (uint32_t code_bits=ranks;code_bits<=4*ranks;code_bits+=ranks/8) for (uint32_t odd:{0u,1u,3u}) {
    const uint32_t budget=code_bits-odd;
    const auto dense=greedy_allocation(model,budget,false);
    if (widest_tier(dense)>lookup_code_bits) continue;
    const auto fallback=lookup_fallback(model,budget,dense);
    const uint32_t limit=lookup_storage_limit(fallback,budget);
    const auto exact=exact_lookup_allocation(model.variance,error,limit);
    require(exact.additive_three==has_three_bit_tier(exact.bits),"invalid padded allocation");
    require(lookup_stored_bits(exact.bits,exact.additive_three)<=limit,"exact allocation exceeds its storage");
    const double predicted=predicted_distortion(model,exact.bits)*(1-1e-12);
    require(widest_tier(fallback)>lookup_code_bits || predicted<=predicted_distortion(model,fallback),"exact allocation loses to the 2/4-bit fallback");
    require(lookup_stored_bits(dense,true)>limit || predicted<=predicted_distortion(model,dense),"exact allocation loses to greedy");
    additive+=exact.additive_three;
  }
  require(additive>=ranks/16,"padded allocation rarely keeps 3-bit tiers");
  std::fill(model.variance.begin(),model.variance.end(),1.0);
  const auto flat=exact_lookup_allocation(model.variance,error,2*ranks);
  require(!flat.additive_three && flat.bits==Allocation(ranks,2),"2 bits/dim does not keep 2-bit tiers");
}
// Reference for exact_lookup_allocation: enumerates the 4-, 3- and 2-bit counts over ranks sorted by variance and
// fills the remaining padded 1-bit slots (the #9 experiment allocator, O(d^2 * groups)).
double enumerated_distortion(const std::vector<double>& variance,const DistortionTable& error,uint32_t limit_bits) {
  const uint32_t d=uint32_t(variance.size()),groups=uint32_t(limit_bits/CHAR_BIT*lookup_pair_groups);
  auto sorted=variance;
  for (auto& v:sorted) v=std::max(0.0,v);
  std::sort(sorted.rbegin(),sorted.rend());
  std::vector<double> prefix(d+1,0);
  for (uint32_t r=0;r<d;++r) prefix[r+1]=prefix[r]+sorted[r];
  auto span=[&](uint32_t begin,uint32_t count) { return prefix[begin+count]-prefix[begin]; };
  double best=INFINITY;
  for (uint32_t four=0;four<=std::min(d,groups);++four)
    for (uint32_t three=0;three<=d-four;++three)
      for (uint32_t two=0;two+three+four<=d;++two) {
        const uint32_t fixed=four+ceil_div(two+three,2);
        if (fixed>groups) break;
        if ((groups-fixed)*4<three) continue;
        const uint32_t coded=four+three+two,one=std::min(d-coded,(groups-fixed)*4-three);
        best=std::min(best,error[4]*span(0,four)+error[3]*span(four,three)+error[2]*span(four+three,two)+
                           error[1]*span(coded,one)+error[0]*span(coded+one,d-coded-one));
      }
  return best;
}
// The exact allocator matches the enumerator on random spectra (with ties and zero variances) at every padded limit.
void exact_allocation_test() {
  DistortionTable error{1};
  for (unsigned b=1;b<=widest_code_bits;++b) error[b]=distortion(gaussian_codebooks()[b]);
  error[split_code_bits]=distortion(additive_three_codebook());
  std::mt19937_64 rng(21);
  size_t cases=0;
  for (int trial=0;trial<300;++trial) {
    const uint32_t d=1+uint32_t(rng()%40);
    std::vector<double> variance(d);
    std::exponential_distribution<double> exponential(1.0/(1+trial%4));
    for (auto& v:variance) v=trial%5==0 ? double(rng()%3) : std::exp(-exponential(rng));
    AllocationModel model{variance,std::vector<uint32_t>(d,widest_code_bits),std::vector(d,error)};
    for (uint32_t limit=CHAR_BIT;limit<=4*d+CHAR_BIT;limit+=1+uint32_t(rng()%3)) {
      const auto exact=exact_lookup_allocation(variance,error,limit);
      require(exact.bits.size()==d && widest_tier(exact.bits)<=lookup_code_bits,"exact allocation left the nibble layout");
      require(lookup_stored_bits(exact.bits,exact.additive_three)<=std::max(limit/CHAR_BIT*CHAR_BIT,0u),"exact allocation exceeds its storage");
      const double reference=enumerated_distortion(variance,error,limit);
      require(std::fabs(predicted_distortion(model,exact.bits)-reference)<=1e-12*(1+reference),"exact allocation is not optimal");
      ++cases;
    }
  }
  require(cases>1000,"too few exact allocation cases");
}
// Fast targets with every tier <= 4 bits use additive 3-bit tiers; training predicts their padded storage. The budget
// bounds padded storage under the 2/4-bit rule, whose padding adds at most an unpaired 2-bit and three 1-bit slots,
// and the realised code bits fit that storage.
void fast_target_allocation_test() {
  size_t with_three=0;
  for (uint32_t code_bits:{72u,80u,88u,96u,112u}) {
    const auto fast=train(code_bits,TieredQIndexMode::fast),full=train(code_bits,TieredQIndexMode::full);
    require(widest(fast)<=lookup_code_bits,"fast allocation left the nibble layout");
    TieredQIndex index(fast);
    constexpr uint32_t rule_padding=2+3;
    require(!nibble_layout || stored_code_bits(index)<=ceil_div(code_bits+rule_padding,CHAR_BIT)*CHAR_BIT,"fast allocation exceeds the budget");
    Allocation bits;
    for (const auto& tier:fast.tiers()) bits.insert(bits.end(),tier.dimensions,tier.bits);
    require(!nibble_layout || stored_code_bits(index)==lookup_stored_bits(bits,true),"training mispredicts padded fast storage");
    require(!nibble_layout || fast.config().code_bits<=stored_code_bits(index),"realised code bits exceed padded storage");
    require(persisted_levels(full)==gaussian_codebooks()[split_code_bits],"full target changed the 3-bit codebook");
    if (!has_three(fast)) continue;
    ++with_three;
    require(persisted_levels(fast)==additive_three_codebook(),"fast 3-bit tiers do not use the additive codebook");
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
  wide_split_test();
  wide_index_test();
  padded_allocation_test();
  exact_allocation_test();
  fast_target_allocation_test();
  additive_index_test();
  lloyd_max_fast_test();
  std::cout<<"PASS additive 3-bit tiers\n";
}
