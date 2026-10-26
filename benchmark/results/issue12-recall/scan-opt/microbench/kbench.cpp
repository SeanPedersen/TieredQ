// Single-thread kernel microbenchmark: baseline code-ID scan vs register-resident variants.
#include "tieredq_scan_codes_x86.h"
#include "codes_new.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

using namespace dv::tieredq_detail;
using Clock=std::chrono::steady_clock;

static FastScanCache make_cache(size_t dims,size_t docs,size_t tiers) {
  FastScanCache c;
  c.count=docs;c.dimensions=dims;c.level_max=fast_code_lookup_levels;
  c.groups=(dims+3)/4;c.blocks=(docs+15)/16;
  std::mt19937 rng(1);std::normal_distribution<float> n;
  c.codebooks.resize(tiers);
  for (auto& b:c.codebooks) { b.resize(256); for (auto& v:b) v=n(rng); }
  c.tier_of.assign(dims,0);
  for (size_t r=0;r<dims;++r) c.tier_of[r]=uint8_t(r*tiers/dims);
  c.levels.resize(c.blocks*c.groups*64);
  for (size_t doc=0;doc<docs;++doc) for (size_t r=0;r<dims;++r)
    c.levels[fast_level_position(c.groups,doc,r)]=int8_t(rng()&255);
  const size_t padded=c.blocks*16;
  c.factor.resize(padded);c.correction.resize(padded);c.centroid.resize(padded);
  for (size_t i=0;i<docs;++i) { c.factor[i]=n(rng);c.correction[i]=n(rng);c.centroid[i]=uint8_t(rng()%4); }
  return c;
}
static std::vector<dv::TieredQQuery> make_queries(size_t dims,size_t rows) {
  std::mt19937 rng(2);std::normal_distribution<float> n;
  std::vector<dv::TieredQQuery> qs(rows);
  for (auto& q:qs) {
    q.weights.resize(dims);for (auto& w:q.weights) w=n(rng);
    q.centroid_scores.resize(4);q.correction_weights.resize(4);
    for (auto& v:q.centroid_scores) v=n(rng);
    for (auto& v:q.correction_weights) v=n(rng);
  }
  return qs;
}

template<class F> double time_it(F f,int reps) {
  f();
  double best=1e30;
  for (int i=0;i<reps;++i) {
    auto t=Clock::now();f();
    best=std::min(best,std::chrono::duration<double>(Clock::now()-t).count());
  }
  return best;
}

int main(int argc,char** argv) {
  const size_t dims=std::atoi(argv[1]),docs=std::atoi(argv[2]),tiers=argc>3?std::atoi(argv[3]):1;
  const size_t tile=4096;
  auto cache=make_cache(dims,docs,tiers);
  constexpr size_t max_rows=32;
  auto qs=make_queries(dims,max_rows);
  std::vector<float> ref(max_rows*tile),out(max_rows*tile);
  auto at=[&](size_t r)->const dv::TieredQQuery&{return qs[r];};
  const double ghz_ns=1e9;
  // Reference using baseline per 4-row tile over full docs; report ns per (doc*rank*query).
  auto run_old=[&](size_t rows){
    for (size_t b=0;b<docs;b+=tile) {
      const size_t n=std::min(tile,docs-b);
      for (size_t r0=0;r0<rows;r0+=4)
        scan_codes_avx512_tile(cache,std::min<size_t>(4,rows-r0),b,n,ref.data()+r0*tile,tile,[&](size_t r)->const dv::TieredQQuery&{return qs[r0+r];});
    }
  };
  auto report=[&](const char* name,size_t rows,double s){
    std::printf("%-22s rows=%2zu  %.3f ns/(doc*dim*query)  %.1f ms\n",name,rows,s*ghz_ns/(double(docs)*dims*rows),s*1e3);
  };
  for (size_t rows:{1,4}) report("baseline(tile4)",rows,time_it([&]{run_old(rows);},5));
  auto check=[&](size_t rows){
    for (size_t r=0;r<rows;++r) for (size_t i=0;i<std::min(tile,docs);++i)
      if (ref[r*tile+i]!=out[r*tile+i]) { std::printf("MISMATCH r=%zu i=%zu %g %g\n",r,i,ref[r*tile+i],out[r*tile+i]); std::exit(1); }
  };
#define VARIANT(R,B) { \
    auto plan=bench::make_plan(cache,R,at); \
    auto f=[&]{ for (size_t b=0;b<docs;b+=tile) bench::kernel512<R,B>(cache,plan,b,std::min(tile,docs-b),out.data(),tile,at); }; \
    double s=time_it(f,5); \
    for (size_t r0=0;r0<R;r0+=4) scan_codes_avx512_tile(cache,std::min<size_t>(4,R-r0),0,std::min(tile,docs),ref.data()+r0*tile,tile,[&](size_t r)->const dv::TieredQQuery&{return qs[r0+r];}); \
    for (size_t b=0;b<docs;b+=tile) { bench::kernel512<R,B>(cache,plan,b,std::min(tile,docs-b),out.data(),tile,at); break; } \
    check(R); report("new R" #R " B" #B,R,s); }
  VARIANT(1,4) VARIANT(4,1) VARIANT(8,1) VARIANT(16,1)
#define TVARIANT(R,B) { \
    auto plan=bench::make_table_plan(cache,R,at); \
    auto f=[&]{ for (size_t b=0;b<docs;b+=tile) bench::table512<R,B>(cache,plan,b,std::min(tile,docs-b),out.data(),tile,at); }; \
    double s=time_it(f,5); \
    for (size_t r0=0;r0<R;r0+=4) scan_codes_avx512_tile(cache,std::min<size_t>(4,R-r0),0,std::min(tile,docs),ref.data()+r0*tile,tile,[&](size_t r)->const dv::TieredQQuery&{return qs[r0+r];}); \
    bench::table512<R,B>(cache,plan,0,std::min(tile,docs),out.data(),tile,at); \
    check(R); report("table R" #R " B" #B,R,s); }
  TVARIANT(1,1) TVARIANT(4,1) TVARIANT(8,1) TVARIANT(12,1) TVARIANT(16,1)
#define DVARIANT(R,T) { \
    std::vector<bench::TablePlan> plans; std::vector<float> dec; \
    for (size_t t=0;t<T;++t) plans.push_back(bench::make_table_plan(cache,R,[&](size_t r)->const dv::TieredQQuery&{return qs[t*R+r];})); \
    auto f=[&]{ for (size_t b=0;b<docs;b+=tile) bench::decoded512<R>(cache,plans,b,std::min(tile,docs-b),out.data(),tile,at,dec); }; \
    double s=time_it(f,5); \
    for (size_t r0=0;r0<R*T;r0+=4) scan_codes_avx512_tile(cache,std::min<size_t>(4,R*T-r0),0,std::min(tile,docs),ref.data()+r0*tile,tile,[&](size_t r)->const dv::TieredQQuery&{return qs[r0+r];}); \
    bench::decoded512<R>(cache,plans,0,std::min(tile,docs),out.data(),tile,at,dec); \
    check(R*T); report("decoded R" #R " T" #T,R*T,s); }
  DVARIANT(1,1) DVARIANT(4,1) DVARIANT(8,1) DVARIANT(16,1) DVARIANT(8,4) DVARIANT(16,2) DVARIANT(12,1)
  { auto plan=bench::make_table_plan(cache,16,at); double s=time_it([&]{ for (int t=0;t<2;++t) for (size_t b=0;b<docs;b+=tile) bench::table512<16,1>(cache,plan,b,std::min(tile,docs-b),out.data(),tile,at); },5); report("table R16 x2 tiles",32,s); }
}
