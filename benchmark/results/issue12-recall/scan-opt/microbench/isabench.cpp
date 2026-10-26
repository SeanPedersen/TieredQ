// Same source against baseline or new headers: single-thread code-ID scan cost per ISA and query count.
#include "tieredq_scan_codes_x86.h"
#include <chrono>
#include <cstdio>
#include <random>
using namespace dv::tieredq_detail;
using Clock=std::chrono::steady_clock;
int main(int argc,char** argv){
  const size_t dims=std::atoi(argv[1]),docs=65536,tile=4096,centroids=256;
  FastScanCache c;c.count=docs;c.dimensions=dims;c.level_max=fast_code_lookup_levels;c.groups=(dims+3)/4;c.blocks=docs/16;
  std::mt19937 rng(1);std::normal_distribution<float> n;
  c.codebooks.assign(1,std::vector<float>(256));for(auto&v:c.codebooks[0]) v=n(rng);
  c.tier_of.assign(dims,0);c.levels.resize(c.blocks*c.groups*64);for(auto&v:c.levels) v=int8_t(rng());
  c.factor.assign(docs,1);c.correction.assign(docs,1);c.centroid.resize(docs);for(auto&v:c.centroid) v=uint8_t(rng());
  std::vector<dv::TieredQQuery> qs(32);for(auto&q:qs){q.weights.resize(dims);for(auto&w:q.weights)w=n(rng);q.centroid_scores.resize(centroids);q.correction_weights.resize(centroids);for(auto&v:q.centroid_scores)v=n(rng);for(auto&v:q.correction_weights)v=n(rng);}
  std::vector<float> out(32*tile);
  for (int isa=0;isa<2;++isa) for (size_t rows:{1,8,32}) {
#ifdef BASELINE
    constexpr size_t chunk=4;
#else
    constexpr size_t chunk=32;
#endif
    auto run=[&]{ for(size_t b=0;b<docs;b+=tile) for(size_t r0=0;r0<rows;r0+=chunk) {
      const size_t m=std::min(chunk,rows-r0);
      auto at=[&](size_t r)->const dv::TieredQQuery&{return qs[r0+r];};
      if (isa) scan_codes_avx512_tile(c,m,b,tile,out.data()+r0*tile,tile,at); else scan_codes_avx2_tile(c,m,b,tile,out.data()+r0*tile,tile,at);} };
    run(); double best=1e9; for(int i=0;i<3;++i){auto t=Clock::now();run();best=std::min(best,std::chrono::duration<double>(Clock::now()-t).count());}
    std::printf("%s rows=%2zu %.3f ns/(doc*dim*query)\n",isa?"avx512":"avx2  ",rows,best*1e9/(double(docs)*dims*rows));
  }
}
