// Production-kernel phase breakdown: decode-only vs accumulate-only per 16-doc block.
#include "tieredq_scan_codes_x86.h"
#include <chrono>
#include <cstdio>
#include <random>
using namespace dv::tieredq_detail;
using Clock=std::chrono::steady_clock;
int main(int argc,char** argv){
  const size_t dims=std::atoi(argv[1]),docs=65536;
  FastScanCache c;c.count=docs;c.dimensions=dims;c.level_max=fast_code_lookup_levels;c.groups=(dims+3)/4;c.blocks=docs/16;
  std::mt19937 rng(1);std::normal_distribution<float> n;
  c.codebooks.assign(1,std::vector<float>(256));for(auto&v:c.codebooks[0]) v=n(rng);
  c.tier_of.assign(dims,0);c.levels.resize(c.blocks*c.groups*64);for(auto&v:c.levels) v=int8_t(rng());
  c.factor.assign(docs,1);c.correction.assign(docs,1);c.centroid.assign(docs,0);
  std::vector<dv::TieredQQuery> qs(32);for(auto&q:qs){q.weights.resize(dims);for(auto&w:q.weights)w=n(rng);q.centroid_scores={0};q.correction_weights={0};}
  auto segs=code_table_segments(c);
  const CentroidTiles C(32,[&](size_t r)->const dv::TieredQQuery&{return qs[r];});std::vector<float> dec(dims*16),out(32*16);
  const CodeScanWeights W(dims,32,8,[&](size_t r)->const dv::TieredQQuery&{return qs[r];});
  auto t0=Clock::now();
  for(int rep=0;rep<3;++rep) for(size_t b=0;b<c.blocks;++b) decode_code_block_avx512(c,segs,b,dec.data());
  double d=std::chrono::duration<double>(Clock::now()-t0).count()/3;
  t0=Clock::now();
  for(int rep=0;rep<3;++rep) for(size_t b=0;b<c.blocks;++b) for(size_t t=0;t<4;++t)
    accumulate_codes_avx512<8>(c,dec.data(),W.tile(t),C.scores(t),C.corrections(t),0,16,out.data(),16);
  double a=std::chrono::duration<double>(Clock::now()-t0).count()/3;
  const double ghz=3.19,ranks=double(c.blocks)*dims;
  std::printf("dims %zu decode %.2f cyc/rank-block  accumulate %.2f cyc/(rank-block-row)  32-row total %.3f ns/doc-dim-query\n",dims,d*ghz*1e9/ranks,a*ghz*1e9/(ranks*32),(d+a)*1e9/(double(docs)*dims*32));
}
