// Single-thread query_weights cost by part: full call vs projection-only estimate.
#include <tieredq/tieredq.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
using Clock=std::chrono::steady_clock;
int main(int argc,char** argv){
  const uint32_t dims=std::atoi(argv[1]);const size_t nq=64;
  std::mt19937 rng(7);std::normal_distribution<float> n;
  std::vector<float> train(8192*dims),qs(nq*dims);for(auto&v:train)v=n(rng);for(auto&v:qs)v=n(rng);
  auto codec=dv::TieredQ::train({dims,dims*8,256,true,8},train.data(),8192);
  for(int rep=0;rep<2;++rep){auto t=Clock::now();float sink=0;
    for(size_t q=0;q<nq;++q){auto w=codec.query_weights(qs.data()+q*dims);sink+=w.weights[0];}
    std::printf("dims %u query_weights %.3f ms/query (%g)\n",dims,std::chrono::duration<double,std::milli>(Clock::now()-t).count()/nq,sink);}
#ifdef BATCHED
  for(int rep=0;rep<2;++rep){auto t=Clock::now();auto w=codec.query_weights(qs.data(),nq);
    std::printf("dims %u batched query_weights %.3f ms/query (%g)\n",dims,std::chrono::duration<double,std::milli>(Clock::now()-t).count()/nq,w[0].weights[0]);}
#endif
}
