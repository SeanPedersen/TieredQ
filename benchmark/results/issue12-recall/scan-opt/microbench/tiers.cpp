#include <tieredq/tieredq.h>
#include <cmath>
#include <cstdio>
#include <random>
int main(){ for (float decay:{0.25f,0.5f}) for (unsigned bits:{5u,6u}) { std::mt19937 r(123);std::normal_distribution<float> n; std::vector<float> t(128*13); for(size_t i=0;i<t.size();++i) t[i]=n(r)*std::exp(-decay*float(i%13));
 auto c=dv::TieredQ::train({13,13*bits,4,true},t.data(),128); std::printf("decay %.2f bits %u:",decay,bits); for(auto&x:c.tiers()) std::printf(" %ux%u",x.dimensions,x.bits); std::printf("\n");}}
