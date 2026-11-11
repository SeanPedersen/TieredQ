// Issue #9: wall time of exact_lookup_allocation on synthetic power-law spectra (variance ~ (rank+1)^-decay,
// raised to the shipped allocation power) at 1-4 bits/dim. Reports the median of several runs in milliseconds.
// Usage: allocation_time [D...] (default 3072 4096).
#include "tieredq_train_allocation.h"
#include "tieredq_train_codebook.h"
#include "tieredq_train_codebook_data.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace tq::tieredq_detail;
constexpr int runs=9;
constexpr double decay=0.8,power=1.5;
int main(int argc,char** argv) {
  std::vector<uint32_t> dims;
  for (int i=1;i<argc;++i) dims.push_back(uint32_t(std::atoi(argv[i])));
  if (dims.empty()) dims={3072,4096};
  DistortionTable error{1};
  for (unsigned b=1;b<=max_code_bits;++b) error[b]=distortion(gaussian_codebooks()[b]);
  error[split_code_bits]=distortion(additive_three_codebook());
  std::printf("d,bits_per_dim,median_ms,max_ms,code_bits\n");
  for (uint32_t d:dims) for (uint32_t rate=1;rate<=4;++rate) {
    std::vector<double> variance(d);
    for (uint32_t r=0;r<d;++r) variance[r]=std::pow(std::pow(r+1.0,-decay),power);
    std::vector<double> ms;
    uint32_t bits=0;
    for (int run=0;run<runs;++run) {
      const auto start=std::chrono::steady_clock::now();
      const auto allocation=exact_lookup_allocation(variance,error,rate*d);
      ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
      bits=0; for (unsigned b:allocation.bits) bits+=b;
    }
    std::sort(ms.begin(),ms.end());
    std::printf("%u,%u,%.3f,%.3f,%u\n",d,rate,ms[runs/2],ms.back(),bits);
  }
}
