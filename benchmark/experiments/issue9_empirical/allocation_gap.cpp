// Issue #9: how far is the greedy allocator from the exact optimum of its own objective (predicted distortion)?
// Captures the real PCA spectrum by training once on the ab_eval seed-0 sample (DV_TIEREDQ_TRAIN_DUMP), then per budget:
//  - dense: greedy vs exact DP over Lloyd-Max widths 0..8 under sum(bits) <= budget (no padding);
//  - fast: the shipped fast choice vs exact_lookup_allocation under padded nibble storage, both at the stored bits
//    the shipped choice uses (equal memory) and at the nominal budget.
// Distortion is reported as a fraction of total variance. Usage: allocation_gap DATA_DIR D BUDGET...
// Pinned: build against 8762da6 (-DISSUE9_GAP=ON); main no longer has fast_lookup_allocation.
#include "tieredq/tieredq.h"
#include "tieredq_train_allocation.h"
#include "tieredq_train_codebook.h"
#include "tieredq_train_codebook_data.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
using namespace dv::tieredq_detail;
template<class T> std::vector<T> load(const std::string& path) {
  std::ifstream in(path,std::ios::binary|std::ios::ate);
  std::vector<T> out(size_t(in.tellg())/sizeof(T));
  in.seekg(0); in.read(reinterpret_cast<char*>(out.data()),std::streamsize(out.size()*sizeof(T)));
  return out;
}
constexpr size_t training_limit=32768;
constexpr uint32_t centroids=256,bits_per_byte=8;
// Descending residual PCA variances of the model ab_eval trains for seed 0.
std::vector<double> spectrum(const std::string& dir,uint32_t d) {
  const auto base=load<float>(dir+"/base.f32");
  const size_t n=base.size()/d,rows=std::min(n,training_limit);
  std::vector<size_t> order(n); std::iota(order.begin(),order.end(),0);
  std::mt19937_64 rng(0); std::shuffle(order.begin(),order.end(),rng);
  std::vector<float> sample(rows*d);
  for (size_t i=0;i<rows;++i) std::copy_n(base.data()+order[i]*d,d,sample.data()+i*d);
  const std::string prefix="/tmp/issue9_spectrum_"+std::to_string(d);
  setenv("DV_TIEREDQ_TRAIN_DUMP",prefix.c_str(),1);
  auto config=dv::TieredQConfig::target_code_bits(d,d); config.num_centroids=centroids;
  dv::TieredQ::train(config,sample.data(),rows,0);
  unsetenv("DV_TIEREDQ_TRAIN_DUMP");
  auto ascending=load<double>(prefix+".eigenvalues.f64");
  return {ascending.rbegin(),ascending.rend()};
}
double distortion_of(const std::vector<double>& variance,const DistortionTable& error,const Allocation& bits) {
  double total=0; for (size_t r=0;r<bits.size();++r) total+=std::max(0.0,variance[r])*error[bits[r]];
  return total;
}
// Exact min of sum variance*error[bits] under sum(bits) <= budget, widths 0..max_code_bits.
double dense_optimum(const std::vector<double>& variance,const DistortionTable& error,uint32_t budget) {
  std::vector<double> best(budget+1,0.0),next(budget+1);
  for (double v:variance) {
    v=std::max(0.0,v);
    for (uint32_t used=0;used<=budget;++used) {
      double value=INFINITY;
      for (uint32_t b=0;b<=max_code_bits && b<=used;++b) value=std::min(value,best[used-b]+v*error[b]);
      next[used]=value;
    }
    std::swap(best,next);
  }
  return best[budget];
}
bool convex(const DistortionTable& error,const std::vector<unsigned>& widths) {
  for (size_t i=1;i+1<widths.size();++i) {
    const double left=(error[widths[i-1]]-error[widths[i]])/(widths[i]-widths[i-1]);
    const double right=(error[widths[i]]-error[widths[i+1]])/(widths[i+1]-widths[i]);
    if (right>left) return false;
  }
  return true;
}
int main(int argc,char** argv) {
  if (argc<4) { std::fprintf(stderr,"usage: %s DATA_DIR D BUDGET...\n",argv[0]); return 2; }
  const std::string dir=argv[1]; const uint32_t d=std::stoul(argv[2]);
  DistortionTable lloyd{1};
  for (unsigned b=1;b<=max_code_bits;++b) lloyd[b]=distortion(gaussian_codebooks()[b]);
  auto additive=lloyd; additive[split_code_bits]=distortion(additive_three_codebook());
  std::printf("# convex: lloyd 0..8 %d, additive 0..4 %d, error additive3 %.5f lloyd3 %.5f\n",
              convex(lloyd,{0,1,2,3,4,5,6,7,8}),convex(additive,{0,1,2,3,4}),additive[3],lloyd[3]);
  const auto variance=spectrum(dir,d);
  const double total=std::accumulate(variance.begin(),variance.end(),0.0,[](double s,double v) { return s+std::max(0.0,v); });
  std::printf("budget\tbits_dim\tdense_greedy\tdense_exact\tfast_mode\tfast_stored\tfast_greedy\texact_equal\texact_equal_counts\texact_nominal\texact_nominal_stored\n");
  for (int a=3;a<argc;++a) {
    const uint32_t budget=std::stoul(argv[a]);
    AllocationModel model{variance,std::vector<uint32_t>(d,max_code_bits),std::vector(d,lloyd)};
    const auto dense=greedy_allocation(model,budget,false);
    std::printf("%u\t%.3f\t%.6f\t%.6f",budget,double(budget)/d,distortion_of(variance,lloyd,dense)/total,dense_optimum(variance,lloyd,budget)/total);
    if (widest_tier(dense)>lookup_code_bits) { std::printf("\tpacked\n"); continue; }
    model.error.assign(d,additive);
    const auto fast=fast_lookup_allocation(model,budget,dense);
    if (widest_tier(fast.bits)>lookup_code_bits) { std::printf("\tpacked-fallback\n"); continue; }
    const auto& table=fast.additive_three ? additive : lloyd;
    const uint32_t stored=lookup_stored_bits(fast.bits,fast.additive_three);
    auto best=[&](uint32_t limit) {
      const auto exact=exact_lookup_allocation(variance,additive,limit);
      return std::pair{distortion_of(variance,additive,exact.bits),exact.bits};
    };
    const auto [equal,equal_bits]=best(stored);
    const auto [nominal,nominal_bits]=best(std::max(budget,stored));
    LookupWidthCounts counts{}; for (unsigned b:equal_bits) ++counts[b];
    std::printf("\t%s\t%u\t%.6f\t%.6f\t%u/%u/%u/%u\t%.6f\t%u\n",fast.additive_three ? "additive" : "2/4",stored,
                distortion_of(variance,table,fast.bits)/total,equal/total,counts[4],counts[3],counts[2],counts[1],
                nominal/total,lookup_stored_bits(nominal_bits,has_three_bit_tier(nominal_bits)));
    std::fflush(stdout);
  }
}
