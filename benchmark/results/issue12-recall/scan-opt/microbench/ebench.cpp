// Index-level fast-search A/B: same synthetic all-8-bit model, link against baseline or new library.
// usage: ebench dims docs [bits_per_dim=8] [max_bits=8]
#include <tieredq/tieredq.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <random>
#include <vector>

using Clock=std::chrono::steady_clock;
static double now_s() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

int main(int argc,char** argv) {
  const uint32_t dims=std::atoi(argv[1]);const size_t docs=std::atoll(argv[2]);
  const uint32_t bits=argc>3?std::atoi(argv[3]):8,max_bits=argc>4?std::atoi(argv[4]):8;
  const size_t train_rows=std::max<size_t>(8192,2*dims),nq=256,k=10;
  std::mt19937 rng(7);std::normal_distribution<float> n;
  std::vector<float> train(train_rows*dims),qs(nq*dims);
  // Anisotropic spectrum so learned tiers differ.
  std::vector<float> scale(dims);for (uint32_t d=0;d<dims;++d) scale[d]=1.0f/std::sqrt(1.0f+d*0.05f);
  for (size_t i=0;i<train.size();++i) train[i]=n(rng)*scale[i%dims];
  for (size_t i=0;i<qs.size();++i) qs[i]=n(rng)*scale[i%dims];
  auto codec=dv::TieredQ::train({dims,dims*bits,256,true,max_bits},train.data(),train_rows);
  std::printf("tiers:");for (auto& t:codec.tiers()) std::printf(" %ux%u",t.dimensions,t.bits);std::printf("\n");
  dv::TieredQIndex index(codec,true);
  std::vector<float> chunk(65536*dims);
  for (size_t done=0;done<docs;done+=65536) {
    const size_t m=std::min<size_t>(65536,docs-done);
    for (size_t i=0;i<m*dims;++i) chunk[i]=n(rng)*scale[i%dims];
    index.add(chunk.data(),m);
  }
  std::vector<float> scores(nq*k);std::vector<uint32_t> ids(nq*k);
  uint64_t hash=1469598103934665603ull;
  auto run=[&](int threads,size_t batch,double min_seconds) {
    omp_set_num_threads(threads);
    index.search_batch(qs.data(),batch,k,scores.data(),ids.data());
    size_t done=0;const double t0=now_s();double t=t0;
    while (t-t0<min_seconds) {
      for (size_t s=0;s+batch<=nq;s+=batch) { index.search_batch(qs.data()+s*dims,batch,k,scores.data()+s*k,ids.data()+s*k); done+=batch; }
      t=now_s();
    }
    return done/(t-t0);
  };
  omp_set_num_threads(8);
  index.search_batch(qs.data(),nq,k,scores.data(),ids.data());
  for (size_t i=0;i<nq*k;++i) { hash=(hash^ids[i])*1099511628211ull; uint32_t b;std::memcpy(&b,&scores[i],4); hash=(hash^b)*1099511628211ull; }
  std::printf("result-hash %016llx\n",(unsigned long long)hash);
  std::printf("qps t8 b32 %.1f\n",run(8,32,3));
  std::printf("qps t8 b1  %.1f\n",run(8,1,3));
  std::printf("qps t1 b1  %.1f\n",run(1,1,3));
  std::printf("qps t1 b32 %.1f\n",run(1,32,3));
}
