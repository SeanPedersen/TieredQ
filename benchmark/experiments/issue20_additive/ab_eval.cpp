// Issue #20 A/B: fast-mode recall, storage and QPS of one TieredQ build at requested code budgets.
// Link once against each library build and interleave processes; rows are tagged with LABEL.
// Recall is the mean over seeds 0-2 (inner-product truth on the raw base); QPS is seed 0, best of REPEATS,
// at 1 thread batch 1, 1 thread batch 32 and THREADS threads batch 32. model_id identifies identical models.
// The last two columns split recall into even and odd query ids, for choosing a setting on one half (#9).
// Usage: issue20_ab_eval LABEL DATA_DIR D TIMED(0|1) BUDGET...
#include "tieredq/tieredq.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <omp.h>
#include <random>
#include <string>
#include <vector>
template<class T> std::vector<T> load(const std::string& path) {
  std::ifstream in(path,std::ios::binary|std::ios::ate);
  std::vector<T> out(size_t(in.tellg())/sizeof(T));
  in.seekg(0); in.read(reinterpret_cast<char*>(out.data()),std::streamsize(out.size()*sizeof(T)));
  return out;
}
constexpr size_t top_k=10,training_limit=32768,batch=32,threads=8,repeats=5,centroids=256;
constexpr uint64_t seeds=3;
template<class F> double best_seconds(F run) {
  double best=1e30;
  for (size_t r=0;r<repeats;++r) {
    const auto start=std::chrono::steady_clock::now(); run();
    best=std::min(best,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
  }
  return best;
}
struct Result { double recall=0,recall_even=0,recall_odd=0,stored_bits=0,vector_bytes=0,three=0,realised=0,widest=0,t1b1=0,t1b32=0,t8b32=0; uint32_t model=0; };
Result evaluate(const std::vector<float>& base,const std::vector<float>& queries,const std::vector<uint32_t>& truth,
                uint32_t d,uint32_t code_bits,uint64_t seed,bool timed) {
  const size_t n=base.size()/d,nq=queries.size()/d,train_count=std::min(n,training_limit);
  std::vector<size_t> order(n); std::iota(order.begin(),order.end(),0);
  std::mt19937_64 rng(seed); std::shuffle(order.begin(),order.end(),rng);
  std::vector<float> sample(train_count*d);
  for (size_t i=0;i<train_count;++i) std::copy_n(base.data()+order[i]*d,d,sample.data()+i*d);
  auto config=dv::TieredQConfig::target_code_bits(d,code_bits);
  config.num_centroids=centroids;
  dv::TieredQIndex index(dv::TieredQ::train(config,sample.data(),train_count,seed));
  index.add(base.data(),n);
  std::vector<float> scores(nq*top_k); std::vector<uint32_t> ids(nq*top_k);
  omp_set_num_threads(threads);
  index.search_batch(queries.data(),nq,top_k,scores.data(),ids.data());
  std::array<size_t,2> hits{},half_queries{};
  for (size_t q=0;q<nq;++q) {
    ++half_queries[q%2];
    for (size_t j=0;j<top_k;++j) hits[q%2]+=std::count(truth.begin()+q*top_k,truth.begin()+(q+1)*top_k,ids[q*top_k+j]);
  }
  Result r; r.recall=double(hits[0]+hits[1])/double(nq*top_k);
  r.recall_even=double(hits[0])/double(half_queries[0]*top_k); r.recall_odd=double(hits[1])/double(half_queries[1]*top_k); r.stored_bits=8.0*double(index.storage_bytes())/double(n);
  r.vector_bytes=double(index.bytes_per_vector()); r.realised=index.codec().config().code_bits; r.model=index.codec().model_id();
  for (const auto& t:index.codec().tiers()) { if (t.bits==3) r.three+=t.dimensions; r.widest=std::max<double>(r.widest,t.bits); }
  if (!timed) return r;
  auto batches=[&] { for (size_t q=0;q<nq;q+=batch)
    index.search_batch(queries.data()+q*d,std::min(batch,nq-q),top_k,scores.data(),ids.data()); };
  r.t8b32=double(nq)/best_seconds(batches);
  omp_set_num_threads(1);
  r.t1b32=double(nq)/best_seconds(batches);
  r.t1b1=double(nq)/best_seconds([&] { for (size_t q=0;q<nq;++q) index.search(queries.data()+q*d,top_k); });
  return r;
}
int main(int argc,char** argv) {
  if (argc<6) { std::fprintf(stderr,"usage: %s LABEL DATA_DIR D TIMED BUDGET...\n",argv[0]); return 2; }
  const std::string label=argv[1],dir=argv[2]; const uint32_t d=std::stoul(argv[3]); const bool timed=std::stoi(argv[4]);
  const auto base=load<float>(dir+"/base.f32"),queries=load<float>(dir+"/queries.f32");
  const auto truth=load<uint32_t>(dir+"/truth.u32");
  for (int a=5;a<argc;++a) {
    const uint32_t budget=std::stoul(argv[a]);
    Result mean; std::string models;
    for (uint64_t seed=0;seed<seeds;++seed) {
      const auto r=evaluate(base,queries,truth,d,budget,seed,timed && seed==0);
      mean.recall+=r.recall/seeds; mean.recall_even+=r.recall_even/seeds; mean.recall_odd+=r.recall_odd/seeds; mean.stored_bits+=r.stored_bits/seeds; mean.three+=r.three/seeds;
      mean.realised+=r.realised/seeds; mean.vector_bytes+=r.vector_bytes/seeds; mean.widest=std::max(mean.widest,r.widest);
      models+=(seed ? "," : "")+std::to_string(r.model);
      if (seed==0) { mean.t1b1=r.t1b1; mean.t1b32=r.t1b32; mean.t8b32=r.t8b32; }
    }
    std::printf("%s\t%u\t%.3f\t%.1f\t%.0f\t%.1f\t%.2f\t%.1f\t%.4f\t%.0f\t%.0f\t%.0f\t%s\t%.4f\t%.4f\n",label.c_str(),budget,double(budget)/d,
                mean.realised,mean.widest,mean.three,mean.vector_bytes,mean.stored_bits,mean.recall,mean.t1b1,mean.t1b32,mean.t8b32,models.c_str(),mean.recall_even,mean.recall_odd);
    std::fflush(stdout);
  }
}
