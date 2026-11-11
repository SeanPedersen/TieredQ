// Issue #9 build and search cost of one TieredQ build: per budget, the median train and add seconds over REPEATS
// (seed-0 sample as in ab_eval, 100k base) and the best 1-thread and 8-thread batch-32 search QPS.
// Allocation switches come from the environment, so one binary times every arm; interleave processes across arms.
// Usage: issue9_build_time LABEL DATA_DIR D REPEATS BUDGET...
#include "tieredq/tieredq.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <omp.h>
#include <optional>
#include <random>
#include <string>
#include <vector>
template<class T> std::vector<T> load(const std::string& path) {
  std::ifstream in(path,std::ios::binary|std::ios::ate);
  std::vector<T> out(size_t(in.tellg())/sizeof(T));
  in.seekg(0); in.read(reinterpret_cast<char*>(out.data()),std::streamsize(out.size()*sizeof(T)));
  return out;
}
constexpr size_t top_k=10,training_limit=32768,batch=32,threads=8,centroids=256;
template<class F> double seconds(F run) {
  const auto start=std::chrono::steady_clock::now(); run();
  return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
double median(std::vector<double> values) {
  std::sort(values.begin(),values.end());
  return values[values.size()/2];
}
int main(int argc,char** argv) {
  if (argc<6) { std::fprintf(stderr,"usage: %s LABEL DATA_DIR D REPEATS BUDGET...\n",argv[0]); return 2; }
  const std::string label=argv[1],dir=argv[2]; const uint32_t d=std::stoul(argv[3]); const size_t repeats=std::stoul(argv[4]);
  const auto base=load<float>(dir+"/base.f32"),queries=load<float>(dir+"/queries.f32");
  const size_t n=base.size()/d,nq=queries.size()/d,train_count=std::min(n,training_limit);
  std::vector<size_t> order(n); std::iota(order.begin(),order.end(),0);
  std::mt19937_64 rng(0); std::shuffle(order.begin(),order.end(),rng);
  std::vector<float> sample(train_count*d);
  for (size_t i=0;i<train_count;++i) std::copy_n(base.data()+order[i]*d,d,sample.data()+i*d);
  std::vector<float> scores(nq*top_k); std::vector<uint32_t> ids(nq*top_k);
  for (int a=5;a<argc;++a) {
    const uint32_t budget=std::stoul(argv[a]);
    auto config=tq::TieredQConfig::target_code_bits(d,budget); config.num_centroids=centroids;
    std::vector<double> train,add; double t1b32=0,t8b32=0; uint32_t model=0,widest=0; double stored=0;
    for (size_t r=0;r<repeats;++r) {
      omp_set_num_threads(threads);
      std::optional<tq::TieredQ> codec;
      train.push_back(seconds([&] { codec.emplace(tq::TieredQ::train(config,sample.data(),train_count,0)); }));
      tq::TieredQIndex index(std::move(*codec));
      add.push_back(seconds([&] { index.add(base.data(),n); }));
      model=index.codec().model_id(); stored=8.0*double(index.storage_bytes())/double(n);
      for (const auto& t:index.codec().tiers()) widest=std::max(widest,t.bits);
      auto batches=[&] { for (size_t q=0;q<nq;q+=batch)
        index.search_batch(queries.data()+q*d,std::min(batch,nq-q),top_k,scores.data(),ids.data()); };
      t8b32=std::max(t8b32,double(nq)/seconds(batches));
      omp_set_num_threads(1);
      t1b32=std::max(t1b32,double(nq)/seconds(batches));
    }
    std::printf("%s\t%u\t%.3f\t%u\t%.1f\t%.3f\t%.3f\t%.0f\t%.0f\t%u\n",label.c_str(),budget,double(budget)/d,widest,stored,
                median(train),median(add),t1b32,t8b32,model);
    std::fflush(stdout);
  }
}
