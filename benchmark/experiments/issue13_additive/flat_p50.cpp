// Issue #13: flat single-query latency and recall of one budget, exact code lookup with range pruning (full-target
// training, Lloyd-Max books) against additive nibble tables (fast-target training). Queries run in order on one thread.
// Usage: flat_p50 DATA_DIR D CODE_BITS ROUNDS; prints arm round p50_ms mean_ms recall@10 bytes_per_vector.
#include "tieredq/tieredq.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

constexpr size_t top_k=10,training_rows=32768;
constexpr uint32_t centroids=16;
constexpr uint64_t seed=0;

template<class T> std::vector<T> load(const std::string& path) {
  std::ifstream file(path,std::ios::binary|std::ios::ate);
  std::vector<T> values(size_t(file.tellg())/sizeof(T));
  file.seekg(0); file.read(reinterpret_cast<char*>(values.data()),std::streamsize(values.size()*sizeof(T)));
  return values;
}
int main(int,char** argv) {
  const std::string dir=argv[1]; const size_t d=std::stoul(argv[2]); const auto code_bits=uint32_t(std::stoul(argv[3])); const int rounds=std::stoi(argv[4]);
  const auto base=load<float>(dir+"/base.f32"),queries=load<float>(dir+"/queries.f32"); const auto truth=load<uint32_t>(dir+"/truth.u32");
  const size_t n=base.size()/d,nq=queries.size()/d;
  std::vector<size_t> order(n); std::iota(order.begin(),order.end(),size_t(0));
  std::mt19937_64 rng(seed); std::shuffle(order.begin(),order.end(),rng); order.resize(std::min(n,training_rows));
  std::vector<float> sample(order.size()*d);
  for (size_t i=0;i<order.size();++i) std::copy_n(base.data()+order[i]*d,d,sample.data()+i*d);
  struct Arm { const char* name; dv::TieredQIndexMode target; };
  std::vector<dv::TieredQIndex> indexes; const std::vector<Arm> arms{{"exact",dv::TieredQIndexMode::full},{"nibble",dv::TieredQIndexMode::fast}};
  for (const auto& arm:arms) {
    dv::TieredQConfig config{uint32_t(d),code_bits,centroids,true}; config.target_mode=arm.target;
    indexes.emplace_back(dv::TieredQ::train(config,sample.data(),order.size(),seed),dv::TieredQIndexMode::fast);
    indexes.back().add(base.data(),n);
  }
  for (int round=0;round<rounds;++round) for (size_t a=0;a<arms.size();++a) {
    std::vector<double> latency(nq); size_t hits=0;
    for (size_t q=0;q<nq;++q) {
      const auto start=std::chrono::steady_clock::now();
      const auto found=indexes[a].search(queries.data()+q*d,top_k);
      latency[q]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      for (const auto& hit:found) hits+=std::count(truth.begin()+q*top_k,truth.begin()+(q+1)*top_k,hit.id);
    }
    const double mean=std::accumulate(latency.begin(),latency.end(),0.0)/nq;
    std::nth_element(latency.begin(),latency.begin()+nq/2,latency.end());
    std::printf("%s\t%d\t%.3f\t%.3f\t%.4f\t%zu\n",arms[a].name,round,latency[nq/2],mean,double(hits)/double(nq*top_k),indexes[a].bytes_per_vector());
    std::fflush(stdout);
  }
}
