// Issue #6: TieredQ fast-mode IVF (TieredQIVFIndex) vs RaBitQ's IVF on the same partition, recall/storage/QPS per nprobe.
// The partition is TieredQ's k-means at NLIST centroids (training sample of up to 32,768 rows, seed 0): RaBitQ gets
// every row's TieredQ list and each list's row mean as its centroid, so both scan identical lists; only the codec
// and the list ranking (TieredQ: query-centroid score, RaBitQ: its own centroid distance) differ. RaBitQ encodes
// with its accurate configuration (faster_config off) and high-accuracy scoring. Disk is the serialized index.
// Modes: tieredq CODE_BITS | rabitq BITS_PER_DIM | tieredq-match TARGET_DISK_BITS (largest code budget whose disk
// bits/vector fit, storage only) | tieredq-profile CODE_BITS (single-query loops only, for perf; prints qps_1t_b1). Rows: codec budget nlist nprobe disk_bits code_bits recall qps_1t_b1 qps_8t_b32
// build_s list_hash. LISTS gives TieredQ a coarse quantizer with that many lists (the codec keeps NLIST centroids) and
// RaBitQ the same partition. PRUNE_Z and PRUNE_PREFIX set TieredQPruning. MAX_BITS caps TieredQ bits per coordinate (TieredQConfig::max_bits). Data as ab_eval: DATA_DIR/{base.f32,queries.f32,truth.u32}, inner-product truth.
// Usage: issue6_ivf_eval MODE BUDGET DATA_DIR D NLIST TIMED(0|1) NPROBE...   (RaBitQ needs BENCH_SEED set)
#include "tieredq/tieredq.h"
#include <rabitqlib/index/ivf/ivf.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <omp.h>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>
namespace {
template<class T> std::vector<T> load(const std::string& path) {
  std::ifstream in(path,std::ios::binary|std::ios::ate);
  std::vector<T> out(size_t(in.tellg())/sizeof(T));
  in.seekg(0); in.read(reinterpret_cast<char*>(out.data()),std::streamsize(out.size()*sizeof(T)));
  return out;
}
constexpr size_t top_k=10,training_limit=32768,batch=32,threads=8,repeats=3,match_resolution=8;
constexpr uint64_t seed=0;
struct Data { std::vector<float> base,queries; std::vector<uint32_t> truth; size_t d,n,nq; };
double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
template<class F> double best_seconds(F run) {
  double best=1e30;
  for (size_t r=0;r<repeats;++r) { const auto start=std::chrono::steady_clock::now(); run(); best=std::min(best,seconds_since(start)); }
  return best;
}
double recall(const Data& data,const std::vector<uint32_t>& ids) {
  size_t hits=0;
  for (size_t q=0;q<data.nq;++q) for (size_t j=0;j<top_k;++j)
    hits+=std::count(data.truth.begin()+q*top_k,data.truth.begin()+(q+1)*top_k,ids[q*top_k+j]);
  return double(hits)/double(data.nq*top_k);
}
dv::TieredQ train(const Data& data,uint32_t code_bits,size_t nlist) {
  const size_t count=std::min(data.n,training_limit);
  std::vector<size_t> order(data.n); std::iota(order.begin(),order.end(),0);
  std::mt19937_64 rng(seed); std::shuffle(order.begin(),order.end(),rng);
  std::vector<float> sample(count*data.d);
  for (size_t i=0;i<count;++i) std::copy_n(data.base.data()+order[i]*data.d,data.d,sample.data()+i*data.d);
  auto config=dv::TieredQConfig::target_code_bits(uint32_t(data.d),code_bits);
  config.num_centroids=uint32_t(nlist);
  if (const char* max_bits=std::getenv("MAX_BITS")) config.max_bits=uint32_t(std::stoul(max_bits));
  return dv::TieredQ::train(config,sample.data(),count,seed);
}
// FNV-1a over list sizes: equal hashes mean both codecs scan the same partition.
uint64_t list_hash(const std::vector<size_t>& sizes) {
  uint64_t hash=14695981039346656037ULL;
  for (size_t size:sizes) { hash^=size; hash*=1099511628211ULL; }
  return hash;
}
struct Searcher {
  std::function<void(const float*,size_t,size_t,uint32_t*)> batch;  // queries, rows, nprobe, ids (8 threads)
  std::function<void(const float*,size_t,uint32_t*)> single;        // one query on the calling thread
};
void sweep(const char* codec,double budget,size_t nlist,const Data& data,const Searcher& search,double disk_bits,
           double code_bits,double build_s,uint64_t lists,bool timed,const std::vector<size_t>& nprobes) {
  std::vector<uint32_t> ids(data.nq*top_k);
  for (size_t nprobe:nprobes) {
    omp_set_num_threads(threads);
    search.batch(data.queries.data(),data.nq,nprobe,ids.data());
    const double r=recall(data,ids);
    double qps1=0,qps8=0;
    if (timed) {
      qps8=double(data.nq)/best_seconds([&] {
        for (size_t q=0;q<data.nq;q+=batch)
          search.batch(data.queries.data()+q*data.d,std::min(batch,data.nq-q),nprobe,ids.data()+q*top_k);
      });
      omp_set_num_threads(1);
      qps1=double(data.nq)/best_seconds([&] {
        for (size_t q=0;q<data.nq;++q) search.single(data.queries.data()+q*data.d,nprobe,ids.data()+q*top_k);
      });
    }
    std::printf("%s\t%g\t%zu\t%zu\t%.1f\t%.0f\t%.4f\t%.0f\t%.0f\t%.2f\t%016llx\n",codec,budget,nlist,nprobe,disk_bits,
                code_bits,r,qps1,qps8,build_s,static_cast<unsigned long long>(lists));
    std::fflush(stdout);
  }
}
double tieredq_disk_bits(const dv::TieredQIVFIndex& index) { return 8.0*double(index.serialize().size())/double(index.size()); }

size_t coarse_lists() { return std::getenv("LISTS") ? std::stoul(std::getenv("LISTS")) : 0; }
// PRUNE_Z / PRUNE_PREFIX: TieredQPruning confidence and prefix tiers (defaults: exact bound, 4 tiers).
dv::TieredQPruning pruning() {
  dv::TieredQPruning options;
  if (const char* z=std::getenv("PRUNE_Z")) options.confidence=std::stod(z);
  if (const char* prefix=std::getenv("PRUNE_PREFIX")) options.prefix_tiers=uint32_t(std::stoul(prefix));
  return options;
}

void run_tieredq(const Data& data,uint32_t code_bits,size_t nlist,bool timed,const std::vector<size_t>& nprobes,double label) {
  const auto start=std::chrono::steady_clock::now();
  omp_set_num_threads(threads);
  dv::TieredQIVFIndex index(train(data,code_bits,nlist),uint32_t(coarse_lists()));
  index.build(data.base.data(),data.n);
  const double build_s=seconds_since(start);
  std::vector<size_t> sizes(index.lists());
  for (size_t list=0;list<sizes.size();++list) sizes[list]=index.list_size(list);
  Searcher search{
    [&](const float* q,size_t rows,size_t nprobe,uint32_t* ids) {
      std::vector<float> scores(rows*top_k);
      index.search_batch(q,rows,top_k,nprobe,scores.data(),ids,pruning());
    },
    [&](const float* q,size_t nprobe,uint32_t* ids) {
      const auto hits=index.search(q,top_k,nprobe,pruning());
      for (size_t i=0;i<top_k;++i) ids[i]=i<hits.size() ? hits[i].id : UINT32_MAX;
    }};
  sweep("tieredq",label,index.lists(),data,search,tieredq_disk_bits(index),index.codec().config().code_bits,build_s,
        list_hash(sizes),timed,nprobes);
}

// Repeated single-query loops on the calling thread, nothing else timed: a clean perf target.
void profile_tieredq(const Data& data,uint32_t code_bits,size_t nlist,const std::vector<size_t>& nprobes) {
  const size_t profile_rounds=std::getenv("PROFILE_ROUNDS") ? std::stoul(std::getenv("PROFILE_ROUNDS")) : 10;
  omp_set_num_threads(threads);
  dv::TieredQIVFIndex index(train(data,code_bits,nlist),uint32_t(coarse_lists()));
  index.build(data.base.data(),data.n);
  omp_set_num_threads(1);
  std::fprintf(stderr,"coded ranks %zu of %zu, tiers (bits x dims):",index.codec().coded_dimensions(),index.codec().model_dimensions());
  for (const auto& tier:index.codec().tiers()) std::fprintf(stderr," %ux%u",tier.bits,tier.dimensions);
  std::fprintf(stderr,"\nprofile ready\n");
  for (size_t nprobe:nprobes) {
    const auto start=std::chrono::steady_clock::now();
    size_t found=0;
    for (size_t round=0;round<profile_rounds;++round)
      for (size_t q=0;q<data.nq;++q) found+=index.search(data.queries.data()+q*data.d,top_k,nprobe,pruning()).size();
    std::printf("profile\t%u\t%zu\t%zu\t%.0f\t%zu\n",code_bits,nlist,nprobe,
                double(profile_rounds*data.nq)/seconds_since(start),found);
    std::fflush(stdout);
  }
}

// Largest code budget (to match_resolution) whose serialized IVF fits target bits/vector; storage only.
uint32_t match_code_bits(const Data& data,double target,size_t nlist) {
  auto fits=[&](uint32_t code_bits) {
    dv::TieredQIVFIndex index(train(data,code_bits,nlist),uint32_t(coarse_lists()));
    index.build(data.base.data(),data.n);
    const double bits=tieredq_disk_bits(index);
    std::fprintf(stderr,"match target %.1f: code_bits %u -> %.1f bits/vector\n",target,code_bits,bits);
    return bits<=target;
  };
  const double max_bits=std::getenv("MAX_BITS") ? std::stod(std::getenv("MAX_BITS")) : 8.0;
  // MATCH_SPAN narrows the search to [target - span, target] when the per-vector overhead is known roughly.
  const double span=std::getenv("MATCH_SPAN") ? std::stod(std::getenv("MATCH_SPAN")) : target;
  uint32_t lo=uint32_t(std::max(1.0,target-span)),hi=uint32_t(std::min<double>(target,max_bits*double(data.d)));
  if (fits(hi)) lo=hi;
  while (hi-lo>match_resolution) { const uint32_t mid=lo+(hi-lo)/2; (fits(mid) ? lo : hi)=mid; }
  std::fprintf(stderr,"matched code budget %u\n",lo);
  return lo;
}

// TieredQ's partition: the codec's k-means lists, or with LISTS its coarse quantizer's; both are independent of the
// code budget at a fixed seed (the coarse quantizer trains in model space, which for inner product is the input).
std::vector<uint32_t> tieredq_partition(const Data& data,size_t nlist) {
  if (!coarse_lists()) {
    const auto codec=train(data,uint32_t(data.d),nlist);
    const auto encoded=codec.encode(data.base.data(),data.n);
    std::vector<uint32_t> lists(data.n);
    for (size_t i=0;i<data.n;++i) lists[i]=encoded[i*codec.bytes_per_vector()+codec.code_bytes()];
    return lists;
  }
  dv::TieredQIVFIndex index(train(data,uint32_t(data.d),nlist),uint32_t(coarse_lists()));
  index.build(data.base.data(),data.n);
  return index.list_assignment();
}

void run_rabitq(const Data& data,size_t bits,size_t codec_lists,bool timed,const std::vector<size_t>& nprobes) {
  omp_set_num_threads(threads);
  const size_t nlist=coarse_lists() ? coarse_lists() : codec_lists;
  const auto partition=tieredq_partition(data,codec_lists);
  std::vector<rabitqlib::PID> assignment(data.n);
  std::vector<float> centroids(nlist*data.d,0.0f);
  std::vector<size_t> sizes(nlist,0);
  for (size_t i=0;i<data.n;++i) {
    const size_t list=partition[i];
    assignment[i]=rabitqlib::PID(list); ++sizes[list];
    for (size_t j=0;j<data.d;++j) centroids[list*data.d+j]+=data.base[i*data.d+j];
  }
  for (size_t list=0;list<nlist;++list) if (sizes[list])
    for (size_t j=0;j<data.d;++j) centroids[list*data.d+j]/=float(sizes[list]);
  const auto start=std::chrono::steady_clock::now();
  rabitqlib::ivf::IVF index(data.n,data.d,nlist,bits,rabitqlib::METRIC_IP);
  index.construct(data.base.data(),centroids.data(),assignment.data(),false);
  const double build_s=seconds_since(start);
  const auto path=std::filesystem::temp_directory_path()/("issue6-rabitq-"+std::to_string(::getpid())+".bin");
  index.save(path.c_str());
  const double disk_bits=8.0*double(std::filesystem::file_size(path))/double(data.n);
  std::filesystem::remove(path);
  Searcher search{
    [&](const float* q,size_t rows,size_t nprobe,uint32_t* ids) {
#pragma omp parallel for schedule(dynamic)
      for (size_t row=0;row<rows;++row) index.search(q+row*data.d,top_k,nprobe,ids+row*top_k,true);
    },
    [&](const float* q,size_t nprobe,uint32_t* ids) { index.search(q,top_k,nprobe,ids,true); }};
  sweep("rabitq",double(bits),nlist,data,search,disk_bits,double(bits*data.d),build_s,list_hash(sizes),timed,nprobes);
}
} // namespace

int main(int argc,char** argv) {
  if (argc<8) { std::fprintf(stderr,"usage: %s MODE BUDGET DATA_DIR D NLIST TIMED NPROBE...\n",argv[0]); return 2; }
  const std::string mode=argv[1],dir=argv[3];
  const double budget=std::stod(argv[2]);
  Data data{load<float>(dir+"/base.f32"),load<float>(dir+"/queries.f32"),load<uint32_t>(dir+"/truth.u32"),std::stoul(argv[4]),0,0};
  data.n=data.base.size()/data.d; data.nq=data.queries.size()/data.d;
  const size_t nlist=std::stoul(argv[5]);
  const bool timed=std::stoi(argv[6]);
  std::vector<size_t> nprobes;
  for (int a=7;a<argc;++a) nprobes.push_back(std::stoul(argv[a]));
  if (mode=="tieredq") run_tieredq(data,uint32_t(budget),nlist,timed,nprobes,budget);
  else if (mode=="tieredq-match") run_tieredq(data,match_code_bits(data,budget,nlist),nlist,timed,nprobes,budget);
  else if (mode=="tieredq-profile") profile_tieredq(data,uint32_t(budget),nlist,nprobes);
  else if (mode=="rabitq") run_rabitq(data,size_t(budget),nlist,timed,nprobes);
  else { std::fprintf(stderr,"unknown mode %s\n",mode.c_str()); return 2; }
}
