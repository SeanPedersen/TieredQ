// Train-only coarse list benchmark (#45): trains lists on a seeded sample of an SQ8 store (memory-mapped, decoded per
// read as TieredQNestedIVFIndex builds do) with Lloyd or e-means, and reports the training time, the peak anonymous
// RSS while training (file-backed store pages excluded), and the held-out objective and list balance of exact
// assignment to the trained (binary16-rounded) centroids. Minutes at 8.84M x 1024 and 64K lists, not a full build.
//
// Usage: tieredq_coarse_training_benchmark BASE.i8 SCALES.f32 OFFSETS.f32 DIMENSIONS LISTS ROWS_PER_LIST
//        lloyd|emeans BUDGET SUPERS PROBES SEED [HELDOUT_ROWS]
// ROWS_PER_LIST 0 trains on every row; BUDGET is Lloyd iterations or e-means epochs; SUPERS 0 assigns exactly.
#include "tieredq/tieredq_nested.h"
#include "tieredq_train_emeans.h"
#include "training_rng.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace {
using namespace tq;
constexpr uint64_t sample_salt=0x51ed27a1b3c9e07dULL,heldout_salt=0x8cb92ba72f3d8dd7ULL;
constexpr size_t default_heldout_rows=131072;
constexpr auto rss_poll=std::chrono::milliseconds(20);

std::vector<float> read_floats(const char* path) {
  std::ifstream in(path,std::ios::binary|std::ios::ate);
  if (!in) throw std::runtime_error(std::string("cannot read ")+path);
  std::vector<float> values(size_t(in.tellg())/sizeof(float));
  in.seekg(0); in.read(reinterpret_cast<char*>(values.data()),std::streamsize(values.size()*sizeof(float)));
  return values;
}

size_t anonymous_rss_bytes() {
  std::ifstream status("/proc/self/status");
  for (std::string line;std::getline(status,line);)
    if (line.rfind("RssAnon:",0)==0) return std::stoull(line.substr(8))*1024;
  return 0;
}

// Polls RssAnon in the background and keeps the maximum.
class AnonymousPeak {
 public:
  AnonymousPeak() : poller_([this] { while (!done_) { sample(); std::this_thread::sleep_for(rss_poll); } }) {}
  ~AnonymousPeak() { done_=true; poller_.join(); }
  size_t bytes() { sample(); return peak_; }
 private:
  void sample() { const size_t now=anonymous_rss_bytes(); size_t seen=peak_; while (now>seen && !peak_.compare_exchange_weak(seen,now)) {} }
  std::atomic<bool> done_{false};
  std::atomic<size_t> peak_{0};
  std::thread poller_;
};

std::vector<uint32_t> sorted_sample(size_t population,size_t count,uint64_t seed) {
  const auto picked=tieredq_detail::TrainingRng(seed).choice(population,count);
  std::vector<uint32_t> ids(picked.begin(),picked.end());
  std::sort(ids.begin(),ids.end());
  return ids;
}
} // namespace

int main(int argc,char** argv) {
  if (argc<12) { std::cerr<<"usage: see the file header\n"; return 2; }
  const size_t d=std::stoul(argv[4]),lists=std::stoul(argv[5]),per_list=std::stoul(argv[6]);
  const std::string trainer=argv[7];
  const double budget=std::stod(argv[8]);
  const size_t supers=std::stoul(argv[9]),probes=std::stoul(argv[10]);
  const uint64_t seed=std::stoull(argv[11]);
  const size_t heldout=argc>12 ? std::stoul(argv[12]) : default_heldout_rows;

  const int fd=open(argv[1],O_RDONLY);
  struct stat info{};
  if (fd<0 || fstat(fd,&info)) { std::perror(argv[1]); return 1; }
  const size_t count=size_t(info.st_size)/d;
  const auto* codes=static_cast<const int8_t*>(mmap(nullptr,size_t(info.st_size),PROT_READ,MAP_SHARED,fd,0));
  const auto loader=sq8_row_loader(codes,count,read_floats(argv[2]),read_floats(argv[3]));

  TieredQCoarseTraining training{};
  if (trainer=="emeans") training.emeans_epochs=budget; else training.iterations=uint32_t(budget);
  training.assignment_supers=uint32_t(supers); training.assignment_probes=uint32_t(probes);
  const size_t rows=per_list ? std::min(count,per_list*lists) : count;
  const auto ids=rows==count ? [&] { std::vector<uint32_t> all(count); std::iota(all.begin(),all.end(),0u); return all; }()
                             : sorted_sample(count,rows,seed^sample_salt);
  std::vector<uint32_t> chunk_ids;
  const tieredq_detail::TrainingRowSource source=[&](const uint32_t* positions,size_t n,float* out) {
    chunk_ids.resize(n);
    for (size_t i=0;i<n;++i) chunk_ids[i]=ids[positions[i]];
    loader(chunk_ids.data(),n,out);
  };

  const size_t baseline_anonymous=anonymous_rss_bytes();
  double train_s=0;
  size_t peak_anonymous=0;
  tieredq_detail::CoarseQuantizer coarse;
  {
    AnonymousPeak peak;
    const auto started=std::chrono::steady_clock::now();
    coarse=tieredq_detail::train_coarse_rows(source,rows,d,lists,seed,training);
    train_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    peak_anonymous=peak.bytes();
  }

  const auto heldout_ids=sorted_sample(count,std::min(heldout,count),seed^heldout_salt);
  std::vector<float> x(tieredq_detail::product(heldout_ids.size(),d)),half(lists);
  loader(heldout_ids.data(),heldout_ids.size(),x.data());
  const auto centroids=coarse.centroids();
  tieredq_detail::SuperRestriction::half_norms(centroids.data(),lists,d,half.data());
  std::vector<uint32_t> labels(heldout_ids.size());
  tieredq_detail::nearest_centroids(centroids.data(),half.data(),lists,d,x.data(),labels.size(),labels.data());
  double objective=0,squares=0;
  std::vector<size_t> sizes(lists);
  for (size_t r=0;r<labels.size();++r) {
    const float* row=x.data()+r*d;
    const float* centroid=centroids.data()+size_t(labels[r])*d;
    double distance=0;
    for (size_t j=0;j<d;++j) distance+=double(row[j]-centroid[j])*(row[j]-centroid[j]);
    objective+=distance; ++sizes[labels[r]];
  }
  for (const size_t s:sizes) squares+=double(s)*double(s);
  // Imbalance k * sum n^2 / (sum n)^2: the expected rows scanned per probe over the balanced partition's.
  const double imbalance=double(lists)*squares/(double(labels.size())*double(labels.size()));
  std::printf("{\"trainer\":\"%s\",\"budget\":%g,\"rows\":%zu,\"lists\":%zu,\"supers\":%zu,\"probes\":%zu,\"seed\":%llu,"
              "\"train_s\":%.2f,\"anon_peak_gb\":%.3f,\"anon_train_gb\":%.3f,\"heldout\":%zu,\"objective\":%.6f,"
              "\"imbalance\":%.4f}\n",
              trainer.c_str(),budget,rows,lists,supers,probes,static_cast<unsigned long long>(seed),train_s,
              double(peak_anonymous)/1e9,double(peak_anonymous-std::min(peak_anonymous,baseline_anonymous))/1e9,
              labels.size(),objective/double(labels.size()),imbalance);
  return 0;
}
