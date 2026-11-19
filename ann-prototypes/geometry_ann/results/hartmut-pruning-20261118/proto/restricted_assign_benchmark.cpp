// #45 experiment: final restricted assignment of every row of an SQ8 store to saved coarse centroids, as
// TieredQIVFIndex::encode_lists assigns (65,536-row chunks through the SQ8 loader, one RestrictedAssigner). Reports the
// setup and assignment times and writes the labels (uint32) for agreement checks. TQ_PRUNE_DIMS selects the pruning
// prototype.
// Usage: restricted_assign_benchmark BASE.i8 SCALES.f32 OFFSETS.f32 DIMENSIONS CENTROIDS.f32 LISTS SUPERS PROBES SEED LABELS.u32
#include "tieredq/tieredq_nested.h"
#include "tieredq_ivf_assign.h"
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>

namespace {
constexpr size_t assign_chunk=65536;

std::vector<float> read_floats(const char* path) {
  std::ifstream in(path,std::ios::binary|std::ios::ate);
  if (!in) throw std::runtime_error(std::string("cannot read ")+path);
  std::vector<float> values(size_t(in.tellg())/sizeof(float));
  in.seekg(0); in.read(reinterpret_cast<char*>(values.data()),std::streamsize(values.size()*sizeof(float)));
  return values;
}
double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
} // namespace

int main(int argc,char** argv) {
  if (argc<11) { std::cerr<<"usage: see the file header\n"; return 2; }
  using namespace tq;
  const size_t d=std::stoul(argv[4]),lists=std::stoul(argv[6]),supers=std::stoul(argv[7]),probes=std::stoul(argv[8]);
  const uint64_t seed=std::stoull(argv[9]);
  const int fd=open(argv[1],O_RDONLY);
  struct stat info{};
  if (fd<0 || fstat(fd,&info)) { std::perror(argv[1]); return 1; }
  const size_t count=size_t(info.st_size)/d;
  const auto* codes=static_cast<const int8_t*>(mmap(nullptr,size_t(info.st_size),PROT_READ,MAP_SHARED,fd,0));
  const auto loader=sq8_row_loader(codes,count,read_floats(argv[2]),read_floats(argv[3]));
  const auto coarse=tieredq_detail::CoarseQuantizer::from(read_floats(argv[5]),lists,d);

  auto started=std::chrono::steady_clock::now();
  const tieredq_detail::RestrictedAssigner assigner(coarse,supers,probes,seed);
  const double setup_s=seconds_since(started);
  std::vector<uint32_t> labels(count),ids(assign_chunk);
  std::vector<float> chunk(assign_chunk*d);
  double load_s=0;
  started=std::chrono::steady_clock::now();
  for (size_t first=0;first<count;first+=assign_chunk) {
    const size_t n=std::min(assign_chunk,count-first);
    std::iota(ids.begin(),ids.begin()+ptrdiff_t(n),uint32_t(first));
    const auto loading=std::chrono::steady_clock::now();
    loader(ids.data(),n,chunk.data());
    load_s+=seconds_since(loading);
    assigner.assign(chunk.data(),n,labels.data()+first);
  }
  const double assign_s=seconds_since(started);
  std::ofstream(argv[10],std::ios::binary).write(reinterpret_cast<const char*>(labels.data()),
                                                 std::streamsize(labels.size()*sizeof(uint32_t)));
  const char* dims=std::getenv("TQ_PRUNE_DIMS");
  std::printf("{\"prune_dims\":%s,\"rows\":%zu,\"lists\":%zu,\"supers\":%zu,\"probes\":%zu,\"setup_s\":%.2f,"
              "\"assign_s\":%.2f,\"load_s\":%.2f}\n",dims ? dims : "0",count,lists,supers,probes,setup_s,assign_s,load_s);
  return 0;
}
