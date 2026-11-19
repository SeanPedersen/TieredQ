// #45 experiment: achieved FLOP rates of the final restricted assignment's stages. Assigns every row of an SQ8 store
// to saved coarse centroids (65,536-row chunks as encode_lists), 256 supers x 16 probes, with bound BOUND (0: off), and
// prints wall seconds, executed GFLOP (padded panel columns included) and GFLOP/s per stage; then reference peaks on
// the same threads: OpenBLAS sgemm (4096 x 4096 x 1024) and both panel kernels on in-cache 64-row blocks per thread.
// Usage: assign_flops_benchmark BASE.i8 SCALES.f32 OFFSETS.f32 DIMENSIONS CENTROIDS.f32 LISTS BOUND
#include "tieredq/tieredq_nested.h"
#include "tieredq_ivf_assign.h"
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>

namespace {
using namespace tq;
using namespace tq::tieredq_detail;
constexpr size_t assign_chunk=65536,supers=256,probes=16,peak_repeats=200;
constexpr uint64_t seed=0;

std::vector<float> read_floats(const char* path) {
  std::ifstream in(path,std::ios::binary|std::ios::ate);
  if (!in) throw std::runtime_error(std::string("cannot read ")+path);
  std::vector<float> values(size_t(in.tellg())/sizeof(float));
  in.seekg(0); in.read(reinterpret_cast<char*>(values.data()),std::streamsize(values.size()*sizeof(float)));
  return values;
}
void stage(const char* name,double seconds,unsigned long long flop) {
  std::printf("  %-22s %8.2f s %10.1f GFLOP %9.1f GFLOP/s\n",name,seconds,flop/1e9,seconds>0 ? flop/1e9/seconds : 0.0);
}
// Panel kernel peak: every thread multiplies its own in-cache 64-row block by a 288-column panel set, repeatedly.
template<class Panels,class Multiply> double panel_peak(size_t depth,Multiply multiply,Panels make) {
  const auto panels=make(depth);
  const size_t width=panels.padded_columns();
  double seconds=0;
#pragma omp parallel
  {
    std::mt19937 random(omp_get_thread_num()); std::normal_distribution<float> normal;
    std::vector<float> rows(CoarseQuantizer::block_rows*depth),out(CoarseQuantizer::block_rows*width);
    for (float& v:rows) v=normal(random);
#pragma omp barrier
#pragma omp master
    seconds=profile_now();
#pragma omp barrier
    for (size_t r=0;r<peak_repeats;++r) multiply(rows.data(),panels,out.data(),depth,width);
#pragma omp barrier
#pragma omp master
    seconds=profile_now()-seconds;
  }
  return 2.0*omp_get_max_threads()*peak_repeats*CoarseQuantizer::block_rows*width*depth/1e9/seconds;
}
} // namespace

int main(int argc,char** argv) {
  if (argc<8) { std::cerr<<"usage: see the file header\n"; return 2; }
  const size_t d=std::stoul(argv[4]),lists=std::stoul(argv[6]),bound=std::stoul(argv[7]);
  const int fd=open(argv[1],O_RDONLY);
  struct stat info{};
  if (fd<0 || fstat(fd,&info)) { std::perror(argv[1]); return 1; }
  const size_t count=size_t(info.st_size)/d;
  const auto* codes=static_cast<const int8_t*>(mmap(nullptr,size_t(info.st_size),PROT_READ,MAP_SHARED,fd,0));
  const auto loader=sq8_row_loader(codes,count,read_floats(argv[2]),read_floats(argv[3]));
  const auto coarse=CoarseQuantizer::from(read_floats(argv[5]),lists,d);

  double started=profile_now();
  const RestrictedAssigner assigner(coarse,supers,probes,seed,bound);
  const double setup_s=profile_now()-started;
  std::vector<uint32_t> labels(count),ids(assign_chunk);
  std::vector<float> chunk(assign_chunk*d);
  double load_s=0;
  started=profile_now();
  for (size_t first=0;first<count;first+=assign_chunk) {
    const size_t n=std::min(assign_chunk,count-first);
    std::iota(ids.begin(),ids.begin()+ptrdiff_t(n),uint32_t(first));
    const double loading=profile_now();
    loader(ids.data(),n,chunk.data());
    load_s+=profile_now()-loading;
    assigner.assign(chunk.data(),n,labels.data()+first);
  }
  const double total_s=profile_now()-started;
  auto& p=AssignProfile::get();
  const double scan_cpu=p.bound_gemm_cpu_s+p.bound_scan_cpu_s;
  std::printf("bound %zu: %zu rows, %d threads, setup %.2f s, assign loop %.2f s (row loading %.2f s)\n",bound,count,
              omp_get_max_threads(),setup_s,total_s,load_s);
  stage("probe supers",p.probe_s,p.probe_flop);
  if (bound) {
    stage("project rows",p.project_s,p.project_flop);
    stage("first probe (full)",p.first_s,p.first_flop);
    stage("limits",p.limits_s,0);
    stage("bounded probes",p.bounded_s,p.bound_flop+p.rescore_flop);
    stage("  of which bound GEMM",p.bounded_s*p.bound_gemm_cpu_s/scan_cpu,p.bound_flop);
    stage("  of which scan+rescore",p.bounded_s*p.bound_scan_cpu_s/scan_cpu,p.rescore_flop);
    std::printf("  candidates %.1f per row, survivors %.2f per row (%.2f%%)\n",double(p.candidates)/count,
                double(p.survivors)/count,100.0*double(p.survivors)/double(p.candidates));
  } else stage("all probes (full)",p.all_s,p.all_flop);
  stage("reduce + fallback",p.reduce_s,0);
  const unsigned long long useful=p.probe_flop+p.project_flop+p.first_flop+p.bound_flop+p.rescore_flop+p.all_flop;
  stage("assignment total",total_s-load_s,useful);

  std::vector<float> a(4096*d),b(4096*d),c(4096*4096);
  std::mt19937 random(1); std::normal_distribution<float> normal;
  for (float& v:a) v=normal(random); for (float& v:b) v=normal(random);
  double sgemm_s=0;
  {
    const BlasThreads threads;
    const int m=4096,n=4096,k=int(d); const float one=1,zero=0; const char t='T',nn='N';
    TrainingLinalg::get().gemm(&t,&nn,&m,&n,&k,&one,a.data(),&k,b.data(),&k,&zero,c.data(),&m);
    started=profile_now();
    for (int r=0;r<5;++r) TrainingLinalg::get().gemm(&t,&nn,&m,&n,&k,&one,a.data(),&k,b.data(),&k,&zero,c.data(),&m);
    sgemm_s=(profile_now()-started)/5;
  }
  std::printf("peaks (%d threads): sgemm 4096x4096x%zu %.1f GFLOP/s",omp_get_max_threads(),d,2.0*4096*4096*d/1e9/sgemm_s);
  const auto half_panels=[](size_t depth) {
    HalfPanels p; p.depth=depth; p.columns=288; p.data=pack_panel_values<uint16_t>(depth,288,[](size_t k,size_t c) {
      return half_encode(0.001*double((k*31+c*17)%97)); });
    return p;
  };
  const auto float_panels=[](size_t depth) {
    return pack_panels(depth,288,[](size_t k,size_t c) { return float(0.001*double((k*31+c*17)%97)); });
  };
  std::printf(", binary16 panel kernel (depth %zu) %.1f GFLOP/s",d,panel_peak(d,[](const float* r,const HalfPanels& p,float* o,size_t dd,size_t w) {
    half_matrix(r,CoarseQuantizer::block_rows,dd,p,o,w); },half_panels));
  if (bound) std::printf(", FP32 panel kernel (depth %zu) %.1f GFLOP/s",bound,panel_peak(bound,[](const float* r,const TieredQPanels& p,float* o,size_t dd,size_t w) {
    float_matrix(r,CoarseQuantizer::block_rows,dd,p,o,w); },float_panels));
  std::printf("\n");
  return 0;
}
