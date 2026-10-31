// Issue #19.3: peak RSS growth per build stage: sample copy, train, flat add or IVF build. Usage: build_peak N D BITS METRIC(ip|cos) MODE(flat|ivf)
#include "tieredq/tieredq.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
long hwm_kb() { std::ifstream f("/proc/self/status"); std::string k; long v=0; while (f>>k) { if (k=="VmHWM:") { f>>v; return v; } } return 0; }
void reset_hwm() { std::ofstream("/proc/self/clear_refs")<<"5"; }
int main(int argc,char** argv) {
  const size_t n=std::stoul(argv[1]),d=std::stoul(argv[2]),bits=std::stoul(argv[3]);
  const bool cosine=!std::strcmp(argv[4],"cos"),ivf=!std::strcmp(argv[5],"ivf"),sampled=argc>6;
  std::vector<float> data(n*d); std::mt19937 rng(1); std::normal_distribution<float> normal;
  for (size_t i=0;i<n*d;++i) data[i]=normal(rng)*std::exp(-float(i%d)/float(d)*4);
  const double per=8.0*1024/double(n);
  reset_hwm(); long base=hwm_kb();
  const size_t sample_rows=std::min<size_t>(n,std::getenv("TRAIN_ROWS") ? std::stoul(std::getenv("TRAIN_ROWS")) : 32768);
  std::vector<size_t> order(n); std::iota(order.begin(),order.end(),0); std::mt19937_64 r(0); std::shuffle(order.begin(),order.end(),r);
  std::vector<uint32_t> ids(order.begin(),order.begin()+sample_rows);
  std::vector<float> sample(sampled ? 0 : sample_rows*d);
  if (!sampled) for (size_t i=0;i<sample_rows;++i) std::copy_n(data.data()+order[i]*d,d,sample.data()+i*d);
  long after_copy=hwm_kb();
  auto config=dv::TieredQConfig::target_code_bits(uint32_t(d),uint32_t(bits*d));
  if (cosine) config.metric=dv::TieredQMetric::cosine;
  reset_hwm(); long before_train=hwm_kb();
  const auto train_start=std::chrono::steady_clock::now();
  auto codec=sampled ? dv::TieredQ::train_sampled(config,data.data(),n,ids.data(),sample_rows,0)
                     : dv::TieredQ::train(config,sample.data(),sample_rows,0);
  const double train_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-train_start).count();
  long train_peak=hwm_kb(), after_train_rss=0;
  { std::ifstream f("/proc/self/status"); std::string k; while (f>>k) if (k=="VmRSS:") { f>>after_train_rss; break; } }
  reset_hwm(); long before_add=hwm_kb();
  const auto add_start=std::chrono::steady_clock::now();
  size_t stored=0;
  if (ivf) { dv::TieredQIVFIndex index(codec,256); index.build(data.data(),n); stored=index.storage_bytes(); }
  else { dv::TieredQIndex index(codec); index.add(data.data(),n); stored=index.storage_bytes(); }
  long add_peak=hwm_kb();
  const double add_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-add_start).count();
  std::printf("n=%zu d=%zu bits=%zu %s %s%s: sample copy %.0f, train peak +%.0f, add/build peak +%.0f, index %.0f bits/vector, train %.2f s, add %.2f s\n",
              n,d,bits,argv[4],argv[5],sampled ? " sampled" : "",(after_copy-base)*per,(train_peak-before_train)*per,(add_peak-before_add)*per,8.0*double(stored)/double(n),train_s,add_s);
}
