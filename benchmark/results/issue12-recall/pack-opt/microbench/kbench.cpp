// Single-core code-lookup kernel cost on hand-built tiers; compile against baseline or new sources (same API).
// usage: kbench "bits:dims,bits:dims,..." docs
#include "tieredq_scan_codes_x86.h"
#include "tieredq_train_codebook_data.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
using namespace dv::tieredq_detail;
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc,char** argv) {
  constexpr size_t tile=4096,centroids=256,queries=32;
  std::vector<dv::TieredQTier> tiers;uint32_t dims=0,bits=0;
  for (char* part=std::strtok(argv[1],",");part;part=std::strtok(nullptr,",")) {
    const uint32_t b=std::atoi(part),d=std::atoi(std::strchr(part,58)+1);
    tiers.push_back({dims,d,b,0,bits});dims+=d;bits+=d*b;
  }
  const size_t docs=std::atoll(argv[2]),code_bytes=(bits+7)/8,stride=code_bytes+dv::TieredQ::metadata_bytes;
  std::mt19937 rng(3);std::normal_distribution<float> n;
  std::vector<uint8_t> rows(docs*stride,0);const Layout layout{code_bytes};
  for (size_t doc=0;doc<docs;++doc) {
    uint8_t* row=rows.data()+doc*stride;
    for (const auto& t:tiers) for (size_t j=0;j<t.dimensions;++j) write_code(row,t.bit_offset+j*t.bits,t.bits,rng()%(1u<<t.bits));
    row[code_bytes]=uint8_t(rng()%centroids);layout.store_half(row,layout.factor_offset(),1);layout.store_half(row,layout.correction_offset(),.5);
  }
  const auto& books=gaussian_codebooks();const std::vector<std::vector<float>> levels(books.begin(),books.end());
  auto cache=build_fast_scan_cache(levels,tiers,dims,docs,stride,code_bytes,centroids,fast_code_lookup_levels,rows.data());
  std::vector<dv::TieredQQuery> qs(queries);
  for (auto& q:qs) { q.weights.resize(dims);for (auto& w:q.weights) w=n(rng);q.centroid_scores.assign(centroids,.1f);q.correction_weights.assign(centroids,.2f); }
  std::vector<float> scores(queries*tile);double sink=0;
  auto pass=[&](size_t rows_) {
    for (size_t first=0;first<docs;first+=tile) {
      const size_t count=std::min(tile,docs-first);
      scan_codes_avx512_tile(*cache,rows_,first,count,scores.data(),tile,[&](size_t r)->const dv::TieredQQuery& {return qs[r];});
      sink+=scores[0];
    }
  };
  std::printf("tiers %s dims %u code_bits %u levels_bytes_per_doc %.2f\n",argv[1],dims,bits,double(cache->levels.size())/(cache->blocks*16));
  for (size_t rows_:{size_t(1),queries}) {
    pass(rows_);double best=1e9;
    for (int rep=0;rep<7;++rep) { const double t0=now_s();pass(rows_);best=std::min(best,now_s()-t0); }
    std::printf("rows %zu: %.3f ns per doc*dim*query (%.2f ms per pass)\n",rows_,best*1e9/(docs*dims*rows_),best*1e3);
  }
  std::printf("sink %g\n",sink);
}
