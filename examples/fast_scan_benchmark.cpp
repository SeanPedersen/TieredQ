// Single-thread end-to-end exact/fast comparison using an existing index and held-out float32 queries.
// Separate mode measures warm steady state; alternating mode also exposes competition between both caches.
// Optional truth.u32 contains ten little-endian uint32 neighbor ids per selected query, starting at row zero.
#include <tieredq/tieredq.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock=std::chrono::steady_clock;
constexpr size_t top_k=10,warmup_queries=64,default_queries=1000,default_query_start=108000;
struct Measurements {
  std::vector<double> exact,fast;
  size_t overlap=0,exact_truth=0,fast_truth=0;
};
double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
double percentile(std::vector<double> values,double fraction) {
  std::sort(values.begin(),values.end());
  return values[size_t(fraction*(values.size()-1))];
}
template<class T>
std::vector<T> read_rows(const std::string& path,size_t first,size_t count,size_t width) {
  std::ifstream file(path,std::ios::binary);
  file.seekg(std::streamoff(first)*std::streamoff(width*sizeof(T)));
  std::vector<T> rows(count*width);
  if (!file.read(reinterpret_cast<char*>(rows.data()),std::streamsize(rows.size()*sizeof(T))))
    throw std::runtime_error("cannot read the requested rows");
  return rows;
}
size_t truth_overlap(const std::vector<dv::TieredQHit>& hits,const uint32_t* truth) {
  return std::count_if(hits.begin(),hits.end(),[&](const auto& hit){
    return std::find(truth,truth+top_k,hit.id)!=truth+top_k;
  });
}

size_t intersection(const std::vector<dv::TieredQHit>& exact,const std::vector<dv::TieredQHit>& fast) {
  size_t found=0;
  for (const auto& hit:exact)
    found+=std::any_of(fast.begin(),fast.end(),[&](const auto& other){return hit.id==other.id;});
  return found;
}
Measurements measure(const dv::TieredQIndex& index,const std::vector<float>& queries,size_t count,bool alternating,
    const std::vector<uint32_t>& truth) {
  const size_t dimensions=index.codec().config().dimensions;
  std::vector<std::vector<dv::TieredQHit>> reference(count);
  Measurements result;
  if (!alternating) {
    for (size_t i=0;i<count+warmup_queries;++i) {
      const auto start=Clock::now();
      reference[i%count]=index.search(queries.data()+i%count*dimensions,top_k);
      const double elapsed=elapsed_ms(start);
      if (i>=warmup_queries) {
        result.exact.push_back(elapsed);
        if (!truth.empty()) result.exact_truth+=truth_overlap(reference[i%count],truth.data()+i%count*top_k);
      }
    }
  }
  for (size_t i=0;i<count+warmup_queries;++i) {
    const float* query=queries.data()+i%count*dimensions;
    if (alternating) {
      const auto start=Clock::now();
      reference[i%count]=index.search(query,top_k);
      const double elapsed=elapsed_ms(start);
      if (i>=warmup_queries) {
        result.exact.push_back(elapsed);
        if (!truth.empty()) result.exact_truth+=truth_overlap(reference[i%count],truth.data()+i%count*top_k);
      }
    }
    const auto start=Clock::now();
    const auto fast=index.search_fast(query,top_k);
    const double elapsed=elapsed_ms(start);
    if (i<warmup_queries) continue;
    result.fast.push_back(elapsed);
    result.overlap+=intersection(reference[i%count],fast);
    if (!truth.empty()) result.fast_truth+=truth_overlap(fast,truth.data()+i%count*top_k);
  }
  return result;
}
} // namespace
int main(int argc,char** argv) {
  try {
    if (argc<3 || argc>7)
      throw std::runtime_error("usage: tieredq_fast_scan_benchmark INDEX EMBEDDINGS [queries=1000] [separate|alternating] [first_query=108000] [truth.u32]");
    const size_t count=argc>3 ? std::stoul(argv[3]) : default_queries;
    const std::string mode=argc>4 ? argv[4] : "separate";
    const size_t first=argc>5 ? std::stoul(argv[5]) : default_query_start;
    if (!count || (mode!="separate" && mode!="alternating")) throw std::runtime_error("invalid count or timing mode");
    omp_set_dynamic(0); omp_set_num_threads(1);
    auto index=dv::TieredQIndex::load(argv[1]);
    if (index.size()<top_k) throw std::runtime_error("benchmark requires at least ten documents");
    const auto queries=read_rows<float>(argv[2],first,count,index.codec().config().dimensions);
    const auto truth=argc>6 ? read_rows<uint32_t>(argv[6],0,count,top_k) : std::vector<uint32_t>{};
    if (std::any_of(truth.begin(),truth.end(),[&](uint32_t id){return id>=index.size();}))
      throw std::runtime_error("ground-truth id outside index");
    const auto start=Clock::now(); index.build_fast_scan(); const double build=elapsed_ms(start);
    if (!index.has_fast_scan()) throw std::runtime_error("fast scan is unavailable");
    const auto result=measure(index,queries,count,mode=="alternating",truth);
    const double exact=percentile(result.exact,.5),fast=percentile(result.fast,.5);
    std::cout<<"threads=1 documents="<<index.size()<<" queries="<<count<<" first_query="<<first
        <<" coded_dimensions="<<index.codec().coded_dimensions()<<" mode="<<mode<<" cache_build_ms="<<build
        <<" exact_p50_ms="<<exact<<" fast_p50_ms="<<fast<<" fast_p95_ms="<<percentile(result.fast,.95)
        <<" speedup="<<exact/fast<<" overlap@10="<<double(result.overlap)/(count*top_k);
    if (!truth.empty()) std::cout<<" exact_recall@10="<<double(result.exact_truth)/(count*top_k)
        <<" fast_recall@10="<<double(result.fast_truth)/(count*top_k);
    std::cout<<'\n';
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n'; return 1;
  }
}
