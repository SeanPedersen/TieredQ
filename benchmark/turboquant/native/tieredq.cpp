// Fast TieredQ-only benchmark binding with exact integer budgets and reload accounting.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <tieredq/tieredq.h>
#include <numeric>
#include <random>
#include <filesystem>
#include <chrono>
#include <omp.h>
#if defined(__OPTIMIZE_SIZE__)
#error "Benchmark requires -O3"
#endif
namespace nb=nanobind;
using Matrix=nb::ndarray<const float,nb::ndim<2>,nb::c_contig,nb::device::cpu>;
using Ids=nb::ndarray<uint32_t,nb::ndim<2>,nb::c_contig,nb::device::cpu>;
constexpr size_t top_k=10, training_limit=32768, centroids=256;
class Index {
 dv::TieredQIndex index_;
 size_t dim_;
 double encode_s_=0;
 static dv::TieredQIndex build(Matrix base,uint32_t budget) {
  const size_t n=base.shape(0),d=base.shape(1),count=std::min(n,training_limit);
  auto config=dv::TieredQConfig::target_code_bits(d,budget);config.num_centroids=std::min(centroids,n/2);
  std::vector<size_t> order(n);std::iota(order.begin(),order.end(),0);std::mt19937_64 rng(0);std::shuffle(order.begin(),order.end(),rng);
  std::vector<float> sample(count*d);for(size_t i=0;i<count;++i)std::copy_n(base.data()+order[i]*d,d,sample.data()+i*d);
  return dv::TieredQIndex(dv::TieredQ::train(config,sample.data(),count,0),true);
 }
 public:
 Index(Matrix base,uint32_t budget):index_(build(base,budget)),dim_(base.shape(1)) {
  auto start=std::chrono::steady_clock::now();index_.add(base.data(),base.shape(0));encode_s_=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 }
 Index(const std::string& path,size_t dim):index_(dv::TieredQIndex::load(path)),dim_(dim){}
 void save(const std::string& path)const{index_.save(path);}
 double encode_seconds()const{return encode_s_;}
 void search(Matrix q,Ids ids,int threads)const{
  if(q.shape(1)!=dim_||ids.shape(0)!=q.shape(0)||ids.shape(1)!=top_k||(threads!=1&&threads!=8))throw std::invalid_argument("search shape/threads");
  omp_set_num_threads(threads);std::vector<float> scores(q.shape(0)*top_k);index_.search_batch(q.data(),q.shape(0),top_k,scores.data(),ids.data());
 }
 nb::dict details()const{
  const auto& c=index_.codec();nb::dict out;nb::list tiers;uint32_t learned=0;
  for(const auto&t:c.tiers()){nb::dict x;x["dimensions"]=t.dimensions;x["bits"]=t.bits;tiers.append(x);learned+=t.dimensions*t.bits;}
  out["tiers"]=tiers;out["learned_code_bits"]=learned;out["index_allocated_bytes"]=index_.storage_bytes();out["model_bytes"]=c.memory_bytes();out["shell_bytes"]=sizeof(Index)-sizeof(dv::TieredQ);out["resident_bytes"]=index_.storage_bytes()+c.memory_bytes()+sizeof(Index)-sizeof(dv::TieredQ);out["packed_code_bytes"]=c.code_bytes();out["vector_bytes"]=index_.bytes_per_vector();return out;
 }
};
NB_MODULE(tieredq_bench,m){
 m.def("set_threads",[](int n){omp_set_dynamic(0);omp_set_num_threads(n);});
 nb::class_<Index>(m,"Index").def(nb::init<Matrix,uint32_t>(),nb::call_guard<nb::gil_scoped_release>()).def(nb::init<const std::string&,size_t>()).def("save",&Index::save).def("details",&Index::details).def_prop_ro("encode_seconds",&Index::encode_seconds).def("search",&Index::search,nb::call_guard<nb::gil_scoped_release>());
}
