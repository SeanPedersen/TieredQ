// TieredQ persistence, independent scoring, SIMD and index behavior.
#include "tieredq/tieredq.h"
#include "tieredq_layout.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unistd.h>
using namespace dv;
void check(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f) { bool rejected=false; try { f(); } catch(const std::exception&) { rejected=true; } check(rejected,"expected rejection"); }
std::vector<uint8_t> read(const std::string& path) { std::ifstream in(path,std::ios::binary); return {std::istreambuf_iterator<char>(in),{}}; }
void write(const std::string& path,const std::vector<uint8_t>& data) { std::ofstream out(path,std::ios::binary); out.write(reinterpret_cast<const char*>(data.data()),data.size()); }
uint64_t checksum(const uint8_t* data,size_t n) {
  uint64_t h=14695981039346656037ULL; for(size_t i=0;i<n;++i) { h^=data[i]; h*=1099511628211ULL; } return h;
}
void u32_at(std::vector<uint8_t>& data,size_t pos,uint32_t v) { for(int i=0;i<4;++i) data.at(pos+i)=uint8_t(v>>(8*i)); }
void fix_model_checksum(std::vector<uint8_t>& data) {
  auto h=checksum(data.data()+80,data.size()-80); for(int i=0;i<8;++i) data[56+i]=uint8_t(h>>(8*i));
}
struct Model {
  std::vector<uint8_t> data; size_t pos=80;
  uint32_t u32() { uint32_t v=0; for(int i=0;i<4;++i) v|=uint32_t(data.at(pos++))<<(8*i); return v; }
  uint64_t u64() { uint64_t lo=u32(); return lo|(uint64_t(u32())<<32); }
  float f32() { return std::bit_cast<float>(u32()); }
  std::vector<float> floats() { auto n=u64(); std::vector<float> v(n); for(auto& x:v) x=f32(); return v; }
};
// Independent coordinate dot product from the persisted model and the encoded bytes.
std::vector<float> reference(const TieredQ& codec,const std::string& path,const std::vector<uint8_t>& codes,const float* query,size_t n) {
  Model r{read(path)}; unsigned d=r.u32(); r.u32(); unsigned k=r.u32();
  r.u32(); r.u32(); r.u32(); r.u32(); unsigned nt=r.u32();
  std::vector<TieredQTier> tiers;
  for(unsigned i=0;i<nt;++i) tiers.push_back({r.u32(),r.u32(),r.u32(),r.u32(),r.u32()});
  auto centers=r.floats(); auto basis=r.floats(),mean=r.floats(),sigma=r.floats();
  std::vector<std::vector<float>> levels; unsigned nl=r.u32(); for(unsigned i=0;i<nl;++i) levels.push_back(r.floats());
  std::vector<double> transformed(sigma.size(),0); for(unsigned i=0;i<d;++i) for(size_t j=0;j<sigma.size();++j) transformed[j]+=query[i]*double(basis[i*sigma.size()+j]);
  std::vector<float> out(n);
  for(size_t i=0;i<n;++i) {
    const uint8_t* doc=codes.data()+i*codec.bytes_per_vector(); auto cid=doc[codec.code_bytes()]; check(cid<k,"centroid out of range");
    double score=0; for(unsigned j=0;j<d;++j) score+=double(query[j])*centers[size_t(cid)*d+j];
    for(size_t j=0;j<sigma.size();++j) score+=transformed[j]*mean[j];
    for(const auto& t:tiers) for(unsigned j=0;j<t.dimensions;++j) {
      unsigned code=0; for(unsigned b=0;b<t.bits;++b) { unsigned offset=t.bit_offset+j*t.bits+b; code|=((doc[offset/8]>>(offset%8))&1)<<b; }
      double s=tieredq_detail::Layout{codec.code_bytes()}.factor(doc);
      score+=s*transformed[t.start+j]*sigma[t.start+j]*levels[t.bits][code];
    }
    double qc=0,cc=0; for(unsigned j=0;j<d;++j) { const double c=centers[size_t(cid)*d+j]; qc+=query[j]*c; cc+=c*c; }
    score+=(cc>0 ? qc/cc : 0)*tieredq_detail::Layout{codec.code_bytes()}.correction(doc);
    out[i]=float(score);
  }
  return out;
}
// Centroid corrections equal c.(x - decode(x)) for each row's persisted centroid c.
void check_corrections(const TieredQ& codec,const std::string& path,const std::vector<uint8_t>& codes,const float* docs,size_t n) {
  std::vector<uint8_t> again(codes.size()); std::vector<float> corrections(n);
  codec.encode(docs,n,again.data(),corrections.data()); check(again==codes,"corrections change codes");
  Model r{read(path)}; unsigned d=r.u32(); r.u32(); r.u32(); r.u32(); r.u32(); r.u32(); r.u32(); unsigned nt=r.u32();
  for(unsigned i=0;i<nt*5;++i) r.u32();
  auto centers=r.floats(); std::vector<float> decoded(n*d); codec.decode(codes.data(),n,decoded.data());
  for(size_t i=0;i<n;++i) {
    const float* c=centers.data()+size_t(codes[i*codec.bytes_per_vector()+codec.code_bytes()])*d;
    double expected=0,magnitude=0;
    for(unsigned j=0;j<d;++j) { expected+=double(c[j])*(docs[i*d+j]-decoded[i*d+j]); magnitude+=std::fabs(double(c[j])*docs[i*d+j]); }
    check(std::fabs(expected-corrections[i])<=1e-3*(1+magnitude),"centroid correction mismatch");
  }
}
std::vector<TieredQHit> ranked_plane_scores(const TieredQ& codec,const std::vector<uint8_t>& codes,
                                           const TieredQQuery& query,TieredQKernel kernel) {
  const size_t count=codes.size()/codec.bytes_per_vector();
  std::vector<uint8_t> planes((count+15)/16*16*codec.bytes_per_vector());
  std::vector<float> scores(count),scalar(count);
  codec.pack(codes.data(),count,planes.data());
  codec.score_planes(planes.data(),count,query,scores.data(),kernel);
  codec.score(codes.data(),count,query,scalar.data(),TieredQKernel::scalar);
  std::vector<TieredQHit> expected;
  for(size_t i=0;i<count;++i) {
    check(std::fabs(scores[i]-scalar[i])<=1e-4f*(1+std::fabs(scalar[i])),"plane coordinate score parity");
    expected.push_back({scores[i],uint32_t(i)});
  }
  std::sort(expected.begin(),expected.end(),[](auto x,auto y){return x.score>y.score || (x.score==y.score && x.id<y.id);});
  return expected;
}
void exercise(unsigned d,unsigned bits,bool rotate,const std::string& prefix) {
  const size_t rows=std::max<size_t>(64,2*d); constexpr size_t n=537; std::mt19937 rng(d+bits); std::normal_distribution<float> normal;
  std::vector<float> sample(rows*d),docs(n*d),q(d);
  for(auto* data:{&sample,&docs,&q}) for(auto& v:*data) v=normal(rng)/std::sqrt(float(d));
  TieredQConfig cfg{d,bits,4,rotate}; auto codec=TieredQ::train(cfg,sample.data(),rows,17);
  uint64_t budget=0; for(auto t:codec.tiers()) budget+=uint64_t(t.dimensions)*t.bits;
  check(budget==bits,"allocation exceeds budget"); check(codec.code_bytes()==(bits+7)/8,"code size mismatch");
  if(d==512 && bits==512) {
    const auto& ts=codec.tiers(); check(ts.size()==3 && ts[0].bits==4 && ts[0].dimensions==64 && ts[1].bits==2 && ts[1].dimensions==64 && ts[2].bits==1 && ts[2].dimensions==128,"fixed schedule regression");
    check(codec.bytes_per_vector()==69,"payload size regression");
  }
  const auto codes=codec.encode(docs.data(),n); const auto lut=codec.query(q.data()); std::vector<float> scalar(n),simd(n);
  codec.score(codes.data(),n,lut,scalar.data(),TieredQKernel::scalar);
  codec.save(prefix+".codec"); auto loaded=TieredQ::load(prefix+".codec");
  check(loaded.encode(docs.data(),n)==codes,"model reopen changes codes");
  const auto ref=reference(codec,prefix+".codec",codes,q.data(),n);
  check_corrections(codec,prefix+".codec",codes,docs.data(),n);
  for(size_t i=0;i<n;++i) check(std::fabs(ref[i]-scalar[i])<=2e-5f*(1+std::fabs(ref[i])),"scorer differs from coordinate dot");
  for(auto kernel:{TieredQKernel::automatic,TieredQKernel::avx2,TieredQKernel::avx512}) {
    if(!TieredQ::kernel_available(kernel)) { rejects([&] { codec.score(codes.data(),n,lut,simd.data(),kernel); }); continue; }
    codec.score(codes.data(),n,lut,simd.data(),kernel); check(simd==scalar,"SIMD scores differ from scalar");
  }
  loaded.score(codes.data(),n,loaded.query(q.data()),simd.data(),TieredQKernel::scalar); check(simd==scalar,"reopened scores differ");
  TieredQIndex index(codec); index.add(docs.data(),3); index.add(docs.data()+3*d,n-3); index.save(prefix+".index");
  auto reopened=TieredQIndex::load(prefix+".index"); auto a=index.search(q.data(),n),b=reopened.search(q.data(),n);
  check(a.size()==n && b.size()==n,"top-k size mismatch");
  const auto expected=ranked_plane_scores(codec,codes,lut,TieredQKernel::automatic);
  for(size_t i=0;i<n;++i) check(a[i].id==b[i].id && a[i].score==b[i].score && a[i].id==expected[i].id,"search parity failure");
  for(auto kernel:{TieredQKernel::scalar,TieredQKernel::neon,TieredQKernel::avx2,TieredQKernel::avx512}) {
    if(!TieredQ::kernel_available(kernel)) continue;
    const auto kernel_expected=ranked_plane_scores(codec,codes,lut,kernel);
    for(size_t k:{1u,17u,100u,537u}) {
      auto hits=reopened.search(q.data(),k,kernel);
      for(size_t i=0;i<hits.size();++i)
        check(hits[i].id==kernel_expected[i].id && hits[i].score==kernel_expected[i].score,"transposed search score parity failure");
    }
  }
  check(index.payload_bytes()==codes.size(),"logical payload accounting changed");
  auto copied=index; auto moved=std::move(copied);
  check(copied.size()==0 && moved.size()==n,"move constructor changes index count");
  copied=std::move(moved);
  check(moved.size()==0 && copied.size()==n,"move assignment changes index count");
  auto moved_hits=copied.search(q.data(),n);
  for(size_t i=0;i<n;++i) check(moved_hits[i].id==expected[i].id && moved_hits[i].score==expected[i].score,"move changes search scores");
  check(index.search(q.data(),0).empty(),"zero top-k failed");
  auto invalid=lut; invalid.offsets[0]=UINT32_MAX; rejects([&]{codec.score(codes.data(),n,invalid,simd.data());});
  auto corrupt=read(prefix+".index"); corrupt.back()^=1; write(prefix+".bad",corrupt); rejects([&]{TieredQIndex::load(prefix+".bad");});
  corrupt=read(prefix+".codec"); corrupt[8]=2; write(prefix+".bad",corrupt); rejects([&]{TieredQ::load(prefix+".bad");});
  corrupt=read(prefix+".codec"); u32_at(corrupt,80+32+12,2); fix_model_checksum(corrupt);
  write(prefix+".bad",corrupt); rejects([&]{TieredQ::load(prefix+".bad");}); // scale group out of range
  corrupt=read(prefix+".codec"); u32_at(corrupt,80+32+16,1); fix_model_checksum(corrupt);
  write(prefix+".bad",corrupt); rejects([&]{TieredQ::load(prefix+".bad");}); // invalid bit offset
  corrupt=read(prefix+".codec"); u32_at(corrupt,80+32+codec.tiers().size()*20,UINT32_MAX);
  fix_model_checksum(corrupt); write(prefix+".bad",corrupt); rejects([&]{TieredQ::load(prefix+".bad");});
  corrupt.resize(79); write(prefix+".bad",corrupt); rejects([&]{TieredQ::load(prefix+".bad");});
  rejects([&]{TieredQ::load(prefix+".index");}); rejects([&]{TieredQIndex::load(prefix+".codec");});
  std::cout<<"PASS D="<<d<<" bits="<<bits<<" rotate="<<rotate<<'\n';
}
// A diagonal covariance forces uniform 5/6-bit tiers and exercises every table
// half in the exact register-permutation path, independently of greedy fitting.
void wide_register_parity(unsigned bits) {
  constexpr unsigned d=64; constexpr size_t n=97;
  std::vector<float> sample(2*d*d,0),docs(n*d),query(d);
  for(unsigned i=0;i<d;++i) { sample[i*d+i]=1; sample[(d+i)*d+i]=-1; }
  std::mt19937 rng(bits); std::normal_distribution<float> normal;
  for(auto& v:docs) v=normal(rng); for(auto& v:query) v=normal(rng);
  auto codec=TieredQ::train(TieredQConfig{d,d*bits,1,false},sample.data(),2*d);
  check(codec.tiers().size()==1 && codec.tiers()[0].bits==bits,"uniform wide allocation failed");
  auto encoded=codec.encode(docs.data(),n);
  if (TieredQ::kernel_available(TieredQKernel::neon)) {
    const auto q=codec.query(query.data());
    for (size_t count:{1u,15u,16u,17u,31u,32u,33u,97u}) {
      std::vector<uint8_t> subset(encoded.begin(),encoded.begin()+count*codec.bytes_per_vector());
      ranked_plane_scores(codec,subset,q,TieredQKernel::neon);
    }
    auto ungrouped=q; ungrouped.scan_groups.clear(); ungrouped.scan_tables.clear();
    ranked_plane_scores(codec,encoded,ungrouped,TieredQKernel::neon);
    auto corrupt=encoded;
    corrupt[codec.code_bytes()]=255;
    std::vector<uint8_t> planes((n+15)/16*16*codec.bytes_per_vector());
    std::vector<float> scores(n);
    codec.pack(corrupt.data(),n,planes.data());
    rejects([&]{codec.score_planes(planes.data(),n,q,scores.data(),TieredQKernel::neon);});
  }
  TieredQIndex index(codec); index.add(docs.data(),n);
  for(auto kernel:{TieredQKernel::scalar,TieredQKernel::neon,TieredQKernel::avx2,TieredQKernel::avx512}) {
    if(!TieredQ::kernel_available(kernel)) continue;
    const auto expected=ranked_plane_scores(codec,encoded,codec.query(query.data()),kernel);
    auto hits=index.search(query.data(),n,kernel);
    for(size_t i=0;i<n;++i) check(hits[i].id==expected[i].id && hits[i].score==expected[i].score,"wide search differs from full plane sort");
  }
}
// Unsupported recipes must fail explicitly rather than reinterpret old metadata.
void reject_unsupported_format(const std::string& prefix) {
  auto fixture=std::filesystem::path(__FILE__).parent_path()/"fixtures/tieredq_codec.codec";
  auto data=read(fixture.string());
  u32_at(data,80+4*4,1); fix_model_checksum(data);
  write(prefix+".old",data);
  rejects([&]{TieredQ::load(prefix+".old");});
  auto codec=TieredQ::load(fixture.string()); codec.save(prefix+".golden");
  check(read(prefix+".golden")==read(fixture.string()),"golden bytes changed");
}
void tieredq_model_tests(const std::string& prefix);
int main() {
  auto dir=std::filesystem::temp_directory_path()/("tieredq_test_"+std::to_string(getpid())); std::filesystem::create_directory(dir);
  try {
    rejects([]{TieredQConfig{0,512}.validate();}); rejects([]{TieredQConfig{64,513}.validate();});
    rejects([]{TieredQConfig{64,0}.validate();}); rejects([]{TieredQConfig::target_bytes(512,UINT32_MAX);});
    rejects([]{TieredQConfig{512,512,257}.validate();});
    check(TieredQConfig::target_bytes_per_vector(512,64).code_bits==512,"byte factory mismatch");
    auto prefix=(dir/"model").string();
    reject_unsupported_format(prefix);
    tieredq_model_tests(prefix);
    for (unsigned bits=1;bits<=8;++bits) wide_register_parity(bits);
    for(unsigned d:{1u,7u,65u,384u,511u,512u,768u,1024u,1536u}) exercise(d,std::min(257u,d*8),d<=65,prefix);
    for(unsigned bytes:{32u,48u,64u,96u,128u}) exercise(512,bytes*8,bytes==64,prefix);
    exercise(768,512,true,prefix); exercise(768,1024,false,prefix);
    exercise(7,1,true,prefix); exercise(7,56,false,prefix);
    std::vector<float> zeros(8*7,0); auto zero=TieredQ::train(TieredQConfig{7,13,1,false},zeros.data(),8);
    TieredQIndex ties(zero); ties.add(zeros.data(),8); auto tied=ties.search(zeros.data(),3);
    for(size_t i=0;i<tied.size();++i) check(tied[i].id==i && tied[i].score==0,"tie ordering failure");
    auto bad_input=zeros; bad_input[0]=INFINITY; rejects([&]{zero.encode(bad_input.data(),1);});
    rejects([&]{zero.query(bad_input.data());});
    rejects([&]{TieredQ::train(TieredQConfig{7,13,4},zeros.data(),7);});
    std::filesystem::remove_all(dir); std::cout<<"OK\n"; return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; std::filesystem::remove_all(dir); return 1; }
}
