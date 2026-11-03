// Tiled batch projection on the refinement's int8 layout [block of 16 panels][input][panel in block][16]:
// Q queries x P panels of a block per pass; per thread chunk of queries, blocks outer so each block stays in L2.
// Build: g++ -O3 -march=native -ffp-contract=off -std=c++20 tile.cpp -o tile; run: ./tile CHUNK. Checks bit equality.
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
constexpr size_t lanes=16,block=16;
template<size_t P,size_t Q>
void tile(const int8_t* base,size_t width,size_t d,const float* x,size_t xs,float* out,size_t os) {
  __m512 acc[Q][P];
  for (auto& r:acc) for (auto& a:r) a=_mm512_setzero_ps();
  for (size_t i=0;i<d;++i) {
    __m512 p[P];
    for (size_t b=0;b<P;++b) p[b]=_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(base+(i*width+b)*lanes))));
    for (size_t q=0;q<Q;++q) { const __m512 v=_mm512_set1_ps(x[q*xs+i]); for (size_t b=0;b<P;++b) acc[q][b]=_mm512_fmadd_ps(v,p[b],acc[q][b]); }
  }
  for (size_t q=0;q<Q;++q) for (size_t b=0;b<P;++b) _mm512_storeu_ps(out+q*os+b*lanes,acc[q][b]);
}
int main(int argc,char** argv) {
  const size_t d=1536,ranks=1056,blocks=ranks/(lanes*block),chunk=argc>1?std::stoul(argv[1]):32,rounds=200;
  std::mt19937 rng(0); std::normal_distribution<float> n;
  std::vector<int8_t> panels(d*ranks); for (auto& p:panels) p=int8_t(std::clamp(std::lround(n(rng)*40),-127L,127L));
  std::vector<float> x(chunk*d),out(chunk*ranks),ref(chunk*ranks); for (auto& v:x) v=n(rng);
  const size_t bs=d*block*lanes;
  auto single=[&] { for (size_t q=0;q<chunk;++q) for (size_t g=0;g<blocks;++g) tile<16,1>(panels.data()+g*bs,block,d,x.data()+q*d,d,ref.data()+q*ranks+g*256,ranks); };
  auto batch=[&] { for (size_t g=0;g<blocks;++g) for (size_t q=0;q<chunk;q+=8) for (size_t b=0;b<block;b+=2)
    tile<2,8>(panels.data()+g*bs+b*lanes,block,d,x.data()+q*d,d,out.data()+q*ranks+g*256+b*lanes,ranks); };
  auto time=[&](auto f) { f(); auto s=std::chrono::steady_clock::now(); for (size_t r=0;r<rounds;++r) f();
    return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-s).count()/double(rounds*chunk); };
  const double a=time(single),b=time(batch);
  std::printf("chunk %zu: single %.1f us/query, tiled 8x2 %.1f us/query, identical %d\n",chunk,a,b,int(std::memcmp(out.data(),ref.data(),out.size()*4)==0));
}
