// Accumulation loop shapes over a decoded block: memory-broadcast weights vs register reuse across blocks.
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <vector>
using Clock=std::chrono::steady_clock;
template<int R,int B> __attribute__((target("avx512f"),noinline))
void acc(const float* dec,const float* w,size_t dims,float* out){
  __m512 a[B][R]; for(auto&x:a) for(auto&y:x) y=_mm512_setzero_ps();
  for(size_t r=0;r<dims;++r){ __m512 v[B]; for(int b=0;b<B;++b) v[b]=_mm512_loadu_ps(dec+(b*dims+r)*16);
    for(int q=0;q<R;++q){ const __m512 s=_mm512_set1_ps(w[r*R+q]); for(int b=0;b<B;++b) a[b][q]=_mm512_add_ps(a[b][q],_mm512_mul_ps(v[b],s)); } }
  for(int b=0;b<B;++b) for(int q=0;q<R;++q) _mm512_storeu_ps(out+(b*R+q)*16,a[b][q]);
}
template<int R,int B> void run(size_t dims){
  std::vector<float> dec(B*dims*16,1.0f),w(dims*R,0.5f),out(B*R*16);
  const int reps=200000/ int(dims)*20; auto t=Clock::now();
  for(int i=0;i<reps;++i) acc<R,B>(dec.data(),w.data(),dims,out.data());
  double s=std::chrono::duration<double>(Clock::now()-t).count();
  std::printf("R%d B%d: %.3f ns per (rank,block,row) -> %.2f cyc@3.4\n",R,B,s*1e9/(double(reps)*dims*R*B),s*3.4e9/(double(reps)*dims*R*B));
}
int main(){ for(size_t d:{132,1536}){ std::printf("dims %zu\n",d); run<8,1>(d); run<12,1>(d); run<16,1>(d); run<4,2>(d); run<6,2>(d); run<4,3>(d); run<8,2>(d);} }
