// Dumps query_weights outputs bitwise for cross-build comparison (baseline vs new library).
#include <tieredq/tieredq.h>
#include <cstdio>
#include <cstdlib>
#include <random>
int main(int argc,char** argv){
  const uint32_t dims=std::atoi(argv[1]),centroids=std::atoi(argv[2]);const size_t nq=21,train_rows=4096;
  std::mt19937 rng(11);std::normal_distribution<float> n;
  std::vector<float> train(train_rows*dims),qs(nq*dims);for(auto&v:train)v=n(rng);for(auto&v:qs)v=n(rng);
  auto codec=dv::TieredQ::train({dims,dims*6,centroids,true,8},train.data(),train_rows);
  FILE* f=std::fopen(argv[3],"wb");
  for(size_t q=0;q<nq;++q){auto w=codec.query_weights(qs.data()+q*dims);
    for(auto* v:{&w.weights,&w.centroid_scores,&w.correction_weights}) std::fwrite(v->data(),4,v->size(),f);}
  std::fclose(f);
}
