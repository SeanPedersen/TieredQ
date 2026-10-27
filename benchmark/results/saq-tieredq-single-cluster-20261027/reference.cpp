// Independent adapter checks: upstream full-distance enumeration and a last-block regression.
#include <algorithm>
#include <fmt/format.h>
#include "index/ivf.hpp"
using namespace saqlib;
extern "C" void saq_reference(const char *model,const float *queries,size_t count,size_t dimensions,const float *basis,const float *mean,size_t original_dims,uint32_t *output) {
  IVF index;
  index.load(model);
  SearcherConfig config;
  config.dist_type=DistType::L2Sqr;
  constexpr size_t top_k=10;
  for (size_t i=0;i<count;++i) {
    FloatVec query=FloatVec::Zero(dimensions);
    query.head(original_dims)=(Eigen::Map<const FloatVec>(queries+i*original_dims,original_dims)-Eigen::Map<const FloatVec>(mean,original_dims))*Eigen::Map<const FloatRowMat>(basis,original_dims,original_dims);
    std::vector<std::pair<PID,float>> distances;
    index.estimate<DistType::L2Sqr>(query,1,config,distances);
    auto compare=[](const auto &a,const auto &b){return a.second<b.second || (a.second==b.second && a.first<b.first);};
    std::partial_sort(distances.begin(),distances.begin()+top_k,distances.end(),compare);
    for (size_t j=0;j<top_k;++j) output[i*top_k+j]=distances[j].first;
  }
}
extern "C" uint32_t saq_last_block_regression() {
  constexpr size_t count=64;
  FloatRowMat data=FloatRowMat::Zero(count,kDimPaddingSize);
  for (size_t i=0;i<count;++i) data(i,0)=float(i+1);
  FloatRowMat center=data.colwise().mean();
  std::vector<PID> assignments(count,0);
  QuantizeConfig config;
  config.avg_bits=8;
  config.enable_segmentation=false;
  config.single.random_rotation=false;
  IVF index(count,kDimPaddingSize,1,config);
  index.construct(data,center,assignments.data(),1);
  PID id;
  SearcherConfig search;
  index.search<DistType::L2Sqr>(data.row(count-1),1,1,search,&id);
  return id;
}

extern "C" void saq_layout(const char *model,size_t *output) {
  IVF index;index.load(model);
  const auto &data=*index.get_saq_data();
  output[0]=data.quant_plan.size();output[1]=index.num_dim();
  output[2]=sizeof(QuantizeConfig);output[3]=sizeof(QuantSingleConfig);
  for (size_t i=0;i<data.quant_plan.size();++i) {
    output[4+2*i]=data.quant_plan[i].first;output[5+2*i]=data.quant_plan[i].second;
  }
}
