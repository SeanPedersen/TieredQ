// Independent FP32 matrix references, fused centroid ties, projected correction, and overflow fallback.
#include "tieredq_encode.h"
#include <bit>
#include <random>
#include <stdexcept>

namespace {
using namespace dv;
using namespace dv::tieredq_detail;
void require(bool value,const char* message) { if (!value) throw std::runtime_error(message); }

float accumulate(float sum,float a,float b,TieredQKernel kernel) {
  return kernel==TieredQKernel::scalar ? sum+a*b : std::fma(a,b,sum);
}

void matrix_reference() {
  constexpr size_t depth=67,columns=65,guard=16;
  std::mt19937 rng(912); std::normal_distribution<float> normal;
  TieredQPanels panels; panels.depth=depth; panels.columns=columns;
  panels.data.resize(depth*panels.padded_columns());
  for (auto& value:panels.data) value=normal(rng);
  for (size_t count:{size_t(0),size_t(1),size_t(7),size_t(65)}) {
    std::vector<float> input(count*depth); for (auto& value:input) value=normal(rng);
    for (auto kernel:{TieredQKernel::scalar,TieredQKernel::avx2,TieredQKernel::avx512}) {
      if (!TieredQ::kernel_available(kernel)) continue;
      const size_t stride=panels.padded_columns();
      std::vector<float> output(count*stride+guard,123);
      float_matrix(input.data(),count,depth,panels,output.data(),stride,kernel);
      for (size_t i=0;i<count;++i) for (size_t c=0;c<stride;++c) {
        float expected=0;
        for (size_t j=0;j<depth;++j)
          expected=accumulate(expected,input[i*depth+j],panels.data[(c/TieredQPanels::PANEL*depth+j)*TieredQPanels::PANEL+c%TieredQPanels::PANEL],kernel);
        require(std::bit_cast<uint32_t>(output[i*stride+c])==std::bit_cast<uint32_t>(expected),"FP32 fused matrix reference");
      }
      for (size_t j=count*stride;j<output.size();++j) require(output[j]==123,"FP32 matrix tail overwrite");
    }
  }
}

void centroid_ties_and_correction() {
  constexpr size_t dimensions=3,width=2,clusters=3,code_bytes=1;
  const std::vector<float> centers{1,0,0, 1,0,0, -1,0,0},mean(width,0),sigma(width,1);
  const std::vector<float> basis{1,0, 0,1, 0,0};
  const std::vector<TieredQTier> tiers{{0,width,2,0,0}};
  std::vector<std::vector<float>> levels(9); levels[2]={-1.5f,-.5f,.5f,1.5f};
  const EncodingModel model{dimensions,centers,mean,sigma,levels,tiers};
  TieredQPanels fused; fused.depth=dimensions; fused.columns=width+clusters;
  fused.data.resize(dimensions*fused.padded_columns());
  for (size_t j=0;j<dimensions;++j) {
    for (size_t r=0;r<width;++r) fused.data[j*TieredQPanels::PANEL+r]=basis[j*width+r];
    for (size_t c=0;c<clusters;++c) fused.data[j*TieredQPanels::PANEL+width+c]=centers[c*dimensions+j];
  }
  const std::vector<double> projections{1,0, 1,0, -1,0},norms(clusters,.5);
  const EncoderState encoder{fused,projections,norms};
  const Layout layout{code_bytes};
  const std::vector<float> inputs{0,0,0, 1,0,0, -1,0,0, 1e-6f,.5f,0, -1e-6f,-.5f,0,
                                .5f,std::nextafter(.5f,INFINITY),0, -.5f,std::nextafter(-.5f,-INFINITY),0};
  const size_t count=inputs.size()/dimensions,stride=code_bytes+layout.metadata_bytes();
  for (auto kernel:{TieredQKernel::scalar,TieredQKernel::avx2,TieredQKernel::avx512}) {
    if (!TieredQ::kernel_available(kernel)) continue;
    std::vector<uint8_t> docs(count*stride);
    encode_block(model,encoder,layout,inputs.data(),count,docs.data(),nullptr,kernel);
    for (size_t i=0;i<count;++i) {
      const auto* doc=docs.data()+i*stride;
      const size_t cid=doc[layout.centroid_offset()];
      require(cid==(inputs[i*dimensions]<0 ? 2u : 0u),"centroid tie must retain first id");
      double exact=0,decoded=0;
      for (size_t j=0;j<dimensions;++j) {
        const double center=centers[cid*dimensions+j];
        exact+=(double(inputs[i*dimensions+j])-center)*center;
        double reconstructed=0;
        for (size_t r=0;r<width;++r) {
          const unsigned code=(doc[0]>>(r*2))&3;
          reconstructed+=basis[j*width+r]*(mean[r]+layout.factor(doc)*double(levels[2][code])*sigma[r]);
        }
        decoded+=center*reconstructed;
      }
      require(layout.half(doc,layout.correction_offset())==half_encode(exact-decoded),"projected correction reference");
    }
  }
}

void overflowing_centroid_dot() {
  const std::vector<float> centers{1e20f},mean{0},sigma{1};
  const std::vector<TieredQTier> tiers{{0,1,1,0,0}};
  std::vector<std::vector<float>> levels(9); levels[1]={-1,1};
  const EncodingModel model{1,centers,mean,sigma,levels,tiers};
  TieredQPanels fused; fused.depth=1; fused.columns=2; fused.data.resize(fused.padded_columns());
  fused.data[0]=1; fused.data[1]=centers[0];
  const std::vector<double> projections{double(centers[0])},norms{double(centers[0])*centers[0]/2};
  const EncoderState encoder{fused,projections,norms};
  const Layout layout{1};
  std::vector<uint8_t> doc(1+layout.metadata_bytes());
  encode_block(model,encoder,layout,centers.data(),1,doc.data(),nullptr);
  require(layout.factor(doc.data())==0 && layout.correction(doc.data())==0,"finite zero residual after FP32 overflow");
}
}

void tieredq_encoder_tests() {
  matrix_reference(); centroid_ties_and_correction(); overflowing_centroid_dot();
}
