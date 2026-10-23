// Independent scalar checks of byte-table packing, signed accumulation, projection and tail stores.
#include "tieredq/tieredq.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include "tieredq_scan_int8_avx512.h"
namespace {
using namespace dv;
using namespace dv::tieredq_detail;
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
void lookup_kernel_test(size_t count) {
  constexpr size_t dimensions=4, weights=40, centroids=2, code_bytes=19, stride=24;
  const std::vector<TieredQTier> tiers{{0,35,4,0,0},{35,5,1,0,140}};
  std::vector<std::vector<float>> levels(9);
  for (unsigned bits:{1u,4u}) {
    const size_t n=size_t(1)<<bits; levels[bits].resize(n);
    for (size_t i=0;i<n;++i) { const float x=2*float(i)/float(n-1)-1; levels[bits][i]=x*x*x; }
  }
  std::vector<float> basis(dimensions*weights),sigma(weights),mean(weights),centers(dimensions*centroids),norms(centroids,1);
  for (size_t i=0;i<basis.size();++i) basis[i]=std::sin(float(i));
  for (size_t i=0;i<weights;++i) { sigma[i]=float(i+1)/weights; mean[i]=std::cos(float(i)); }
  for (size_t i=0;i<centers.size();++i) centers[i]=float(i)/centers.size();
  const auto map=build_fast_query_projection(dimensions,basis,sigma,mean,centers,norms);
  std::vector<uint8_t> rows(count*stride);
  const Layout layout{code_bytes};
  for (size_t d=0;d<count;++d) {
    auto* row=rows.data()+d*stride;
    for (const auto& tier:tiers) for (uint32_t j=0;j<tier.dimensions;++j)
      write_code(row,tier.bit_offset+j*tier.bits,tier.bits,unsigned((d+j) % (1u<<tier.bits)));
    row[code_bytes]=uint8_t(d%centroids);
    layout.store_half(row,layout.factor_offset(),1);
    layout.store_half(row,layout.correction_offset(),.25);
  }
  const auto cache=build_fast_lookup_cache(levels,tiers,map,count,stride,code_bytes,centroids,rows.data());
  const std::vector<float> input{.3f,-.7f,.1f,.2f};
  auto query=prepare_fast_lookup_query(*cache->lookup,input.data());
  const auto output=project_fast_query(map,input.data());
  for (size_t r=0;r<weights;++r) {
    float expected=0; for (size_t i=0;i<dimensions;++i) expected+=input[i]*basis[i*weights+r];
    require(std::fabs(output[r]-expected*sigma[r])<1e-5f,"fast query projection differs");
  }
  float mean_correction=0;
  for (size_t r=0;r<weights;++r) {
    float transformed=0;
    for (size_t i=0;i<dimensions;++i) transformed+=input[i]*basis[i*weights+r];
    mean_correction+=transformed*mean[r];
  }
  for (size_t cid=0;cid<centroids;++cid) {
    float dot=0; for (size_t i=0;i<dimensions;++i) dot+=input[i]*centers[cid*dimensions+i];
    require(std::fabs(output[weights+cid]-dot-mean_correction)<1e-5f,"projected centroid score differs");
    require(std::fabs(output[weights+centroids+cid]-dot/(2*norms[cid]))<1e-5f,"projected centroid correction differs");
  }
  constexpr float sentinel=12345;
  std::vector<float> scores(count+16,sentinel);
  scan_fast_lookup_avx512(*cache,query,scores.data());
  for (size_t d=0;d<count;++d) {
    float dot=0;
    const auto& groups=cache->lookup->groups;
    for (size_t start=0;start<groups.size();start+=lookup_scale_groups) {
      int sum=0;
      for (size_t g=start;g<std::min(start+lookup_scale_groups,groups.size());++g) {
        const auto& group=groups[g];
        const unsigned code=read_code(rows.data()+d*stride,group.bit,group.bits*group.count);
        sum+=query.tables[g*lookup_table_entries+code];
      }
      dot+=float(sum)*query.scales[start/lookup_scale_groups];
    }
    const size_t cid=d%centroids;
    const float expected=query.centroid_scores[cid]+query.correction_weights[cid]*.25f+dot;
    require(std::fabs(scores[d]-expected)<1e-5f,"byte-table kernel differs from scalar sum");
  }
  require(std::all_of(scores.begin()+count,scores.end(),[](float x){return x==sentinel;}),"byte-table tail store overflow");
  std::fill(query.centroid_scores.begin(),query.centroid_scores.end(),0);
  std::fill(query.correction_weights.begin(),query.correction_weights.end(),0);
  std::fill(query.scales.begin(),query.scales.end(),1);
  for (int bound:{-lookup_quant_max,lookup_quant_max}) {
    std::fill_n(query.tables.begin(),cache->lookup->groups.size()*lookup_table_entries,int8_t(bound));
    scan_fast_lookup_avx512(*cache,query,scores.data());
    const float expected=float(bound)*cache->lookup->groups.size();
    require(std::all_of(scores.begin(),scores.begin()+count,[&](float score){return score==expected;}),"byte-table signed sum overflow");
  }

  const std::vector<float> zero(dimensions);
  const auto zero_query=prepare_fast_lookup_query(*cache->lookup,zero.data());
  scan_fast_lookup_avx512(*cache,zero_query,scores.data());
  require(std::all_of(scores.begin(),scores.begin()+count,[](float x){return x==0;}),"zero byte-table query differs");
}
} // namespace
#endif
void tieredq_lookup_tests() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if (!fast_scan_available()) return;
  for (size_t count:{1u,15u,16u,17u,63u,64u,65u,255u,256u,257u}) lookup_kernel_test(count);
#endif
}
