// Independent scalar checks of byte-table packing, signed accumulation, projection and tail stores.
#include "tieredq/tieredq.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include "tieredq_scan_int8_avx512.h"
#include "tieredq_scan_int8_avx2.h"
namespace {
using namespace dv;
using namespace dv::tieredq_detail;
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
void wide_kernel_range_test(bool avx512) {
  constexpr size_t dimensions=5,count=4097,code_bytes=4,stride=9;
  constexpr int level_max=63;
  const std::vector<TieredQTier> tiers{{0,dimensions,6,0,0}};
  std::vector<std::vector<float>> levels(9);
  levels[6].resize(64);
  for (size_t code=0;code<levels[6].size();++code) levels[6][code]=(float(code)-31.5f)/31.5f;
  std::vector<uint8_t> rows(count*stride,0);
  const Layout layout{code_bytes};
  for (size_t doc=0;doc<count;++doc) {
    for (size_t rank=0;rank<dimensions;++rank)
      write_code(rows.data()+doc*stride,rank*6,6,(doc+rank)%levels[6].size());
    layout.store_half(rows.data()+doc*stride,layout.factor_offset(),1);
  }
  auto cache=build_fast_scan_cache(levels,tiers,dimensions,count,stride,code_bytes,1,level_max,rows.data());
  TieredQQuery q;
  q.weights={1,-1,0,.5f,-.25f}; q.centroid_scores={0};q.correction_weights={0};
  std::vector<FastScanQueryX86> prepared;
  std::vector<TieredQQuery> queries;
  constexpr size_t query_rows=8,guard=7;
  constexpr float sentinel=-123456;
  for (size_t row=0;row<query_rows;++row) {
    auto next=q;
    if (row%2) std::fill(next.weights.begin(),next.weights.end(),row%4==1 ? 1.f : -1.f);
    prepared.push_back(fast_scan_query_x86(*cache,next));
    queries.push_back(std::move(next));
  }
  for (size_t first=0;first<count;first+=lookup_block_docs) {
    const size_t remaining=std::min(lookup_block_docs,count-first);
    for (size_t n:{std::min(remaining,guard),remaining}) for (size_t rows:{1u,3u,8u}) {
      const size_t output_stride=n+guard;
      std::vector<float> scores(rows*output_stride,sentinel);
      if (avx512) scan_fast_avx512_tile(*cache,prepared.data(),rows,first,n,scores.data(),output_stride);
      else scan_fast_avx2_tile(*cache,prepared.data(),rows,first,n,scores.data(),output_stride);
      for (size_t row=0;row<rows;++row) {
        for (size_t doc=0;doc<n;++doc) {
          int dot=0;
          for (size_t rank=0;rank<dimensions;++rank) {
            const size_t code=(first+doc+rank)%levels[6].size();
            const int level=std::lround(levels[6][code]/cache->level_scale[0]);
            const int weight=std::lround(queries[row].weights[rank]*cache->level_scale[0]/prepared[row].scale);
            dot+=level*weight;
          }
          require(scores[row*output_stride+doc]==float(dot)*prepared[row].scale,
              "wide range differs from scalar numerical-level dot");
        }
        require(std::all_of(scores.begin()+row*output_stride+n,scores.begin()+(row+1)*output_stride,
            [](float x){return x==sentinel;}),"wide range overwrites tail");
      }
    }
  }
}
void lookup_kernel_test(size_t count,bool avx512) {
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
  auto query=prepare_fast_lookup_query(*cache->lookup,input.data(),{basis,sigma});
  const auto output=project_fast_query(map,input.data(),{basis,sigma});
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
  auto scan=[&](const FastScanLookupQuery& q,size_t first,size_t n,float* out) {
    if (avx512) scan_fast_lookup_avx512_range(*cache,q,first,n,out);
    else scan_fast_lookup_avx2_range(*cache,q,first,n,out);
  };
  constexpr float sentinel=12345;
  std::vector<float> scores(count+16,sentinel);
  scan(query,0,count,scores.data());
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
  for (size_t first=0;first<count;first+=lookup_block_docs) {
    const size_t rows=std::min(lookup_block_docs,count-first);
    std::vector<float> tile(rows+7,sentinel);
    scan(query,first,rows,tile.data());
    require(std::equal(tile.begin(),tile.begin()+rows,scores.begin()+first),"lookup range changes scores");
    require(std::all_of(tile.begin()+rows,tile.end(),[](float x){return x==sentinel;}),"lookup range overwrites tail");
    constexpr size_t partial_rows=7;
    if (rows>partial_rows) {
      std::fill(tile.begin(),tile.end(),sentinel);
      scan(query,first,partial_rows,tile.data());
      require(std::equal(tile.begin(),tile.begin()+partial_rows,scores.begin()+first),"partial lookup range changes scores");
      require(std::all_of(tile.begin()+partial_rows,tile.end(),[](float x){return x==sentinel;}),"partial lookup range overwrites tail");
    }
  }
  std::fill(query.centroid_scores.begin(),query.centroid_scores.end(),0);
  std::fill(query.correction_weights.begin(),query.correction_weights.end(),0);
  std::fill(query.scales.begin(),query.scales.end(),1);
  for (int bound:{-lookup_quant_max,lookup_quant_max}) {
    std::fill_n(query.tables.begin(),cache->lookup->groups.size()*lookup_table_entries,int8_t(bound));
    scan(query,0,count,scores.data());
    const float expected=float(bound)*cache->lookup->groups.size();
    require(std::all_of(scores.begin(),scores.begin()+count,[&](float score){return score==expected;}),"byte-table signed sum overflow");
  }

  const std::vector<float> zero(dimensions);
  const auto zero_query=prepare_fast_lookup_query(*cache->lookup,zero.data(),{basis,sigma});
  scan(zero_query,0,count,scores.data());
  require(std::all_of(scores.begin(),scores.begin()+count,[](float x){return x==0;}),"zero byte-table query differs");
}
} // namespace
#endif
void tieredq_lookup_tests() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  for (bool avx512:{false,true}) {
    const bool available=avx512 ? (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw"))
        : bool(__builtin_cpu_supports("avx2"));
    if (!available) continue;
    wide_kernel_range_test(avx512);
    for (size_t count:{1u,7u,8u,9u,15u,16u,17u,31u,32u,33u,63u,64u,65u,255u,256u,257u,511u,512u,4097u})
      lookup_kernel_test(count,avx512);
  }
#endif
}
