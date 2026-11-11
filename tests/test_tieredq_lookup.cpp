// Independent scalar checks of byte-table packing, signed accumulation, projection and tail stores.
#include "tieredq/tieredq.h"
#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <vector>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define TQ_TEST_X86_LOOKUP 1
#include "tieredq_scan_int8_avx512.h"
#include "tieredq_lookup_range_avx512.h"
#include "tieredq_lookup_range_avx2.h"
#include "tieredq_scan_int8_avx2.h"
#elif defined(__aarch64__)
#define TQ_TEST_NEON_LOOKUP 1
#include "tieredq_scan_int8_neon.h"
#include "tieredq_lookup_range_neon.h"
#endif
#if defined(TQ_TEST_X86_LOOKUP) || defined(TQ_TEST_NEON_LOOKUP)
#include "tieredq_scan_fast_scalar.h"
#include "tieredq_fast_scan.h"
#include "tieredq_train_codebook_data.h"
namespace {
using namespace tq;
using namespace tq::tieredq_detail;
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
// A byte-table range kernel: scores [first, first + count) of one query.
using LookupRangeKernel=void(*)(const FastScanCache&,const FastScanLookupQuery&,size_t,size_t,float*);
// Its pruned counterpart: rejected documents score -inf.
using LookupPrunedKernel=void(*)(const FastScanCache&,const FastScanLookupQuery&,size_t,size_t,float*,const LookupRangeCheckpoint&);
#ifdef TQ_TEST_X86_LOOKUP
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
// Tier layouts of 40 coded weights; the second scores additive 3-bit codes as split 2- and 1-bit groups.
#endif
struct LookupLayout { std::vector<TieredQTier> tiers; size_t code_bytes; };
const LookupLayout narrow_layout{{{0,35,4,0,0},{35,5,1,0,140}},19};
const LookupLayout split_layout{{{0,7,4,0,0},{7,13,3,0,28},{20,11,2,0,67},{31,9,1,0,89}},13};
void lookup_kernel_test(size_t count,LookupRangeKernel kernel,const LookupLayout& layout_spec) {
  constexpr size_t dimensions=4, weights=40, centroids=2;
  const auto& tiers=layout_spec.tiers;
  const size_t code_bytes=layout_spec.code_bytes, stride=code_bytes+TieredQ::metadata_bytes;
  std::vector<std::vector<float>> levels(9);
  for (unsigned bits:{1u,2u,4u}) {
    const size_t n=size_t(1)<<bits; levels[bits].resize(n);
    for (size_t i=0;i<n;++i) { const float x=2*float(i)/float(n-1)-1; levels[bits][i]=x*x*x; }
  }
  levels[3]=additive_three_codebook();
  std::vector<float> basis(dimensions*weights),sigma(weights),mean(weights),centers(dimensions*centroids),norms(centroids,1);
  for (size_t i=0;i<basis.size();++i) basis[i]=std::sin(float(i));
  for (size_t i=0;i<weights;++i) { sigma[i]=float(i+1)/weights; mean[i]=std::cos(float(i)); }
  for (size_t i=0;i<centers.size();++i) centers[i]=float(i)/centers.size();
  const std::vector<uint16_t> fp32_basis_only;
  const FastQueryBasis source{basis,fp32_basis_only,sigma,centers,norms,dimensions};
  const auto map=build_fast_query_projection(dimensions,source,mean);
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
  LookupWidthCounts widths{};
  for (const auto& tier:tiers) widths[tier.bits]+=tier.dimensions;
  const size_t group_count=lookup_group_count(widths,true);
  require(cache->lookup->groups.size()==group_count && cache->lookup->splits[split_code_bits],"split lookup groups missing");
  const std::vector<float> input{.3f,-.7f,.1f,.2f};
  auto query=prepare_fast_lookup_query(*cache->lookup,input.data(),source);
  const auto output=project_fast_query(map,input.data(),source);
  {
    // Dispatched (AVX-512) tables equal the portable builder, also on exact .5 ties after scaling.
    FastScanLookupQuery portable;
    portable.tables.assign(query.tables.size(),0);
    fast_lookup_tables_scalar(*cache->lookup,output.data(),portable);
    require(portable.tables==query.tables && portable.scales==query.scales,"lookup tables differ from portable build");
    for (float r:{2.5f,-2.5f,0.5f,-0.5f,1.49999988f,126.5f,-126.5f,0.0f,-0.0f})
      require(round_half_away(r)==float(std::lround(r)),"half-away rounding differs from lround");
    auto ties=output;
    for (size_t r=0;r<weights;++r) ties[r]=float(r%5)*0.25f-0.5f;
    auto dispatched=fast_lookup_query(*cache->lookup,ties);
    portable.tables.assign(query.tables.size(),0); portable.scales.clear();
    fast_lookup_tables_scalar(*cache->lookup,ties.data(),portable);
    require(portable.tables==dispatched.tables && portable.scales==dispatched.scales,"lookup ties differ from portable build");
  }
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
  auto scan=[&](const FastScanLookupQuery& q,size_t first,size_t n,float* out) { kernel(*cache,q,first,n,out); };
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
        sum+=query.tables[g*lookup_table_entries+lookup_group_code(*cache->lookup,group,rows.data()+d*stride)];
      }
      dot+=float(sum)*query.scales[start/lookup_scale_groups];
    }
    const size_t cid=d%centroids;
    const float expected=query.centroid_scores[cid]+query.correction_weights[cid]*.25f+dot;
    require(std::fabs(scores[d]-expected)<1e-5f,"byte-table kernel differs from scalar sum");
    // Independent float score from whole codes: table rounding bounds the error per group.
    float exact=query.centroid_scores[cid]+query.correction_weights[cid]*.25f,bound=1e-4f;
    for (const auto& tier:tiers) for (uint32_t j=0;j<tier.dimensions;++j)
      exact+=output[tier.start+j]*levels[tier.bits][read_code(rows.data()+d*stride,tier.bit_offset+j*tier.bits,tier.bits)];
    for (size_t g=0;g<group_count;++g) bound+=query.scales[g/lookup_scale_groups]/2;
    require(std::fabs(scores[d]-exact)<=bound,"byte-table score differs from whole-code levels");
  }
  FastScanScalarQuery scalar_query; scalar_query.lookup=query;
  std::vector<float> scalar(count);
  scan_fast_scalar_tile(*cache,&scalar_query,1,0,count,scalar.data(),count);
  for (size_t d=0;d<count;++d) require(std::fabs(scalar[d]-scores[d])<1e-5f,"scalar fallback differs from byte-table kernel");
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
  const auto zero_query=prepare_fast_lookup_query(*cache->lookup,zero.data(),source);
  scan(zero_query,0,count,scores.data());
  require(std::all_of(scores.begin(),scores.begin()+count,[](float x){return x==0;}),"zero byte-table query differs");
}
// Range checkpoint: survivors keep the unpruned kernel's score bits; with the exact bound only scores below the
// threshold drop. Range starts at first_offset exercise chunks that begin inside a block. Prefixes are automatic
// (a share of sections) and explicit (fast_range_prefix_tiers tiers).
void lookup_range_checkpoint_test(const FastScanCache& cache,const std::vector<TieredQTier>& tiers,
    const std::vector<FastScanLookupQuery>& queries,LookupRangeKernel full,LookupPrunedKernel pruned_kernel) {
  const size_t sections=cache.lookup->section_ends.size();
  for (double confidence:{0.0,8.0}) {
    const double share=confidence>0 ? lookup_confident_suffix_share : lookup_exact_suffix_share;
    const size_t suffix=std::max<size_t>(1,size_t(std::lround(share*double(sections))));
    const size_t tiered=lookup_tier_prefix_sections(*cache.lookup,tiers,fast_range_prefix_tiers);
    const size_t expected=tiered ? std::min(tiered,sections-suffix) : sections-suffix;
    require(lookup_prefix_sections(*cache.lookup,tiers,TieredQPruning::automatic_prefix,confidence)==expected,
        "automatic prefix leaves the wrong suffix share");
  }
  constexpr size_t first_offset=3*fast_list_alignment,offset_count=700;
  const size_t count=cache.count;
  for (size_t prefix_tiers:{size_t(TieredQPruning::automatic_prefix),fast_range_prefix_tiers}) {
    require(lookup_range_checkpoint_supported(cache,tiers,prefix_tiers),"wide test model has no lookup suffix");
    size_t rejected=0;
    for (const auto& query:queries) {
      std::vector<float> reference(count);
      full(cache,query,0,count,reference.data());
      std::vector<float> sorted(reference);
      std::sort(sorted.begin(),sorted.end());
      for (double confidence:{0.0,8.0}) for (float threshold:{-std::numeric_limits<float>::infinity(),sorted[count/2],
          sorted[count*9/10],sorted[count-1],std::numeric_limits<float>::infinity()}) for (size_t first:{size_t(0),first_offset}) {
        const size_t n=first ? offset_count : count;
        const LookupRangeCheckpoint checkpoint{prepare_lookup_range_bounds(cache,tiers,query,prefix_tiers,confidence),threshold};
        std::vector<float> pruned(n);
        pruned_kernel(cache,query,first,n,pruned.data(),checkpoint);
        for (size_t i=0;i<n;++i) {
          const float unpruned=reference[first+i];
          const bool dropped=pruned[i]==-std::numeric_limits<float>::infinity();
          require(dropped || std::bit_cast<uint32_t>(pruned[i])==std::bit_cast<uint32_t>(unpruned),"pruned lookup changed a surviving score");
          require(!dropped || confidence>0 || unpruned<threshold,"exact lookup bound rejected a competitive document");
          rejected+=dropped;
        }
      }
    }
    require(rejected>0,"lookup range checkpoint rejected nothing");
  }
}
// Wide additive models: range-sorted groups and fine sections score alike on every backend and query grouping (#13).
void wide_additive_backend_test(bool fine) {
  constexpr size_t dimensions=48,count=1031,centroids=3,query_count=19;
  constexpr float tolerance=2e-5f;
  std::vector<std::vector<float>> levels(gaussian_codebooks().begin(),gaussian_codebooks().end());
  levels[split_code_bits]=additive_three_codebook();
  for (uint32_t b=lookup_code_bits+1;b<=widest_code_bits;++b) levels[b]=additive_wide_codebook(b);
  std::vector<TieredQTier> tiers;
  uint32_t start=0,offset=0;
  for (uint32_t bits:{8u,7u,6u,5u,4u,3u,2u,1u}) {
    if (!fine && bits>=fine_section_min_bits) continue;
    constexpr uint32_t tier_dimensions=5;
    tiers.push_back({start,tier_dimensions,bits,0,offset}); start+=tier_dimensions; offset+=tier_dimensions*bits;
  }
  const size_t weights=start,code_bytes=(offset+CHAR_BIT-1)/CHAR_BIT,stride=code_bytes+TieredQ::metadata_bytes;
  std::vector<float> basis(dimensions*weights),sigma(weights),mean(weights),centers(dimensions*centroids),norms(centroids,1);
  for (size_t i=0;i<basis.size();++i) basis[i]=std::sin(float(i));
  for (size_t i=0;i<weights;++i) { sigma[i]=1/float(i+1); mean[i]=std::cos(float(i)); }
  for (size_t i=0;i<centers.size();++i) centers[i]=float(i)/centers.size();
  const std::vector<uint16_t> fp32_basis_only;
  const FastQueryBasis source{basis,fp32_basis_only,sigma,centers,norms,dimensions};
  const auto map=build_fast_query_projection(dimensions,source,mean);
  std::vector<uint8_t> rows(count*stride);
  const Layout layout{code_bytes};
  for (size_t d=0;d<count;++d) {
    auto* row=rows.data()+d*stride;
    for (const auto& tier:tiers) for (uint32_t j=0;j<tier.dimensions;++j)
      write_code(row,tier.bit_offset+j*tier.bits,tier.bits,unsigned((d*7+j*3)%(1u<<tier.bits)));
    row[code_bytes]=uint8_t(d%centroids);
    layout.store_half(row,layout.factor_offset(),1);
    layout.store_half(row,layout.correction_offset(),.25);
  }
  const auto cache=build_fast_lookup_cache(levels,tiers,map,count,stride,code_bytes,centroids,rows.data(),sigma);
  const auto& lookup=*cache->lookup;
  require(lookup.section_ends.back()==lookup.groups.size(),"sections do not cover the groups");
  require(fine==(lookup.section_ends.front()==lookup_fine_scale_groups),"wrong section size");
  std::vector<FastScanLookupQuery> queries(query_count);
  std::vector<FastScanScalarQuery> scalar(query_count);
  std::vector<std::vector<float>> outputs(query_count);
  for (size_t q=0;q<query_count;++q) {
    std::vector<float> input(dimensions);
    for (size_t i=0;i<dimensions;++i) input[i]=std::cos(float(q*dimensions+i));
    queries[q]=prepare_fast_lookup_query(lookup,input.data(),source);
    scalar[q].lookup=queries[q];
    outputs[q]=project_fast_query(map,input.data(),source);
    FastScanLookupQuery portable; portable.tables.assign(queries[q].tables.size(),0);
    fast_lookup_tables_scalar(lookup,outputs[q].data(),portable);
    require(portable.tables==queries[q].tables && portable.scales==queries[q].scales,"wide tables differ from portable build");
  }
  std::vector<float> reference(query_count*count);
  for (size_t q=0;q<query_count;++q) scan_fast_scalar_tile(*cache,&scalar[q],1,0,count,reference.data()+q*count,count);
  auto same=[&](const std::vector<float>& scores,const char* message) {
    for (size_t i=0;i<reference.size();++i) require(std::fabs(scores[i]-reference[i])<=tolerance*(1+std::fabs(reference[i])),message);
  };
#ifdef TQ_TEST_X86_LOOKUP
  std::vector<FastScanQueryX86> x86(query_count);
  for (size_t q=0;q<query_count;++q) x86[q].lookup=queries[q];
  // Grouped calls span full and partial query groups, as the batch drivers issue them.
  auto grouped=[&](auto scan,size_t group) {
    std::vector<float> scores(query_count*count);
    for (size_t q=0;q<query_count;q+=group) scan(*cache,x86.data()+q,std::min(group,query_count-q),0,count,scores.data()+q*count,count);
    return scores;
  };
  if (__builtin_cpu_supports("avx2")) same(grouped(scan_fast_avx2_tile,fast_scan_query_tile_avx2),"wide AVX2 scores differ from scalar");
  if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw"))
    same(grouped(scan_fast_avx512_tile,fast_scan_query_tile_x86),"wide AVX-512 scores differ from scalar");
#endif
  // Independent score from whole levels: table rounding bounds the error by half a scale per group.
  for (size_t q=0;q<query_count;++q) for (size_t d=0;d<count;++d) {
    const size_t cid=d%centroids;
    double exact=queries[q].centroid_scores[cid]+queries[q].correction_weights[cid]*.25,bound=1e-3;
    for (const auto& tier:tiers) for (uint32_t j=0;j<tier.dimensions;++j)
      exact+=outputs[q][tier.start+j]*levels[tier.bits][read_code(rows.data()+d*stride,tier.bit_offset+j*tier.bits,tier.bits)];
    for (size_t section=0,first=0;section<lookup.section_ends.size();first=lookup.section_ends[section++])
      bound+=(lookup.section_ends[section]-first)*queries[q].scales[section]/2;
    require(std::fabs(reference[q*count+d]-exact)<=bound,"wide nibble score differs from whole-code levels");
  }
  if (!fine) return;
#ifdef TQ_TEST_NEON_LOOKUP
  lookup_range_checkpoint_test(*cache,tiers,queries,scan_fast_lookup_neon_range,scan_range_lookup_neon);
#else
  if (__builtin_cpu_supports("avx2"))
    lookup_range_checkpoint_test(*cache,tiers,queries,scan_fast_lookup_avx2_range,scan_range_lookup_avx2);
  if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw"))
    lookup_range_checkpoint_test(*cache,tiers,queries,scan_fast_lookup_avx512_range,scan_range_lookup_avx512);
#endif
}
constexpr size_t lookup_test_counts[]={1,7,8,9,15,16,17,31,32,33,63,64,65,255,256,257,511,512,4097};
void lookup_kernel_tests(LookupRangeKernel kernel) {
  for (const auto* layout:{&narrow_layout,&split_layout})
    for (size_t count:lookup_test_counts) lookup_kernel_test(count,kernel,*layout);
}
} // namespace
#endif
void tieredq_lookup_tests() {
#if defined(TQ_TEST_X86_LOOKUP) || defined(TQ_TEST_NEON_LOOKUP)
  for (bool fine:{false,true}) wide_additive_backend_test(fine);
#endif
#ifdef TQ_TEST_NEON_LOOKUP
  lookup_kernel_tests(scan_fast_lookup_neon_range);
#endif
#ifdef TQ_TEST_X86_LOOKUP
  for (bool avx512:{false,true}) {
    const bool available=avx512 ? (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw"))
        : bool(__builtin_cpu_supports("avx2"));
    if (!available) continue;
    wide_kernel_range_test(avx512);
    lookup_kernel_tests(avx512 ? scan_fast_lookup_avx512_range : scan_fast_lookup_avx2_range);
  }
#endif
}
