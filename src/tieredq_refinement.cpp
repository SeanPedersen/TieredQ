// TieredQRefinement: nested suffix codes over a trained prefix model (tieredq_refinement.h).
#include "tieredq/tieredq_refinement.h"
#include "tieredq_basis_panels.h"
#include "tieredq_encode_float.h"
#include "tieredq_io.h"
#include "tieredq_layout.h"
#include "training_rng.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <vector>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define DV_TIEREDQ_REFINEMENT_X86 1
#endif

namespace dv {
namespace tieredq_detail {
struct RefinementEncoder {
  std::once_flag once;
  TieredQPanels panels;               // model dimensions x positions
  std::vector<float> centroid_dots;   // centroid x padded positions
};
} // namespace tieredq_detail
namespace {
using tieredq_detail::bad;
using tieredq_detail::product;
constexpr uint32_t refinement_magic=0x46525154,refinement_format=1;  // "TQRF"
constexpr uint32_t max_bits=8;
constexpr size_t lloyd_iterations=60,level_rows=20000,encode_block_rows=256;
// Rotated prefix blocks must span the finer model's ranks at the same positions (FP16 bases).
constexpr double span_tolerance=1e-2,centroid_tolerance=1e-4;

uint32_t read_bits(const uint8_t* doc,uint32_t offset,uint32_t bits) {
  if (!bits) return 0;
  uint32_t value=uint32_t(doc[offset/8])>>(offset%8);
  if (offset%8+bits>8) value|=uint32_t(doc[offset/8+1])<<(8-offset%8);
  return value&((1u<<bits)-1);
}
uint16_t read_half(const uint8_t* doc) { return uint16_t(doc[0])|uint16_t(doc[1])<<8; }
void write_half(uint8_t* doc,double value) {
  const uint16_t bits=tieredq_detail::half_encode(value); doc[0]=uint8_t(bits); doc[1]=uint8_t(bits>>8);
}
float half_round(double value) { return tieredq_detail::half_decode(tieredq_detail::half_encode(value)); }
// Index of the nearest of `count` sorted levels.
uint32_t nearest(const float* levels,size_t count,double value) {
  const size_t above=size_t(std::lower_bound(levels,levels+count,float(value))-levels);
  if (above==0) return 0;
  if (above==count) return uint32_t(count-1);
  return uint32_t(value-levels[above-1]<=levels[above]-value ? above-1 : above);
}
// 1-D Lloyd from quantile starts; an empty sample keeps `fallback` at every level.
std::vector<float> lloyd(std::vector<double>& samples,size_t count,float fallback) {
  if (samples.empty()) return std::vector<float>(count,fallback);
  std::sort(samples.begin(),samples.end());
  std::vector<double> sums(samples.size()+1,0.0);
  std::partial_sum(samples.begin(),samples.end(),sums.begin()+1);
  std::vector<double> levels(count);
  for (size_t i=0;i<count;++i) levels[i]=samples[std::min(samples.size()-1,size_t((double(i)+0.5)*double(samples.size())/double(count)))];
  for (size_t it=0;it<lloyd_iterations;++it) {
    size_t lo=0;
    for (size_t i=0;i<count;++i) {
      const size_t hi=i+1<count ? size_t(std::lower_bound(samples.begin(),samples.end(),(levels[i]+levels[i+1])/2)-samples.begin())
                                : samples.size();
      if (hi>lo) levels[i]=(sums[hi]-sums[lo])/double(hi-lo);
      lo=std::max(lo,hi);
    }
    std::sort(levels.begin(),levels.end());
  }
  return {levels.begin(),levels.end()};
}
// Rows are copied into buffers padded for 16-byte field windows past their last code byte.
constexpr size_t read_padding=16;
constexpr uint32_t segment_lanes=16;
// Level tables pad past their end so 16-float register loads of the last tables stay inside.
constexpr size_t level_padding=2*segment_lanes;
// A field of bits (<= 8) at bit offset `bit` of a padded row.
uint32_t field(const uint8_t* doc,uint32_t bit,uint32_t bits) {
  uint32_t word; std::memcpy(&word,doc+bit/8,sizeof(word));
  return (word>>(bit%8))&((1u<<bits)-1);
}
// Sum over the segments' positions of weights[r] * levels[table + (prefix code << shift | suffix code)].
template<class Segment>
float segment_sum_scalar(const uint8_t* prefix,const uint8_t* suffix,const Segment* first,const Segment* last,
                         const float* levels,const float* weights) {
  float sum=0;
  for (const Segment* s=first;s!=last;++s) for (uint32_t j=0;j<s->count;++j) {
    const uint32_t code=field(prefix,s->prefix_bit+j*s->prefix_bits,s->prefix_bits)<<s->shift|
                        field(suffix,s->suffix_bit+j*s->suffix_bits,s->suffix_bits);
    sum+=weights[s->position+j]*levels[s->table+code];
  }
  return sum;
}
#ifdef DV_TIEREDQ_REFINEMENT_X86
// 16 fields of `bits` from a 16-byte window starting `start` (< 8) bits in: lane j shuffles window bytes t / 8 and
// t / 8 + 1 (t = start + j * bits) into its dword and shifts by t % 8. Only 8-bit fields off a byte boundary need a
// 17th byte (fits = false).
struct FieldPattern { alignas(64) uint8_t control[64]; alignas(64) uint32_t shift[segment_lanes]; bool fits; };
constexpr uint32_t pattern_widths=max_bits+1;
const FieldPattern* field_patterns() {
  static const auto patterns=[] {
    std::vector<FieldPattern> out(8*pattern_widths);
    for (uint32_t start=0;start<8;++start) for (uint32_t bits=0;bits<pattern_widths;++bits) {
      FieldPattern& pattern=out[start*pattern_widths+bits];
      pattern.fits=true;
      for (uint32_t j=0;j<segment_lanes;++j) {
        const uint32_t t=start+j*bits,byte=t/8;
        const bool second=t%8+bits>8;
        if (second && byte+1>=16) pattern.fits=false;
        uint8_t* control=pattern.control+4*j;
        control[0]=uint8_t(byte); control[1]=second && byte+1<16 ? uint8_t(byte+1) : 0x80; control[2]=control[3]=0x80;
        pattern.shift[j]=t%8;
      }
    }
    return out;
  }();
  return patterns.data();
}
__attribute__((target("avx512f,avx512bw")))
inline __m512i fields_avx512(const FieldPattern* patterns,const uint8_t* doc,uint32_t bit,uint32_t bits) {
  if (!bits) return _mm512_setzero_si512();
  const __m512i mask=_mm512_set1_epi32(int((1u<<bits)-1));
  const FieldPattern& pattern=patterns[(bit%8)*pattern_widths+bits];
  if (!pattern.fits) {
    const __m512i offset=_mm512_add_epi32(_mm512_set1_epi32(int(bit)),
        _mm512_mullo_epi32(_mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0),_mm512_set1_epi32(int(bits))));
    const __m512i word=_mm512_i32gather_epi32(_mm512_srli_epi32(offset,3),doc,1);
    return _mm512_and_si512(_mm512_srlv_epi32(word,_mm512_and_si512(offset,_mm512_set1_epi32(7))),mask);
  }
  const __m512i window=_mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(doc+bit/8)));
  const __m512i word=_mm512_shuffle_epi8(window,_mm512_load_si512(pattern.control));
  return _mm512_and_si512(_mm512_srlv_epi32(word,_mm512_load_si512(pattern.shift)),mask);
}
// Tables of up to 32 levels are looked up in registers, larger ones gathered.
template<class Segment>
__attribute__((target("avx512f,avx512bw")))
float segment_sum_avx512(const uint8_t* prefix,const uint8_t* suffix,const Segment* first,const Segment* last,
                         const float* levels,const float* weights) {
  const FieldPattern* patterns=field_patterns();
  __m512 sum=_mm512_setzero_ps();
  for (const Segment* s=first;s!=last;++s) {
    const __mmask16 live=__mmask16((1u<<s->count)-1);
    const __m512i code=_mm512_or_si512(
        _mm512_sllv_epi32(fields_avx512(patterns,prefix,s->prefix_bit,s->prefix_bits),_mm512_set1_epi32(int(s->shift))),
        fields_avx512(patterns,suffix,s->suffix_bit,s->suffix_bits));
    const float* table=levels+s->table;
    const __m512 level=s->table_size<=segment_lanes ? _mm512_permutexvar_ps(code,_mm512_loadu_ps(table)) :
        s->table_size<=2*segment_lanes ? _mm512_permutex2var_ps(_mm512_loadu_ps(table),code,_mm512_loadu_ps(table+segment_lanes)) :
        _mm512_mask_i32gather_ps(_mm512_setzero_ps(),live,code,table,4);
    sum=_mm512_fmadd_ps(_mm512_maskz_loadu_ps(live,weights+s->position),level,sum);
  }
  return _mm512_reduce_add_ps(sum);
}
#endif
// Query projection onto int8 rank panels (int8_panel_position, zero padded): out[r] = scale[r] * sum_i x[i] * c(i, r).
// It streams from memory, so int8 halves the binary16 panels' traffic.
constexpr double panel_levels=127;
// Blocks of up to int8_block_panels panels interleave them per input: [block][input][panel in block][rank % 16], so
// one projection pass reads the panels as a single sequential stream.
constexpr size_t int8_block_panels=16;
size_t int8_panel_position(size_t inputs,size_t ranks,size_t input,size_t rank) {
  using tieredq_detail::basis_panel_ranks;
  const size_t panel=rank/basis_panel_ranks,first=panel/int8_block_panels*int8_block_panels;
  const size_t width=std::min(int8_block_panels,tieredq_detail::basis_panel_count(ranks)-first);
  return (first*inputs+input*width+panel-first)*basis_panel_ranks+rank%basis_panel_ranks;
}
void project_int8_panels_scalar(const int8_t* panels,const float* scale,size_t inputs,size_t ranks,const float* x,float* out) {
  using tieredq_detail::basis_panel_ranks;
  for (size_t panel=0;panel<tieredq_detail::basis_panel_count(ranks);++panel) {
    float sums[basis_panel_ranks]={};
    for (size_t i=0;i<inputs;++i) for (size_t l=0;l<basis_panel_ranks;++l)
      sums[l]+=x[i]*float(panels[int8_panel_position(inputs,ranks,i,panel*basis_panel_ranks+l)]);
    for (size_t l=0;l<basis_panel_ranks && panel*basis_panel_ranks+l<ranks;++l)
      out[panel*basis_panel_ranks+l]=scale[panel*basis_panel_ranks+l]*sums[l];
  }
}
#ifdef DV_TIEREDQ_REFINEMENT_X86
// Blocks of Panels panels keep their accumulators in registers over all inputs.
template<size_t Panels>
__attribute__((target("avx512f")))
void project_int8_block_avx512(const int8_t* panels,const float* scale,size_t inputs,size_t ranks,const float* x,float* out,
                               size_t panel) {
  using tieredq_detail::basis_panel_ranks;
  __m512 acc[Panels];
  for (auto& value:acc) value=_mm512_setzero_ps();
  const int8_t* base=panels+panel*inputs*basis_panel_ranks;
  for (size_t i=0;i<inputs;++i) {
    const __m512 value=_mm512_set1_ps(x[i]);
    for (size_t b=0;b<Panels;++b) {
      const __m128i codes=_mm_loadu_si128(reinterpret_cast<const __m128i*>(base+(i*Panels+b)*basis_panel_ranks));
      acc[b]=_mm512_fmadd_ps(value,_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(codes)),acc[b]);
    }
  }
  for (size_t b=0;b<Panels;++b) {
    const size_t rank=(panel+b)*basis_panel_ranks;
    const __mmask16 live=__mmask16((1u<<std::min(basis_panel_ranks,ranks-rank))-1);
    _mm512_mask_storeu_ps(out+rank,live,_mm512_mul_ps(_mm512_maskz_loadu_ps(live,scale+rank),acc[b]));
  }
}
template<size_t Panels>
__attribute__((target("avx512f")))
void project_int8_tail_avx512(const int8_t* panels,const float* scale,size_t inputs,size_t ranks,const float* x,float* out,
                              size_t panel,size_t live) {
  if constexpr (Panels>1) if (live<Panels) return project_int8_tail_avx512<Panels-1>(panels,scale,inputs,ranks,x,out,panel,live);
  project_int8_block_avx512<Panels>(panels,scale,inputs,ranks,x,out,panel);
}
__attribute__((target("avx512f")))
void project_int8_panels_avx512(const int8_t* panels,const float* scale,size_t inputs,size_t ranks,const float* x,float* out) {
  const size_t count=tieredq_detail::basis_panel_count(ranks);
  size_t panel=0;
  for (;panel+int8_block_panels<=count;panel+=int8_block_panels)
    project_int8_block_avx512<int8_block_panels>(panels,scale,inputs,ranks,x,out,panel);
  if (panel<count) project_int8_tail_avx512<int8_block_panels>(panels,scale,inputs,ranks,x,out,panel,count-panel);
}
#endif
void project_int8_panels(const int8_t* panels,const float* scale,size_t inputs,size_t ranks,const float* x,float* out) {
#ifdef DV_TIEREDQ_REFINEMENT_X86
  static const bool avx512=__builtin_cpu_supports("avx512f");
  if (avx512) return project_int8_panels_avx512(panels,scale,inputs,ranks,x,out);
#endif
  project_int8_panels_scalar(panels,scale,inputs,ranks,x,out);
}
bool avx512_kernel_available() {
#ifdef DV_TIEREDQ_REFINEMENT_X86
  static const bool available=__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
  return available;
#else
  return false;
#endif
}
TieredQKernel resolve_kernel(TieredQKernel kernel) {
  if (kernel==TieredQKernel::automatic)
    return avx512_kernel_available() && !std::getenv("DV_TIEREDQ_NO_AVX512") ? TieredQKernel::avx512 : TieredQKernel::scalar;
  if (kernel==TieredQKernel::scalar || (kernel==TieredQKernel::avx512 && avx512_kernel_available())) return kernel;
  bad("refinement kernel unavailable");
  return TieredQKernel::scalar;
}
} // namespace

size_t TieredQRefinement::positions() const { return prefix_.coded_dimensions()+extra_mean_.size(); }

TieredQRefinement TieredQRefinement::train(const TieredQ& prefix,uint32_t total_code_bits,const float* sample,size_t rows,
                                           uint64_t seed) {
  if (total_code_bits<=prefix.config().code_bits) bad("refinement needs a larger budget than its prefix");
  auto config=prefix.config();
  config.code_bits=total_code_bits; config.rotate_tiers=false;
  const auto finer=TieredQ::train(config,sample,rows,seed);
  const size_t d=prefix.model_dimensions(),m1=prefix.coded_dimensions(),mf=finer.coded_dimensions();
  if (mf<m1) bad("refinement's finer model codes fewer ranks than the prefix");
  double centroid_error=0,centroid_scale=0;
  for (size_t i=0;i<prefix.centroids_.size();++i) {
    centroid_error=std::max(centroid_error,double(std::fabs(prefix.centroids_[i]-finer.centroids_[i])));
    centroid_scale=std::max(centroid_scale,double(std::fabs(prefix.centroids_[i])));
  }
  if (finer.centroids_.size()!=prefix.centroids_.size() || centroid_error>centroid_tolerance*std::max(centroid_scale,1e-30))
    bad("refinement must train on the prefix's sample and seed (centroids differ)");
  // Model-space bases as row-major d x m floats.
  std::vector<float> coarse(product(d,m1)),fine(product(d,mf));
  for (size_t i=0;i<coarse.size();++i) coarse[i]=prefix.basis_value(i);
  for (size_t i=0;i<fine.size();++i) fine[i]=finer.basis_value(i);
  std::vector<uint32_t> finer_bits;
  for (const auto& tier:finer.tiers()) finer_bits.insert(finer_bits.end(),tier.dimensions,tier.bits);

  TieredQRefinement out;
  out.prefix_=prefix;
  uint32_t offset=0;
  for (const auto& tier:prefix.tiers()) {
    // Each prefix block must span the finer model's ranks at its positions (rotation only).
    double span=0,bits=0;
    for (uint32_t a=0;a<tier.dimensions;++a) for (uint32_t b=0;b<tier.dimensions;++b) {
      double dot=0;
      for (size_t j=0;j<d;++j) dot+=double(coarse[j*m1+tier.start+a])*fine[j*mf+tier.start+b];
      span+=dot*dot;
    }
    if (std::fabs(span-tier.dimensions)>span_tolerance*tier.dimensions)
      bad("refinement must train on the prefix's sample and seed (PCA ranks differ)");
    for (uint32_t a=0;a<tier.dimensions;++a) bits+=finer_bits[tier.start+a];
    const double target=std::round(bits/tier.dimensions)-double(tier.bits);
    const uint32_t gain=uint32_t(std::clamp(target,0.0,double(max_bits-tier.bits)));
    out.blocks_.push_back({tier.start,tier.dimensions,tier.bits,tier.bit_offset,gain,offset,{}});
    offset+=gain*tier.dimensions;
  }
  // Ranks only the suffix codes: runs of equal finer bits, each rotated as training rotates a tier.
  const size_t extra=mf-m1;
  out.extra_basis_.assign(product(d,extra),0);
  for (size_t start=m1;start<mf;) {
    size_t end=start;
    while (end<mf && finer_bits[end]==finer_bits[start]) ++end;
    const uint32_t width=uint32_t(end-start);
    const auto rotation=tieredq_detail::seeded_rotation(int(width),seed+start);
    for (size_t j=0;j<d;++j) for (uint32_t c=0;c<width;++c) {
      double sum=0;
      for (uint32_t k=0;k<width;++k) sum+=double(fine[j*mf+start+k])*rotation[size_t(k)*width+c];
      out.extra_basis_[j*extra+start-m1+c]=tieredq_detail::half_encode(sum);
    }
    out.runs_.push_back({uint32_t(start),width,finer_bits[start],offset,{}});
    offset+=finer_bits[start]*width;
    start=end;
  }
  out.suffix_bits_=offset;
  out.extra_mean_.assign(extra,0.0f); out.extra_sigma_.assign(extra,1.0f);
  out.prepare();
  const size_t fitted=std::min(rows,level_rows);
  std::vector<float> model_rows(product(fitted,d));
  prefix.lift(sample,fitted,false,model_rows.data());
  out.fit_levels(model_rows.data(),fitted,prefix.encode(sample,fitted));
  out.prepare();
  return out;
}

void TieredQRefinement::prepare() {
  const size_t d=prefix_.model_dimensions(),extra=extra_mean_.size();
  // basis * sigma per rank as int8 at a per-rank scale, in rank panels.
  extra_panels_.assign(tieredq_detail::basis_panel_size(d,extra),0);
  extra_panel_scale_.assign(extra,0.0f);
  auto coefficient=[&](size_t j,size_t e) { return double(tieredq_detail::half_decode(extra_basis_[j*extra+e]))*extra_sigma_[e]; };
  for (size_t e=0;e<extra;++e) {
    double peak=0;
    for (size_t j=0;j<d;++j) peak=std::max(peak,std::fabs(coefficient(j,e)));
    const double scale=peak>0 ? peak/panel_levels : 1.0;
    extra_panel_scale_[e]=float(scale);
    for (size_t j=0;j<d;++j)
      extra_panels_[int8_panel_position(d,extra,j,e)]=int8_t(std::lround(coefficient(j,e)/scale));
  }
  extra_mean_ratio_.resize(extra);
  for (size_t e=0;e<extra;++e) extra_mean_ratio_[e]=extra_mean_[e]/extra_sigma_[e];
  level_table_.clear(); segments_.clear();
  auto add=[&](const std::vector<float>& levels,uint32_t start,uint32_t dimensions,uint32_t prefix_bit,uint32_t prefix_bits,
               uint32_t suffix_bit,uint32_t suffix_bits) {
    const uint32_t table=uint32_t(level_table_.size()),size=uint32_t(levels.size());
    level_table_.insert(level_table_.end(),levels.begin(),levels.end());
    for (uint32_t j=0;j<dimensions;j+=segment_lanes)
      segments_.push_back({start+j,std::min(segment_lanes,dimensions-j),prefix_bit+j*prefix_bits,prefix_bits,
                           suffix_bit+j*suffix_bits,suffix_bits,suffix_bits*(prefix_bits>0),table,size});
  };
  for (const auto& b:blocks_) add(b.levels,b.start,b.dimensions,b.prefix_offset,b.bits,b.suffix_offset,b.gain);
  extra_segment_=segments_.size();
  for (const auto& run:runs_) add(run.levels,run.start,run.dimensions,0,0,run.suffix_offset,run.bits);
  level_table_.resize(level_table_.size()+level_padding,0.0f);
  encoder_=std::make_shared<tieredq_detail::RefinementEncoder>();
}

const tieredq_detail::RefinementEncoder& TieredQRefinement::encoder() const {
  std::call_once(encoder_->once,[&] {
    const size_t d=prefix_.model_dimensions(),m1=prefix_.coded_dimensions(),extra=extra_mean_.size();
    const size_t k=prefix_.config().num_centroids;
    auto& state=*encoder_;
    state.panels=tieredq_detail::pack_panels(d,m1+extra,[&](size_t j,size_t r) {
      return r<m1 ? prefix_.basis_value(j*m1+r) : tieredq_detail::half_decode(extra_basis_[j*extra+r-m1]);
    });
    state.centroid_dots.resize(product(k,state.panels.padded_columns()));
    tieredq_detail::float_matrix(prefix_.centroids_.data(),k,d,state.panels,state.centroid_dots.data(),state.panels.padded_columns());
  });
  return *encoder_;
}

void TieredQRefinement::project(const float* model_rows,size_t count,const uint8_t* prefix_rows,std::vector<float>& out) const {
  const size_t d=prefix_.model_dimensions(),m1=prefix_.coded_dimensions(),p=positions();
  const auto& state=encoder();
  const size_t stride=state.panels.padded_columns(),prefix_stride=prefix_.bytes_per_vector();
  std::vector<float> dots(product(count,stride));
  tieredq_detail::float_matrix(model_rows,count,d,state.panels,dots.data(),stride);
  out.resize(product(count,p));
  for (size_t i=0;i<count;++i) {
    const uint8_t cid=prefix_rows[i*prefix_stride+prefix_.code_bytes()];
    if (cid>=prefix_.config().num_centroids) bad("invalid centroid id");
    const float* shift=state.centroid_dots.data()+size_t(cid)*stride;
    for (size_t r=0;r<p;++r)
      out[i*p+r]=dots[i*stride+r]-shift[r]-(r<m1 ? prefix_.mean_[r] : extra_mean_[r-m1]);
  }
}

void TieredQRefinement::fit_levels(const float* model_rows,size_t rows,const std::vector<uint8_t>& prefix_rows) {
  const size_t m1=prefix_.coded_dimensions(),p=positions(),prefix_stride=prefix_.bytes_per_vector();
  const tieredq_detail::Layout layout{prefix_.code_bytes()};
  std::vector<float> v;
  project(model_rows,rows,prefix_rows.data(),v);
  for (size_t e=0;e<extra_mean_.size();++e) {
    double sum=0,square=0;
    for (size_t i=0;i<rows;++i) { const double x=v[i*p+m1+e]; sum+=x; square+=x*x; }
    const double mean=sum/double(rows);
    extra_mean_[e]=float(mean);
    extra_sigma_[e]=float(std::sqrt(std::max(square/double(rows)-mean*mean,0.0))+1e-30);
  }
  for (auto& run:runs_) {
    std::vector<double> z;
    for (size_t i=0;i<rows;++i) for (uint32_t c=0;c<run.dimensions;++c) {
      const size_t e=run.start-m1+c;
      z.push_back((v[i*p+run.start+c]-extra_mean_[e])/extra_sigma_[e]);
    }
    run.levels=lloyd(z,size_t(1)<<run.bits,0.0f);
  }
  for (auto& block:blocks_) {
    const auto& prefix_levels=prefix_.levels_[block.bits];
    if (!block.gain) { block.levels=prefix_levels; continue; }
    std::vector<std::vector<double>> cells(prefix_levels.size());
    for (size_t i=0;i<rows;++i) {
      const uint8_t* doc=prefix_rows.data()+i*prefix_stride;
      const double factor=layout.factor(doc);
      if (factor==0) continue;
      for (uint32_t j=0;j<block.dimensions;++j) {
        const size_t r=block.start+j;
        cells[read_bits(doc,block.prefix_offset+j*block.bits,block.bits)].push_back(v[i*p+r]/(factor*prefix_.sigma_[r]));
      }
    }
    block.levels.clear();
    for (size_t cell=0;cell<cells.size();++cell) {
      const auto sub=lloyd(cells[cell],size_t(1)<<block.gain,prefix_levels[cell]);
      block.levels.insert(block.levels.end(),sub.begin(),sub.end());
    }
  }
}

void TieredQRefinement::encode(const float* rows,size_t count,const uint8_t* prefix_rows,uint8_t* out) const {
  if (count && (!rows || !prefix_rows || !out)) bad("null refinement buffer");
  const size_t d=prefix_.model_dimensions(),inputs=prefix_.config().dimensions,m1=prefix_.coded_dimensions(),p=positions();
  const size_t prefix_stride=prefix_.bytes_per_vector(),stride=bytes_per_vector(),code_bytes=suffix_code_bytes();
  const auto& state=encoder();
  const size_t panel_stride=state.panels.padded_columns();
  for (size_t i=0;i<product(count,inputs);++i) if (!std::isfinite(rows[i])) bad("nonfinite vector data");
  const tieredq_detail::Layout layout{prefix_.code_bytes()};
#pragma omp parallel for schedule(dynamic)
  for (size_t first=0;first<count;first+=encode_block_rows) {
    const size_t block_rows=std::min(encode_block_rows,count-first);
    std::vector<float> model_rows(product(block_rows,d)),v;
    prefix_.lift(rows+first*inputs,block_rows,false,model_rows.data());
    project(model_rows.data(),block_rows,prefix_rows+first*prefix_stride,v);
    std::vector<double> value(p);
    for (size_t i=0;i<block_rows;++i) {
      const uint8_t* prefix_doc=prefix_rows+(first+i)*prefix_stride;
      uint8_t* doc=out+(first+i)*stride;
      std::fill_n(doc,stride,0);
      const double factor=layout.factor(prefix_doc);
      for (const auto& block:blocks_) {
        const size_t sub=size_t(1)<<block.gain;
        for (uint32_t j=0;j<block.dimensions;++j) {
          const size_t r=block.start+j;
          const uint32_t cell=read_bits(prefix_doc,block.prefix_offset+j*block.bits,block.bits);
          const double unit=factor*prefix_.sigma_[r];
          const float* levels=block.levels.data()+size_t(cell)*sub;
          const uint32_t code=unit!=0 ? nearest(levels,sub,v[i*p+r]/unit) : 0;
          if (block.gain) tieredq_detail::write_code(doc,block.suffix_offset+j*block.gain,block.gain,code);
          value[r]=unit*levels[code];
        }
      }
      double dot=0,norm=0;
      for (const auto& run:runs_) for (uint32_t c=0;c<run.dimensions;++c) {
        const size_t r=run.start+c; const double sigma=extra_sigma_[r-m1];
        const uint32_t code=nearest(run.levels.data(),run.levels.size(),v[i*p+r]/sigma);
        tieredq_detail::write_code(doc,run.suffix_offset+c*run.bits,run.bits,code);
        value[r]=sigma*run.levels[code]; dot+=v[i*p+r]*value[r]; norm+=value[r]*value[r];
      }
      const double scale=half_round(norm>0 ? dot/norm : 0.0);
      write_half(doc+code_bytes,scale);
      for (size_t r=m1;r<p;++r) value[r]*=scale;
      // Correction: the reconstruction's remaining error along the centroid, c.(u - u_hat).
      const uint8_t cid=prefix_doc[prefix_.code_bytes()];
      const float* center=prefix_.centroids_.data()+size_t(cid)*d;
      const float* shift=state.centroid_dots.data()+size_t(cid)*panel_stride;
      double error=0;
      for (size_t j=0;j<d;++j) error+=(double(model_rows[i*d+j])-center[j])*center[j];
      for (size_t r=0;r<p;++r) error-=double(shift[r])*((r<m1 ? prefix_.mean_[r] : extra_mean_[r-m1])+value[r]);
      write_half(doc+code_bytes+2,error);
    }
  }
}

TieredQRefinementQuery TieredQRefinement::query(const float* vector) const {
  return query(vector,prefix_.query_weights(vector));
}

TieredQRefinementQuery TieredQRefinement::query(const float* vector,TieredQQuery prefix_query) const {
  const size_t d=prefix_.model_dimensions(),m1=prefix_.coded_dimensions(),extra=extra_mean_.size();
  if (prefix_query.weights.size()!=m1 || prefix_query.centroid_scores.size()!=prefix_.config().num_centroids)
    bad("refinement prefix query shape mismatch");
  std::vector<float> lifted(d);
  prefix_.lift(vector,1,true,lifted.data());
  TieredQRefinementQuery q;
  q.prefix=std::move(prefix_query);
  // Weights of the suffix-only ranks (sigma folded into the panels).
  std::vector<float> weights(extra);
  project_int8_panels(extra_panels_.data(),extra_panel_scale_.data(),d,extra,lifted.data(),weights.data());
  q.weights=q.prefix.weights;
  q.weights.insert(q.weights.end(),weights.begin(),weights.end());
  for (size_t e=0;e<extra;++e) q.extra_mean+=weights[e]*extra_mean_ratio_[e];
  return q;
}

void TieredQRefinement::score(const uint8_t* prefix_rows,const uint8_t* suffix_rows,size_t count,
                              const TieredQRefinementQuery& query,float* scores,TieredQKernel kernel) const {
  if (count && (!prefix_rows || !suffix_rows || !scores)) bad("null refinement buffer");
  if (query.weights.size()!=positions()) bad("refinement query shape mismatch");
  const size_t prefix_stride=prefix_.bytes_per_vector(),stride=bytes_per_vector(),code_bytes=suffix_code_bytes();
  const tieredq_detail::Layout layout{prefix_.code_bytes()};
  auto sum=segment_sum_scalar<Segment>;
#ifdef DV_TIEREDQ_REFINEMENT_X86
  if (resolve_kernel(kernel)==TieredQKernel::avx512) sum=segment_sum_avx512<Segment>;
#else
  resolve_kernel(kernel);
#endif
  const Segment* segments=segments_.data();
  const Segment* extra=segments+extra_segment_;
  const Segment* end=segments+segments_.size();
  thread_local std::vector<uint8_t> prefix_doc,doc;
  prefix_doc.assign(prefix_stride+read_padding,0); doc.assign(stride+read_padding,0);
  for (size_t i=0;i<count;++i) {
    std::copy_n(prefix_rows+i*prefix_stride,prefix_stride,prefix_doc.data());
    std::copy_n(suffix_rows+i*stride,stride,doc.data());
    const uint8_t cid=prefix_doc[prefix_.code_bytes()];
    if (cid>=query.prefix.centroid_scores.size()) bad("invalid centroid id");
    const float refined=sum(prefix_doc.data(),doc.data(),segments,extra,level_table_.data(),query.weights.data());
    const float rest=sum(prefix_doc.data(),doc.data(),extra,end,level_table_.data(),query.weights.data());
    const float scale=tieredq_detail::half_decode(read_half(doc.data()+code_bytes));
    const float correction=tieredq_detail::half_decode(read_half(doc.data()+code_bytes+2));
    scores[i]=query.prefix.centroid_scores[cid]+query.extra_mean+layout.factor(prefix_doc.data())*refined+scale*rest
              +query.prefix.correction_weights[cid]*correction;
  }
}

std::vector<uint8_t> TieredQRefinement::serialize() const {
  tieredq_detail::Writer w;
  w.u32(refinement_magic); w.u32(refinement_format); w.u32(prefix_.model_id());
  w.u64(suffix_bits_);
  w.u32(uint32_t(blocks_.size()));
  for (const auto& b:blocks_) {
    w.u32(b.start); w.u32(b.dimensions); w.u32(b.bits); w.u32(b.prefix_offset); w.u32(b.gain); w.u32(b.suffix_offset);
    w.floats(b.levels);
  }
  w.u32(uint32_t(runs_.size()));
  for (const auto& r:runs_) { w.u32(r.start); w.u32(r.dimensions); w.u32(r.bits); w.u32(r.suffix_offset); w.floats(r.levels); }
  w.halves(extra_basis_); w.floats(extra_mean_); w.floats(extra_sigma_);
  return std::move(w.data);
}

TieredQRefinement TieredQRefinement::deserialize(const std::vector<uint8_t>& data,TieredQ prefix) {
  tieredq_detail::Reader r{data};
  if (r.u32()!=refinement_magic) bad("not a TieredQ refinement");
  if (r.u32()!=refinement_format) bad("unsupported refinement format");
  if (r.u32()!=prefix.model_id()) bad("refinement belongs to a different prefix model");
  TieredQRefinement out;
  out.prefix_=std::move(prefix);
  out.suffix_bits_=size_t(r.u64());
  const size_t m1=out.prefix_.coded_dimensions(),d=out.prefix_.model_dimensions();
  uint64_t bits=0;
  const uint32_t blocks=r.u32();
  if (blocks!=out.prefix_.tiers().size()) bad("refinement blocks do not match the prefix tiers");
  for (uint32_t i=0;i<blocks;++i) {
    Block b{r.u32(),r.u32(),r.u32(),r.u32(),r.u32(),r.u32(),r.floats()};
    const auto& tier=out.prefix_.tiers()[i];
    if (b.start!=tier.start || b.dimensions!=tier.dimensions || b.bits!=tier.bits || b.prefix_offset!=tier.bit_offset ||
        b.bits+b.gain>max_bits || b.suffix_offset!=bits || b.levels.size()!=size_t(1)<<(b.bits+b.gain))
      bad("invalid refinement block");
    bits+=uint64_t(b.gain)*b.dimensions;
    out.blocks_.push_back(std::move(b));
  }
  const uint32_t runs=r.u32();
  size_t next=m1;
  for (uint32_t i=0;i<runs;++i) {
    Run run{r.u32(),r.u32(),r.u32(),r.u32(),r.floats()};
    if (run.start!=next || !run.dimensions || !run.bits || run.bits>max_bits || run.suffix_offset!=bits ||
        run.levels.size()!=size_t(1)<<run.bits)
      bad("invalid refinement run");
    bits+=uint64_t(run.bits)*run.dimensions; next+=run.dimensions;
    out.runs_.push_back(std::move(run));
  }
  out.extra_basis_=r.halves(); out.extra_mean_=r.floats(); out.extra_sigma_=r.floats();
  const size_t extra=next-m1;
  if (bits!=out.suffix_bits_ || out.extra_mean_.size()!=extra || out.extra_sigma_.size()!=extra ||
      out.extra_basis_.size()!=product(d,extra) || r.remaining())
    bad("invalid refinement shape");
  if (!std::all_of(out.extra_sigma_.begin(),out.extra_sigma_.end(),[](float sigma) { return sigma>0; }))
    bad("invalid refinement sigma");
  out.prepare();
  return out;
}
} // namespace dv
