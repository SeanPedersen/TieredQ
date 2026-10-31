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
enum LayoutArray : size_t { prefix_bit, prefix_mask, prefix_shift, suffix_bit, suffix_mask, level_offset, layout_arrays };
// Rows copied into padded buffers so 32-bit code reads never leave them.
constexpr size_t read_padding=4;
// Sum over positions [first, last) of weights[r] * levels[offset(r) + (prefix code << shift | suffix code)].
float level_sum_scalar(const uint8_t* prefix,const uint8_t* suffix,const uint32_t* layout,size_t positions,size_t first,
                       size_t last,const float* levels,const float* weights) {
  float sum=0;
  for (size_t r=first;r<last;++r) {
    auto code=[&](LayoutArray bit,LayoutArray mask,const uint8_t* doc) {
      const uint32_t offset=layout[bit*positions+r];
      uint32_t word; std::memcpy(&word,doc+offset/8,sizeof(word));
      return (word>>(offset%8))&layout[mask*positions+r];
    };
    const uint32_t index=layout[level_offset*positions+r]+
        (code(prefix_bit,prefix_mask,prefix)<<layout[prefix_shift*positions+r]|code(suffix_bit,suffix_mask,suffix));
    sum+=weights[r]*levels[index];
  }
  return sum;
}
#ifdef DV_TIEREDQ_REFINEMENT_X86
// Codes of 16 positions: (32-bit word at bit / 8) >> bit % 8, masked.
__attribute__((target("avx512f")))
inline __m512i codes_avx512(const uint32_t* bits,const uint32_t* masks,const uint8_t* doc,__mmask16 live) {
  const __m512i offset=_mm512_maskz_loadu_epi32(live,bits);
  const __m512i word=_mm512_mask_i32gather_epi32(_mm512_setzero_si512(),live,_mm512_srli_epi32(offset,3),doc,1);
  return _mm512_and_si512(_mm512_srlv_epi32(word,_mm512_and_si512(offset,_mm512_set1_epi32(7))),
                          _mm512_maskz_loadu_epi32(live,masks));
}
__attribute__((target("avx512f")))
float level_sum_avx512(const uint8_t* prefix,const uint8_t* suffix,const uint32_t* layout,size_t positions,size_t first,
                       size_t last,const float* levels,const float* weights) {
  constexpr size_t lanes=16;
  const uint32_t* at[layout_arrays];
  for (size_t array=0;array<layout_arrays;++array) at[array]=layout+array*positions;
  __m512 sum=_mm512_setzero_ps();
  for (size_t r=first;r<last;r+=lanes) {
    const __mmask16 live=__mmask16(last-r>=lanes ? 0xffff : (1u<<(last-r))-1);
    const __m512i high=_mm512_sllv_epi32(codes_avx512(at[prefix_bit]+r,at[prefix_mask]+r,prefix,live),
                                         _mm512_maskz_loadu_epi32(live,at[prefix_shift]+r));
    const __m512i index=_mm512_add_epi32(_mm512_maskz_loadu_epi32(live,at[level_offset]+r),
                                         _mm512_or_si512(high,codes_avx512(at[suffix_bit]+r,at[suffix_mask]+r,suffix,live)));
    const __m512 level=_mm512_mask_i32gather_ps(_mm512_setzero_ps(),live,index,levels,4);
    sum=_mm512_add_ps(sum,_mm512_mul_ps(_mm512_maskz_loadu_ps(live,weights+r),level));
  }
  return _mm512_reduce_add_ps(sum);
}
#endif
using LevelSum=float (*)(const uint8_t*,const uint8_t*,const uint32_t*,size_t,size_t,size_t,const float*,const float*);
LevelSum level_sum() {
#ifdef DV_TIEREDQ_REFINEMENT_X86
  static const bool avx512=__builtin_cpu_supports("avx512f") && !std::getenv("DV_TIEREDQ_NO_AVX512");
  if (avx512) return level_sum_avx512;
#endif
  return level_sum_scalar;
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
  extra_panels_=tieredq_detail::basis_panels_from_rows(extra_basis_,d,extra);
  extra_mean_ratio_.resize(extra);
  for (size_t e=0;e<extra;++e) extra_mean_ratio_[e]=extra_mean_[e]/extra_sigma_[e];
  const size_t p=positions();
  code_layout_.assign(layout_arrays*p,0);
  auto set=[&](LayoutArray array,size_t r,uint32_t value) { code_layout_[array*p+r]=value; };
  level_table_.clear();
  for (const auto& block:blocks_) {
    const uint32_t table=uint32_t(level_table_.size());
    level_table_.insert(level_table_.end(),block.levels.begin(),block.levels.end());
    for (uint32_t j=0;j<block.dimensions;++j) {
      const size_t r=block.start+j;
      set(prefix_bit,r,block.prefix_offset+j*block.bits); set(prefix_mask,r,(1u<<block.bits)-1); set(prefix_shift,r,block.gain);
      set(suffix_bit,r,block.suffix_offset+j*block.gain); set(suffix_mask,r,(1u<<block.gain)-1); set(level_offset,r,table);
    }
  }
  for (const auto& run:runs_) {
    const uint32_t table=uint32_t(level_table_.size());
    level_table_.insert(level_table_.end(),run.levels.begin(),run.levels.end());
    for (uint32_t c=0;c<run.dimensions;++c) {
      const size_t r=run.start+c;
      set(suffix_bit,r,run.suffix_offset+c*run.bits); set(suffix_mask,r,(1u<<run.bits)-1); set(level_offset,r,table);
    }
  }
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
  // Sigma folded into the panels: weights of the suffix-only ranks.
  std::vector<float> weights(extra);
  tieredq_detail::project_basis_panels({extra_panels_.data(),d,extra,extra_sigma_.data(),lifted.data(),d,weights.data(),extra},1);
  q.weights=q.prefix.weights;
  q.weights.insert(q.weights.end(),weights.begin(),weights.end());
  for (size_t e=0;e<extra;++e) q.extra_mean+=weights[e]*extra_mean_ratio_[e];
  return q;
}

void TieredQRefinement::score(const uint8_t* prefix_rows,const uint8_t* suffix_rows,size_t count,
                              const TieredQRefinementQuery& query,float* scores) const {
  if (count && (!prefix_rows || !suffix_rows || !scores)) bad("null refinement buffer");
  if (query.weights.size()!=positions()) bad("refinement query shape mismatch");
  const size_t prefix_stride=prefix_.bytes_per_vector(),stride=bytes_per_vector(),code_bytes=suffix_code_bytes();
  const size_t m1=prefix_.coded_dimensions(),p=positions();
  const tieredq_detail::Layout layout{prefix_.code_bytes()};
  const LevelSum sum=level_sum();
  thread_local std::vector<uint8_t> prefix_doc,doc;
  prefix_doc.assign(prefix_stride+read_padding,0); doc.assign(stride+read_padding,0);
  for (size_t i=0;i<count;++i) {
    std::copy_n(prefix_rows+i*prefix_stride,prefix_stride,prefix_doc.data());
    std::copy_n(suffix_rows+i*stride,stride,doc.data());
    const uint8_t cid=prefix_doc[prefix_.code_bytes()];
    if (cid>=query.prefix.centroid_scores.size()) bad("invalid centroid id");
    const float refined=sum(prefix_doc.data(),doc.data(),code_layout_.data(),p,0,m1,level_table_.data(),query.weights.data());
    const float extra=sum(prefix_doc.data(),doc.data(),code_layout_.data(),p,m1,p,level_table_.data(),query.weights.data());
    const float scale=tieredq_detail::half_decode(read_half(doc.data()+code_bytes));
    const float correction=tieredq_detail::half_decode(read_half(doc.data()+code_bytes+2));
    scores[i]=query.prefix.centroid_scores[cid]+query.extra_mean+layout.factor(prefix_doc.data())*refined+scale*extra
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
