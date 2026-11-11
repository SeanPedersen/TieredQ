// encoding: fused FP32 assignment/projection, ordered quantization, projected FP64 correction.
#pragma once
#include "tieredq_layout.h"
#include "tieredq_lookup_layout.h"
#include "tieredq_profile.h"
#include "tieredq_encode_grid.h"
#include "tieredq_encode_float.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <mutex>
#include <vector>

namespace tq::tieredq_detail {
struct EncodingModel {
  size_t dimensions;
  const std::vector<float>& centroids;
  const std::vector<float>& mean;
  const std::vector<float>& sigma;
  const std::vector<std::vector<float>>& levels;
  const std::vector<TieredQTier>& tiers;
};
struct EncoderState {
  const TieredQPanels& fused_panels;
  const std::vector<double>& centroid_projections;
  const std::vector<double>& half_norms;
  size_t projection_stride() const { return centroid_projections.size()/half_norms.size(); }
};
// Owned encoder-only state derived from an immutable model; serving never reads it.
struct EncoderData {
  TieredQPanels fused_panels;
  std::vector<double> centroid_projections, half_norms;
  EncoderState view() const { return {fused_panels,centroid_projections,half_norms}; }
  size_t bytes() const {
    return fused_panels.data.capacity()*sizeof(float)+(centroid_projections.capacity()+half_norms.capacity())*sizeof(double);
  }
};
// Builds EncoderData on first use and keeps it; copies of one model share the instance.
class LazyEncoder {
 public:
  template<class Build> const EncoderData& get(Build build) {
    std::call_once(once_,[&] { data_=build(); ready_.store(true,std::memory_order_release); });
    return data_;
  }
  size_t bytes() const { return sizeof(*this)+(ready_.load(std::memory_order_acquire) ? data_.bytes() : 0); }
 private:
  std::once_flag once_;
  std::atomic<bool> ready_{false};
  EncoderData data_;
};
struct EncodingWorkspace {
  std::vector<double> v, z, chosen;
  std::vector<uint32_t> chosen_codes;
  std::array<std::vector<double>,widest_fine_code_bits+1> thresholds;
  std::array<std::array<double,16>,5> small_levels{};
  explicit EncodingWorkspace(const EncodingModel& model)
      : v(model.sigma.size()), z(v.size()), chosen(v.size()), chosen_codes(v.size()) {
    for (size_t b=1;b<model.levels.size();++b) {
      const auto& levels=model.levels[b];
      if (b<small_levels.size()) std::copy(levels.begin(),levels.end(),small_levels[b].begin());
      for (size_t i=0;i+1<levels.size();++i)
        thresholds[b].push_back((double(levels[i])+levels[i+1])/2);
    }
  }
};
inline double scale_grid_search(const EncodingModel& model,double rms,EncodingWorkspace& work,
                              TieredQKernel kernel=TieredQKernel::automatic) {
  for (auto& value:work.z) value/=rms;
  std::array<double,scale_grid_storage> dots{},norms{};
  scale_grid_terms(model,work,dots.data(),norms.data(),kernel);
  double best=-INFINITY; size_t best_point=0;
  for (size_t point=0;point<scale_grid_points;++point) {
    const double energy=dots[point]*dots[point]/std::max(norms[point],1e-30);
    // Strict improvement retains the first grid point on ties.
    if (energy>best) { best=energy; best_point=point; }
  }
  for (const auto& tier:model.tiers) for (size_t j=0;j<tier.dimensions;++j) {
    const size_t r=tier.start+j;
    const unsigned code=quantize_code(work.z[r]*scale_grid()[best_point],work.thresholds[tier.bits]);
    work.chosen_codes[r]=code; work.chosen[r]=double(model.levels[tier.bits][code])*model.sigma[r];
  }
  return dots[best_point];
}
inline void quantize_row(const EncodingModel& model,const Layout& layout,const double* projection,
                             size_t cid,uint8_t* doc,EncodingWorkspace& work,TieredQKernel kernel) {
  const size_t m=model.sigma.size();
  double square=0,numerator=0;
  for (size_t r=0;r<m;++r) {
    work.v[r]=projection[r]-model.mean[r];
    work.z[r]=work.v[r]/model.sigma[r];
    square+=work.z[r]*work.z[r]; numerator+=work.v[r]*work.v[r];
  }
  const double rms=std::sqrt(square/m)+1e-30;
  const double best_dot=scale_grid_search(model,rms,work,kernel);
  std::fill_n(doc,layout.code_bytes+layout.metadata_bytes(),0); doc[layout.centroid_offset()]=uint8_t(cid);
  for (const auto& tier:model.tiers) for (size_t j=0;j<tier.dimensions;++j)
    write_code(doc,tier.bit_offset+j*tier.bits,tier.bits,work.chosen_codes[tier.start+j]);
  layout.store_half(doc,layout.factor_offset(),std::fabs(best_dot)>1e-12 ? numerator/best_dot : 0);
  const double factor=layout.factor(doc);
  for (size_t r=0;r<m;++r) work.chosen[r]=model.mean[r]+factor*work.chosen[r];
}
inline void encode_block(const EncodingModel& model,const EncoderState& encoder,const Layout& layout,const float* rows,size_t count,
                            uint8_t* output,float* corrections,TieredQKernel kernel=TieredQKernel::automatic, StageProfile* sink=nullptr) {
  StageProfile profile("tieredq_encode_worker_seconds",sink);
  const size_t d=model.dimensions,m=model.sigma.size(),stride=layout.code_bytes+layout.metadata_bytes();
  const size_t projection_stride=encoder.projection_stride(),fused_stride=encoder.fused_panels.padded_columns();
  std::vector<float> fused(count*fused_stride);
  std::vector<double> projected(m);
  profile.mark("workspace");
  float_matrix(rows,count,d,encoder.fused_panels,fused.data(),fused_stride,kernel);
  std::vector<double> dots(fused.begin(),fused.end());
  // Finite FP32 inputs can overflow a dot product; retain the FP64 range on that rare block.
  if (!std::all_of(dots.begin(),dots.end(),[](double value) { return std::isfinite(value); })) {
    const std::vector<double> input(rows,rows+count*d);
    ordered_matrix(input.data(),count,encoder.fused_panels,dots.data(),kernel);
  }
  profile.mark("assignment_projection");
  EncodingWorkspace work(model);
  for (size_t i=0;i<count;++i) {
    size_t cid=0; double best=INFINITY;
    for (size_t c=0;c<encoder.half_norms.size();++c) {
      const double distance=encoder.half_norms[c]-dots[i*fused_stride+m+c];
      if (distance<best) { best=distance; cid=c; }
    }
    const double* shift=encoder.centroid_projections.data()+cid*projection_stride;
    for (size_t r=0;r<m;++r) projected[r]=double(dots[i*fused_stride+r])-shift[r];
    uint8_t* doc=output+i*stride;
    quantize_row(model,layout,projected.data(),cid,doc,work,kernel);
    const float* center=model.centroids.data()+cid*d;
    double exact=0,decoded=0;
    for (size_t j=0;j<d;++j) exact+=(double(rows[i*d+j])-center[j])*center[j];
    for (size_t r=0;r<m;++r) decoded+=work.chosen[r]*shift[r];
    layout.store_half(doc,layout.correction_offset(),exact-decoded);
    if (corrections) corrections[i]=layout.correction(doc);
  }
  profile.mark("quantization_correction");
}
} // namespace tq::tieredq_detail
