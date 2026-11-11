// Query-only range bounds for the fast-scan checkpoint after the prefix tiers (four by default); no document metadata
// is needed. The suffix range is exact (every code), or with a positive confidence clipped to that many standard
// deviations of the suffix score when coordinates are unit Gaussian and quantized to their nearest level (approximate).
#pragma once
#include "tieredq_scan_int8.h"
#include <cmath>
#include <limits>

namespace tq::tieredq_detail {
constexpr size_t fast_range_prefix_tiers=4;
struct FastRangeBounds { float high,low,magnitude,allowance_scale; };

inline bool fast_range_checkpoint_supported(const FastScanCache& cache,
    const std::vector<TieredQTier>& tiers,size_t prefix_tiers=fast_range_prefix_tiers) {
  return cache.code_lookup() && prefix_tiers>0 && tiers.size()>prefix_tiers;
}
// E[level^2] of a sorted codebook for a unit-Gaussian coordinate: each level owns the cell between its midpoints.
inline double gaussian_level_moment(const std::vector<float>& book) {
  auto cdf=[](double x) { return 0.5*std::erfc(-x/std::sqrt(2.0)); };
  double moment=0,below=0;
  for (size_t i=0;i<book.size();++i) {
    const double above=i+1<book.size() ? cdf(0.5*(double(book[i])+book[i+1])) : 1.0;
    moment+=(above-below)*double(book[i])*book[i];
    below=above;
  }
  return moment;
}
inline float range_round_up(double value) {
  const float converted=float(value);
  return double(converted)<value ? std::nextafter(converted,std::numeric_limits<float>::infinity()) : converted;
}
inline float range_round_down(double value) {
  const float converted=float(value);
  return double(converted)>value ? std::nextafter(converted,-std::numeric_limits<float>::infinity()) : converted;
}
inline FastRangeBounds prepare_fast_range_bounds(const FastScanCache& cache,
    const std::vector<TieredQTier>& tiers,const TieredQQuery& query,size_t prefix_tiers=fast_range_prefix_tiers,
    double confidence=0) {
  if (!fast_range_checkpoint_supported(cache,tiers,prefix_tiers) || tiers.size()!=cache.codebooks.size() ||
      query.weights.size()!=cache.dimensions || !(confidence>=0) || !std::isfinite(confidence))
    throw std::invalid_argument("TieredQ: incompatible fast range checkpoint");
  double high=0,low=0,magnitude=0,variance=0;
  for (size_t ti=tiers.size();ti-- > 0;) {
    const auto& tier=tiers[ti];
    const auto& book=cache.codebooks[ti];
    if (book.empty() || size_t(tier.start)+tier.dimensions>cache.dimensions)
      throw std::invalid_argument("TieredQ: invalid fast range tier");
    const auto [min_it,max_it]=std::minmax_element(book.begin(),book.end());
    const double minimum=*min_it,maximum=*max_it,peak=std::max(std::fabs(minimum),std::fabs(maximum));
    const double moment=ti>=prefix_tiers && confidence>0 ? gaussian_level_moment(book) : 0;
    for (size_t j=0;j<tier.dimensions;++j) {
      const double weight=query.weights[tier.start+j];
      magnitude+=std::fabs(weight)*peak;
      if (ti<prefix_tiers) continue;
      high+=std::max(weight*minimum,weight*maximum);
      low+=std::min(weight*minimum,weight*maximum);
      variance+=weight*weight*moment;
    }
  }
  if (confidence>0) {
    const double spread=confidence*std::sqrt(variance);
    high=std::min(high,spread); low=std::max(low,-spread);
  }
  constexpr double epsilon=double(std::numeric_limits<float>::epsilon());
  constexpr double operations_per_dimension=2.0,finalization_operations=8.0;
  const double rounding_error=(operations_per_dimension*double(cache.dimensions)+finalization_operations)*epsilon;
  if (rounding_error>=1.0)
    throw std::invalid_argument("TieredQ: dimension exceeds range-bound rounding budget");
  return {range_round_up(high),range_round_down(low),range_round_up(magnitude),
      range_round_up(rounding_error/(1.0-rounding_error))};
}
} // namespace tq::tieredq_detail
