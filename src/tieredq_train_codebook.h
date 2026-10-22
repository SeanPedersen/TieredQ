// Gaussian distortion estimates for allocating bits to pinned reconstruction tables.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
namespace dv::tieredq_detail {
constexpr double pi = 3.14159265358979323846;
inline double pdf(double x) { return std::isfinite(x) ? std::exp(-x*x/2)/std::sqrt(2*pi) : 0; }
inline double cdf(double x) { return 0.5*std::erfc(-x/std::sqrt(2.0)); }
inline double xpdf(double x) { return std::isfinite(x) ? x*pdf(x) : 0; }
inline double distortion(const std::vector<float>& levels) {
  double result=0;
  for (size_t i=0;i<levels.size();++i) {
    const double lo = i ? (levels[i-1]+double(levels[i]))/2 : -INFINITY;
    const double hi = i+1<levels.size() ? (levels[i]+double(levels[i+1]))/2 : INFINITY;
    const double mass=cdf(hi)-cdf(lo), first=pdf(lo)-pdf(hi);
    result += mass+xpdf(lo)-xpdf(hi)-2*levels[i]*first+levels[i]*levels[i]*mass;
  }
  return result;
}
} // namespace dv::tieredq_detail
