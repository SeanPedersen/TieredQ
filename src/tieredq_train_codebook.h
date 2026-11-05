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
constexpr int lloyd_max_iterations=400,inverse_cdf_steps=80;
constexpr double inverse_cdf_bound=40;
inline double inverse_cdf(double p) {
  double lo=-inverse_cdf_bound,hi=inverse_cdf_bound;
  for (int step=0;step<inverse_cdf_steps;++step) { const double mid=(lo+hi)/2; (cdf(mid)<p ? lo : hi)=mid; }
  return (lo+hi)/2;
}
// Fine-tier book (#26): every nearest-level cell of the sorted prefix book splits into 2^fine_bits Gaussian
// Lloyd-Max levels of the standard normal truncated to that cell. Levels stay sorted and cell major, so a code's top
// bits (code >> fine_bits) are its prefix code and the prefix book still scores them alone.
inline std::vector<float> refined_prefix_codebook(const std::vector<float>& prefix,unsigned fine_bits) {
  const size_t cells=prefix.size(),count=size_t(1)<<fine_bits;
  std::vector<float> book; book.reserve(cells*count);
  for (size_t c=0;c<cells;++c) {
    const double lo=c ? (prefix[c-1]+double(prefix[c]))/2 : -INFINITY;
    const double hi=c+1<cells ? (prefix[c]+double(prefix[c+1]))/2 : INFINITY;
    const double base=cdf(lo),mass=cdf(hi)-base;
    std::vector<double> levels(count);
    for (size_t i=0;i<count;++i)
      levels[i]=mass>0 ? std::clamp(inverse_cdf(base+mass*(double(i)+0.5)/double(count)),lo,hi) : prefix[c];
    for (int it=0;it<lloyd_max_iterations && mass>0;++it)
      for (size_t i=0;i<count;++i) {
        const double a=i ? (levels[i-1]+levels[i])/2 : lo, b=i+1<count ? (levels[i]+levels[i+1])/2 : hi;
        const double part=cdf(b)-cdf(a);
        if (part>0) levels[i]=(pdf(a)-pdf(b))/part;
      }
    book.insert(book.end(),levels.begin(),levels.end());
  }
  return book;
}
} // namespace dv::tieredq_detail
