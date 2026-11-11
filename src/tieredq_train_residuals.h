// Center residuals without storing an intermediate matrix; each coordinate sums rows in input order.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace tq::tieredq_detail {
struct CenteredResiduals { std::vector<double> values,mean; };
// value(row, j) reads a sample coordinate.
template<class Value>
CenteredResiduals centered_residuals_with(Value value,size_t rows,size_t dimensions,const uint32_t* ids,
                                          const std::vector<float>& centroids) {
  constexpr size_t column_tile=16;
  CenteredResiduals result; result.mean.resize(dimensions);
#pragma omp parallel for schedule(static)
  for (size_t block=0;block<(dimensions+column_tile-1)/column_tile;++block) {
    const size_t begin=block*column_tile,width=std::min(column_tile,dimensions-begin);
    std::array<double,column_tile> sums{};
    for (size_t i=0;i<rows;++i) {
      const float* center=centroids.data()+size_t(ids[i])*dimensions+begin;
      for (size_t j=0;j<width;++j) sums[j]+=double(value(i,begin+j))-center[j];
    }
    for (size_t j=0;j<width;++j) result.mean[begin+j]=sums[j]/rows;
  }
  result.values.resize(rows*dimensions);
#pragma omp parallel for schedule(static)
  for (size_t i=0;i<rows;++i) {
    const float* center=centroids.data()+size_t(ids[i])*dimensions;
    for (size_t j=0;j<dimensions;++j)
      result.values[i*dimensions+j]=(double(value(i,j))-center[j])-result.mean[j];
  }
  return result;
}
inline CenteredResiduals centered_residuals(const float* sample,size_t rows,size_t dimensions,
                                           const uint32_t* ids,const std::vector<float>& centroids) {
  return centered_residuals_with([&](size_t row,size_t j) { return sample[row*dimensions+j]; },rows,dimensions,ids,
                                 centroids);
}
} // namespace tq::tieredq_detail
