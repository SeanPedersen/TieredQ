// Projected mean and variance with row-ordered reductions; tiles columns for cache locality.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <utility>
#include "tieredq/tieredq.h"

namespace dv::tieredq_detail {
inline std::pair<float,float> projected_distribution(double average,double variance) {
  constexpr float sigma_floor=1e-9f;
  return {float(average),float(std::sqrt(std::max(0.0,variance)))+sigma_floor};
}
inline std::pair<float,float> projected_moments(double sum,double square,size_t rows) {
  const double average=sum/rows;
  return projected_distribution(average,square/rows-average*average);
}
inline void spectral_projection_stats(const std::vector<double>& eigenvalues,const std::vector<int>& ranks,
                                      const std::vector<TieredQTier>& tiers,const std::vector<std::vector<float>>& rotations,
                                      const std::vector<double>& residual_mean,const std::vector<float>& basis,size_t rows,
                                      std::vector<float>& mean,std::vector<float>& sigma) {
  const size_t dimensions=eigenvalues.size(),columns=ranks.size();
  mean.resize(columns); sigma.resize(columns);
  for (size_t tier=0;tier<tiers.size();++tier) {
    const auto& layout=tiers[tier]; const auto& rotation=rotations[tier];
    for (size_t j=0;j<layout.dimensions;++j) {
      const size_t column=layout.start+j;
      double average=0,variance=0;
      for (size_t i=0;i<dimensions;++i) average+=residual_mean[i]*basis[i*columns+column];
      for (size_t i=0;i<layout.dimensions;++i) {
        const double weight=rotation[i*layout.dimensions+j];
        variance+=weight*weight*std::max(0.0,eigenvalues[dimensions-1-ranks[layout.start+i]]);
      }
      const auto moments=projected_distribution(average,variance*double(rows-1)/rows);
      mean[column]=moments.first; sigma[column]=moments.second;
    }
  }
}
inline void projection_stats_reference(const std::vector<float>& projected,size_t rows,size_t columns,
                                       std::vector<float>& mean,std::vector<float>& sigma) {
  mean.resize(columns); sigma.resize(columns);
  for (size_t j=0;j<columns;++j) {
    double sum=0,square=0;
    for (size_t i=0;i<rows;++i) { double v=projected[i*columns+j]; sum+=v; square+=v*v; }
    const auto moments=projected_moments(sum,square,rows);
    mean[j]=moments.first; sigma[j]=moments.second;
  }
}
inline void projection_stats(const std::vector<float>& projected,size_t rows,size_t columns,
                             std::vector<float>& mean,std::vector<float>& sigma) {
  constexpr size_t column_tile=8;
  mean.resize(columns); sigma.resize(columns);
#pragma omp parallel for schedule(static)
  for (size_t block=0;block<(columns+column_tile-1)/column_tile;++block) {
    const size_t begin=block*column_tile,width=std::min(column_tile,columns-begin);
    std::array<double,column_tile> sums{},squares{};
    for (size_t i=0;i<rows;++i) for (size_t j=0;j<width;++j) {
      const double value=projected[i*columns+begin+j];
      sums[j]+=value; squares[j]+=value*value;
    }
    for (size_t j=0;j<width;++j) {
      const auto moments=projected_moments(sums[j],squares[j],rows);
      mean[begin+j]=moments.first; sigma[begin+j]=moments.second;
    }
  }
}
} // namespace dv::tieredq_detail
