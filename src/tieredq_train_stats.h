// Projected moments from PCA spectra or sampled residuals, including rounded-basis leakage into null directions.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <utility>
#include "tieredq/tieredq.h"
#include "tieredq_encode_float.h"
#include "tieredq_train_rows.h"

namespace dv::tieredq_detail {
inline std::pair<float,float> projected_distribution(double average,double variance) {
  constexpr float sigma_floor=1e-9f;
  return {float(average),float(std::sqrt(std::max(0.0,variance)))+sigma_floor};
}
inline std::pair<float,float> projected_moments(double sum,double square,size_t rows) {
  const double average=sum/rows;
  return projected_distribution(average,square/rows-average*average);
}
// Variance the binary16 basis rounding alone can leak into a direction; eigenvalues are ascending.
inline double rounding_variance(const std::vector<double>& eigenvalues) {
  constexpr double binary16_epsilon=0x1p-10;
  return eigenvalues.back()*binary16_epsilon*binary16_epsilon;
}
inline bool needs_empirical_projection_stats(const std::vector<double>& eigenvalues,const std::vector<int>& ranks) {
  const double noise=rounding_variance(eigenvalues);
  return std::any_of(ranks.begin(),ranks.end(),[&](int rank) {
    return eigenvalues[eigenvalues.size()-1-rank]<=noise;
  });
}
// A rank-deficient sample codes directions it never varied in, whose fitted sigma is only rounding noise. Rows added
// later can carry real energy there: their z then swamps the scale search and the fp16 factor overflows (#35).
// Unobserved directions are assumed to vary at least as much as the least-varying observed one. A sample that varied
// in no direction (one row, duplicates) only reveals its coordinates' magnitude, the centroids' mean square; an
// all-zero sample reveals nothing, so the floor falls back to the coordinates of a unit-norm row.
inline void floor_unobserved_sigma(const std::vector<double>& eigenvalues,const std::vector<float>& centroids,
                                   std::vector<float>& sigma) {
  const size_t dimensions=eigenvalues.size();
  const auto observed=std::upper_bound(eigenvalues.begin(),eigenvalues.end(),rounding_variance(eigenvalues));
  double variance=observed==eigenvalues.end() ? 0 : *observed;
  if (!(variance>0)) {
    for (const float value:centroids) variance+=double(value)*value;
    variance/=double(centroids.size());
  }
  if (!(variance>0)) variance=1.0/double(dimensions);
  const float floor=float(std::sqrt(variance));
  for (auto& value:sigma) value=std::max(value,floor);
}
// Spectral moments describe the orthogonal basis before binary16 rounding. Near the covariance's null space,
// rounding mixes in enough positive-variance directions that those moments no longer describe the stored basis.
// Fit the rounded projections in bounded row chunks, retaining the same input-order moment reductions.
inline void empirical_projection_stats(const TrainingRows& sample,size_t rows,const std::vector<uint32_t>& ids,
                                      const std::vector<float>& centroids,const std::vector<float>& basis,size_t columns,
                                      std::vector<float>& mean,std::vector<float>& sigma) {
  constexpr size_t chunk_rows=64;
  const size_t dimensions=sample.dimensions;
  const auto panels=pack_panels(dimensions,columns,[&](size_t j,size_t r) { return basis[j*columns+r]; });
  const size_t stride=panels.padded_columns();
  std::vector<float> residual(std::min(rows,chunk_rows)*dimensions),projected(std::min(rows,chunk_rows)*stride);
  std::vector<double> sums(columns),squares(columns);
  sample.visit([&](auto value) {
    for (size_t first=0;first<rows;first+=chunk_rows) {
      const size_t count=std::min(chunk_rows,rows-first);
      for (size_t i=0;i<count;++i) for (size_t j=0;j<dimensions;++j)
        residual[i*dimensions+j]=float(double(value(first+i,j))-centroids[size_t(ids[first+i])*dimensions+j]);
      float_matrix(residual.data(),count,dimensions,panels,projected.data(),stride);
      for (size_t i=0;i<count;++i) for (size_t j=0;j<columns;++j) {
        const double v=projected[i*stride+j]; sums[j]+=v; squares[j]+=v*v;
      }
    }
  });
  mean.resize(columns); sigma.resize(columns);
  for (size_t j=0;j<columns;++j) {
    const auto moments=projected_moments(sums[j],squares[j],rows);
    mean[j]=moments.first; sigma[j]=moments.second;
  }
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
