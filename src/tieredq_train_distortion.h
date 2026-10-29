// Empirical per-rank codebook distortion on standardized PCA coordinates of the training residuals (#9).
// Raw mode quantizes each coordinate as is. Shape-gain mode mirrors the encoder, which divides a row by its RMS
// before quantizing and restores the scale afterwards, so a row's error is weighted by its squared RMS.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace dv::tieredq_detail {
enum class DistortionModel { gaussian, raw, shape_gain };
// Sums over rows of squared error, per rank and codebook: column 0 is the uncoded coordinate, column c+1 is
// books[c]. Divide by the row count for mean distortion in units of the coordinate's variance.
class DistortionSums {
public:
  DistortionSums(size_t ranks,std::vector<std::vector<float>> books) : ranks_(ranks),books_(std::move(books)),sums_(ranks*columns()) {
    for (const auto& levels:books_) {
      auto& cuts=thresholds_.emplace_back();
      for (size_t i=0;i+1<levels.size();++i) cuts.push_back((double(levels[i])+levels[i+1])/2);
    }
  }
  size_t columns() const { return books_.size()+1; }
  double mean(size_t rank,size_t column) const { return rows_ ? sums_[rank*columns()+column]/double(rows_) : 0; }
  // z holds rows x ranks standardized coordinates.
  void add(const float* z,size_t rows,bool shape_gain) {
    std::vector<double> scale(rows,1.0);
    if (shape_gain) for (size_t i=0;i<rows;++i) {
      double square=0; for (size_t r=0;r<ranks_;++r) square+=double(z[i*ranks_+r])*z[i*ranks_+r];
      scale[i]=std::sqrt(square/double(ranks_))+1e-30;
    }
#pragma omp parallel for schedule(static)
    for (size_t r=0;r<ranks_;++r) {
      double* out=sums_.data()+r*columns();
      for (size_t i=0;i<rows;++i) {
        const double value=z[i*ranks_+r]/scale[i],weight=scale[i]*scale[i];
        out[0]+=weight*value*value;
        for (size_t c=0;c<books_.size();++c) {
          const auto& cuts=thresholds_[c];
          const size_t code=size_t(std::upper_bound(cuts.begin(),cuts.end(),value)-cuts.begin());
          const double error=value-books_[c][code];
          out[c+1]+=weight*error*error;
        }
      }
    }
    rows_+=rows;
  }
private:
  size_t ranks_,rows_=0;
  std::vector<std::vector<float>> books_;
  std::vector<std::vector<double>> thresholds_;
  std::vector<double> sums_;
};
} // namespace dv::tieredq_detail
