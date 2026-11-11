// Training rows read in place: rows of the input (all, or the ones a sample's ids name, in that order) and, for
// cosine models, each normalized on access exactly as normalize_row does (float(x / norm) with a double norm), so
// training stores neither a gathered nor a normalized copy of the sample.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tq::tieredq_detail {
struct TrainingRows {
  const float* data;
  size_t dimensions;
  const uint32_t* ids = nullptr;  // sample row -> input row; null for the input rows in order
  std::vector<double> norms;      // per sample row when normalizing; empty for rows as given

  const float* row(size_t sample_row) const { return data+size_t(ids ? ids[sample_row] : sample_row)*dimensions; }
  TrainingRows normalized(size_t rows) const {
    TrainingRows result=*this; result.norms.resize(rows);
    for (size_t i=0;i<rows;++i) {
      const float* x=row(i);
      double square=0;
      for (size_t j=0;j<dimensions;++j) square+=double(x[j])*x[j];
      result.norms[i]=std::sqrt(square);
    }
    return result;
  }
  // Calls f with value(row, j), the sample row's model-space coordinate; one callable type per representation.
  template<class F> decltype(auto) visit(F&& f) const {
    if (!norms.empty()) return f([this](size_t r,size_t j) {
      const double norm=norms[r];
      return norm>0 ? float(row(r)[j]/norm) : 0.0f;
    });
    if (ids) return f([this](size_t r,size_t j) { return data[size_t(ids[r])*dimensions+j]; });
    return f([this](size_t r,size_t j) { return data[r*dimensions+j]; });
  }
  // Sample rows [first, first + count) contiguous: in place, or gathered / normalized into buffer.
  const float* block(size_t first,size_t count,std::vector<float>& buffer) const {
    if (norms.empty() && !ids) return data+first*dimensions;
    buffer.resize(count*dimensions);
    visit([&](auto value) {
      for (size_t i=0;i<count;++i) for (size_t j=0;j<dimensions;++j) buffer[i*dimensions+j]=value(first+i,j);
    });
    return buffer.data();
  }
};
} // namespace tq::tieredq_detail
