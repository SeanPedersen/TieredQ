// IVF coarse quantizer for list counts beyond the codec's centroids: k-means in model space, with assignment and
// query ranking through the encoder's SIMD GEMM. Lists rank by q.c - |c|^2/2, the squared-distance order.
#pragma once
#include "tieredq_encode_float.h"
#include "tieredq_io.h"
#include "tieredq_train_kmeans.h"
#include "training_rng.h"
#include <cmath>

namespace dv::tieredq_detail {
struct CoarseQuantizer {
  static constexpr size_t block_rows = 64;
  size_t lists = 0, dimensions = 0;
  TieredQPanels panels;           // the centroids, held only as GEMM panels
  std::vector<float> half_norms;  // |c|^2 / 2

  static CoarseQuantizer from(const std::vector<float>& centroids,size_t lists,size_t dimensions) {
    CoarseQuantizer q; q.lists=lists; q.dimensions=dimensions;
    q.panels=pack_panels(dimensions,lists,[&](size_t k,size_t c) { return centroids[c*dimensions+k]; });
    q.half_norms.resize(lists);
    for (size_t c=0;c<lists;++c) {
      double sum=0;
      for (size_t k=0;k<dimensions;++k) sum+=double(centroids[c*dimensions+k])*centroids[c*dimensions+k];
      q.half_norms[c]=float(sum/2);
    }
    return q;
  }
  std::vector<float> centroids() const {
    std::vector<float> out(product(lists,dimensions));
    for (size_t c=0;c<lists;++c) for (size_t k=0;k<dimensions;++k)
      out[c*dimensions+k]=panels.data[(c/TieredQPanels::PANEL*dimensions+k)*TieredQPanels::PANEL+c%TieredQPanels::PANEL];
    return out;
  }
  // out[r * lists + c] = rows[r].c - |c|^2 / 2, parallel over row blocks.
  void closeness(const float* rows,size_t count,float* out) const {
    const size_t width=panels.padded_columns(),blocks=(count+block_rows-1)/block_rows;
#pragma omp parallel for schedule(dynamic) if(blocks>1 && !omp_in_parallel())
    for (size_t block=0;block<blocks;++block) {
      const size_t first=block*block_rows,n=std::min(block_rows,count-first);
      std::vector<float> dots(n*width);
      float_matrix(rows+first*dimensions,n,dimensions,panels,dots.data(),width);
      for (size_t r=0;r<n;++r) for (size_t c=0;c<lists;++c) out[(first+r)*lists+c]=dots[r*width+c]-half_norms[c];
    }
  }
  // Nearest list per row; ties keep the lowest list id.
  void assign(const float* rows,size_t count,uint32_t* ids) const {
    std::vector<float> scores(product(std::min(count,assign_chunk_rows),lists));
    for (size_t first=0;first<count;first+=assign_chunk_rows) {
      const size_t n=std::min(assign_chunk_rows,count-first);
      closeness(rows+first*dimensions,n,scores.data());
      for (size_t r=0;r<n;++r) {
        const float* row=scores.data()+r*lists;
        ids[first+r]=uint32_t(std::max_element(row,row+lists)-row);
      }
    }
  }
  size_t allocated_bytes() const { return sizeof(*this)+panels.data.capacity()*sizeof(float)+half_norms.capacity()*sizeof(float); }
 private:
  static constexpr size_t assign_chunk_rows = 4096;
};

// Lloyd k-means from lists distinct seeded sample rows, as the codec's centroid training.
inline CoarseQuantizer train_coarse(const float* sample,size_t rows,size_t dimensions,size_t lists,uint64_t seed) {
  constexpr int iterations=20;
  if (lists<1 || rows<lists) bad("IVF coarse training needs at least one row per list");
  TrainingRng rng(seed);
  const auto initial=rng.choice(rows,lists);
  std::vector<float> centroids(product(lists,dimensions));
  for (size_t c=0;c<lists;++c) std::copy_n(sample+initial[c]*dimensions,dimensions,centroids.data()+c*dimensions);
  std::vector<uint32_t> ids(rows);
  for (int it=0;it<iterations;++it) {
    CoarseQuantizer::from(centroids,lists,dimensions).assign(sample,rows,ids.data());
    update_centroids(sample,rows,dimensions,ids.data(),centroids);
  }
  return CoarseQuantizer::from(centroids,lists,dimensions);
}
} // namespace dv::tieredq_detail
