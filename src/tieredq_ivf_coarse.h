// IVF coarse quantizer for list counts beyond the codec's centroids: k-means in the coarse space (model space, or input
// space for L2 indexes), with assignment and query ranking through the encoder's SIMD GEMM. Lists rank by q.c - |c|^2/2,
// the squared-distance order.
// Centroids are held as binary16 panels of c / scale(c), scale(c) = max |c_k|: a single query streams every
// centroid from L3 (#41: 6 MiB of FP32 at 2048 x 768 was ~0.2 ms per query), and the per-centroid scale keeps any
// data range inside binary16. Assignment, ranking and saved centroids all use these rounded centroids.
#pragma once
#include "tieredq_encode_float.h"
#include "tieredq_io.h"
#include "tieredq_layout.h"
#include "tieredq_train_kmeans.h"
#include "training_rng.h"
#include <cmath>
#include <omp.h>

namespace tq::tieredq_detail {
struct CoarseQuantizer {
  static constexpr size_t block_rows = 64;
  size_t lists = 0, dimensions = 0;
  HalfPanels panels;              // the centroids over their scale, held only as GEMM panels
  std::vector<float> scales;      // scale(c)
  std::vector<float> half_norms;  // |c|^2 / 2 of the rounded centroid

  static CoarseQuantizer from(const std::vector<float>& centroids,size_t lists,size_t dimensions) {
    CoarseQuantizer q; q.lists=lists; q.dimensions=dimensions;
    q.scales.assign(lists,1.0f);
    for (size_t c=0;c<lists;++c) {
      const float* centroid=centroids.data()+c*dimensions;
      const float peak=std::fabs(*std::max_element(centroid,centroid+dimensions,[](float a,float b) { return std::fabs(a)<std::fabs(b); }));
      if (peak>0) q.scales[c]=peak;
    }
    q.panels.depth=dimensions; q.panels.columns=lists;
    q.panels.data=pack_panel_values<uint16_t>(dimensions,lists,[&](size_t k,size_t c) {
      return half_encode(double(centroids[c*dimensions+k])/q.scales[c]);
    });
    q.half_norms.resize(lists);
    for (size_t c=0;c<lists;++c) {
      double sum=0;
      for (size_t k=0;k<dimensions;++k) { const double value=q.centroid(c,k); sum+=value*value; }
      q.half_norms[c]=float(sum/2);
    }
    return q;
  }
  std::vector<float> centroids() const {
    std::vector<float> out(product(lists,dimensions));
    for (size_t c=0;c<lists;++c) for (size_t k=0;k<dimensions;++k) out[c*dimensions+k]=centroid(c,k);
    return out;
  }
  // out[r * lists + c] = rows[r].c - |c|^2 / 2, parallel over row blocks.
  void closeness(const float* rows,size_t count,float* out) const {
    const size_t width=panels.padded_columns(),blocks=(count+block_rows-1)/block_rows;
#pragma omp parallel for schedule(dynamic) if(blocks>1 && !omp_in_parallel())
    for (size_t block=0;block<blocks;++block) {
      const size_t first=block*block_rows,n=std::min(block_rows,count-first);
      std::vector<float> dots(n*width);
      half_matrix(rows+first*dimensions,n,dimensions,panels,dots.data(),width);
      for (size_t r=0;r<n;++r) for (size_t c=0;c<lists;++c) out[(first+r)*lists+c]=dots[r*width+c]*scales[c]-half_norms[c];
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
  // Centroid c's binary16 values over its scale, as the panels hold them (d values, row-major).
  void half_row(size_t c,uint16_t* out) const {
    constexpr size_t PANEL=TieredQPanels::PANEL;
    const uint16_t* panel=panels.data.data()+c/PANEL*dimensions*PANEL+c%PANEL;
    for (size_t k=0;k<dimensions;++k) out[k]=panel[k*PANEL];
  }
  size_t allocated_bytes() const {
    return sizeof(*this)+panels.data.capacity()*sizeof(uint16_t)+(scales.capacity()+half_norms.capacity())*sizeof(float);
  }
 private:
  static constexpr size_t assign_chunk_rows = 4096;
  float centroid(size_t c,size_t k) const {
    constexpr size_t PANEL=TieredQPanels::PANEL;
    return half_decode(panels.data[(c/PANEL*dimensions+k)*PANEL+c%PANEL])*scales[c];
  }
};

// Lloyd k-means from lists distinct seeded sample rows, as the codec's centroid training; assign(quantizer, rows,
// count, ids) labels the sample each iteration (exact nearest list by default).
constexpr size_t default_coarse_iterations=20;
template<class Assign>
CoarseQuantizer train_coarse(const float* sample,size_t rows,size_t dimensions,size_t lists,uint64_t seed,
                             size_t iterations,Assign&& assign) {
  if (lists<1 || rows<lists) bad("IVF coarse training needs at least one row per list");
  TrainingRng rng(seed);
  const auto initial=rng.choice(rows,lists);
  std::vector<float> centroids(product(lists,dimensions));
  for (size_t c=0;c<lists;++c) std::copy_n(sample+initial[c]*dimensions,dimensions,centroids.data()+c*dimensions);
  std::vector<uint32_t> ids(rows);
  for (size_t it=0;it<iterations;++it) {
    assign(CoarseQuantizer::from(centroids,lists,dimensions),sample,rows,ids.data());
    update_centroids(sample,rows,dimensions,ids.data(),centroids);
  }
  return CoarseQuantizer::from(centroids,lists,dimensions);
}
inline CoarseQuantizer train_coarse(const float* sample,size_t rows,size_t dimensions,size_t lists,uint64_t seed,
                                    size_t iterations=default_coarse_iterations) {
  return train_coarse(sample,rows,dimensions,lists,seed,iterations,
                      [](const CoarseQuantizer& q,const float* x,size_t n,uint32_t* ids) { q.assign(x,n,ids); });
}
} // namespace tq::tieredq_detail
