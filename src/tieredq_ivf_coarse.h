// IVF coarse quantizer for list counts beyond the codec's centroids: k-means in model space, with assignment and
// query ranking through the encoder's SIMD GEMM. Lists rank by q.c - |c|^2/2, the squared-distance order.
// Centroids are held as binary16 rows of c / scale(c), scale(c) = max |c_k|, so any data range fits binary16;
// assignment, ranking and saved centroids all use these rounded centroids.
// A single query streams its centroid bytes from L3 (#41: 3 MiB of binary16 at 2048 x 768, ~0.12 ms per query), so
// queries first rank lists on the centroids' leading principal axes and rerank only a shortlist exactly; the probe
// order is the exact one whenever the shortlist holds the exact probe set.
#pragma once
#include "tieredq_encode_float.h"
#include "tieredq_io.h"
#include "tieredq_layout.h"
#include "tieredq_train_kmeans.h"
#include "tieredq_train_linalg.h"
#include "training_rng.h"
#include <cmath>
#include <numeric>
#include <omp.h>

namespace dv::tieredq_detail {
struct CoarseQuantizer {
  static constexpr size_t block_rows = 64;
  // Shortlist ranking: lists score on this many leading axes and the shortlist_multiple * nprobe best rerank
  // exactly. On VIBE arxiv (2048 lists, d = 768) the exact probe set left the shortlist for about 1e-4 of the probed
  // lists at nprobe 32-64, always lists ranked near the cut.
  static constexpr size_t shortlist_axes = 128, shortlist_multiple = 3;
  size_t lists = 0, dimensions = 0;
  std::vector<uint16_t> rows;     // c / scale(c) in binary16, row-major
  std::vector<float> scales;      // scale(c)
  std::vector<float> half_norms;  // |c|^2 / 2 of the rounded centroid
  HalfPanels axes;                // dimensions x shortlist_axes leading principal axes; empty: rank exactly
  HalfPanels reduced;             // shortlist_axes x lists: (c - mean of centroids) on the axes, over scale(c)

  // shortlist prepares the query shortlist (a PCA of the centroids); training iterations only assign.
  static CoarseQuantizer from(const std::vector<float>& centroids,size_t lists,size_t dimensions,bool shortlist=true) {
    CoarseQuantizer q; q.lists=lists; q.dimensions=dimensions;
    q.scales.assign(lists,1.0f);
    q.rows.resize(product(lists,dimensions));
    q.half_norms.resize(lists);
    for (size_t c=0;c<lists;++c) {
      const float* centroid=centroids.data()+c*dimensions;
      const float peak=std::fabs(*std::max_element(centroid,centroid+dimensions,[](float a,float b) { return std::fabs(a)<std::fabs(b); }));
      if (peak>0) q.scales[c]=peak;
      double sum=0;
      for (size_t k=0;k<dimensions;++k) {
        q.rows[c*dimensions+k]=half_encode(double(centroid[k])/q.scales[c]);
        const double value=q.centroid(c,k); sum+=value*value;
      }
      q.half_norms[c]=float(sum/2);
    }
    if (shortlist) q.prepare_shortlist();
    return q;
  }
  std::vector<float> centroids() const {
    std::vector<float> out(product(lists,dimensions));
    for (size_t c=0;c<lists;++c) for (size_t k=0;k<dimensions;++k) out[c*dimensions+k]=centroid(c,k);
    return out;
  }
  // out[r * lists + c] = rows[r].c - |c|^2 / 2, parallel over row blocks.
  void closeness(const float* rows,size_t count,float* out) const {
    HalfPanels panels;
    pack(all_lists(),panels);
    closeness(panels,rows,count,out);
  }
  // Nearest list per row; ties keep the lowest list id.
  void assign(const float* rows,size_t count,uint32_t* ids) const {
    HalfPanels panels;
    pack(all_lists(),panels);
    std::vector<float> scores(product(std::min(count,assign_chunk_rows),lists));
    for (size_t first=0;first<count;first+=assign_chunk_rows) {
      const size_t n=std::min(assign_chunk_rows,count-first);
      closeness(panels,rows+first*dimensions,n,scores.data());
      for (size_t r=0;r<n;++r) {
        const float* row=scores.data()+r*lists;
        ids[first+r]=uint32_t(std::max_element(row,row+lists)-row);
      }
    }
  }
  // The nprobe lists with list_rows > 0 of highest closeness to query; ties go to the lower list id.
  std::vector<uint32_t> probe(const float* query,const std::vector<uint32_t>& list_rows,size_t nprobe) const {
    std::vector<uint32_t> order;
    for (uint32_t list=0;list<lists;++list) if (list_rows[list]) order.push_back(list);
    nprobe=std::min(nprobe,order.size());
    const size_t shortlist=shortlist_multiple*nprobe;
    thread_local std::vector<float> scores;
    scores.resize(order.size());
    if (!axes.data.empty() && shortlist<order.size()) {
      thread_local std::vector<float> projected,approximate;
      projected.resize(axes.padded_columns()); approximate.resize(reduced.padded_columns());
      half_matrix(query,1,dimensions,axes,projected.data(),projected.size());
      half_matrix(projected.data(),1,shortlist_axes,reduced,approximate.data(),approximate.size());
      for (size_t i=0;i<order.size();++i) scores[i]=approximate[order[i]]*scales[order[i]]-half_norms[order[i]];
      keep_best(order,scores,shortlist);
    }
    // Exact closeness of the remaining lists, from their rows packed into panels for one GEMM row.
    thread_local HalfPanels panels;
    pack(order,panels);
    thread_local std::vector<float> dots;
    dots.resize(panels.padded_columns());
    half_matrix(query,1,dimensions,panels,dots.data(),dots.size());
    for (size_t i=0;i<order.size();++i) scores[i]=dots[i]*scales[order[i]]-half_norms[order[i]];
    keep_best(order,scores,nprobe);
    return order;
  }
  size_t allocated_bytes() const {
    return sizeof(*this)+(rows.capacity()+axes.data.capacity()+reduced.data.capacity())*sizeof(uint16_t)+
        (scales.capacity()+half_norms.capacity())*sizeof(float);
  }
 private:
  static constexpr size_t assign_chunk_rows = 4096;
  float centroid(size_t c,size_t k) const { return half_decode(rows[c*dimensions+k])*scales[c]; }
  std::vector<uint32_t> all_lists() const {
    std::vector<uint32_t> ids(lists);
    std::iota(ids.begin(),ids.end(),0u);
    return ids;
  }
  // The rows of lists ids as GEMM panels (column i holds list ids[i]), reusing p's storage.
  void pack(const std::vector<uint32_t>& ids,HalfPanels& p) const {
    constexpr size_t PANEL=TieredQPanels::PANEL;
    p.depth=dimensions; p.columns=ids.size();
    p.data.assign(product(dimensions,p.padded_columns()),0);
    for (size_t i=0;i<ids.size();++i) {
      const uint16_t* row=rows.data()+size_t(ids[i])*dimensions;
      uint16_t* column=p.data.data()+i/PANEL*dimensions*PANEL+i%PANEL;
      for (size_t k=0;k<dimensions;++k) column[k*PANEL]=row[k];
    }
  }
  void closeness(const HalfPanels& panels,const float* rows,size_t count,float* out) const {
    const size_t width=panels.padded_columns(),blocks=(count+block_rows-1)/block_rows;
#pragma omp parallel for schedule(dynamic) if(blocks>1 && !omp_in_parallel())
    for (size_t block=0;block<blocks;++block) {
      const size_t first=block*block_rows,n=std::min(block_rows,count-first);
      std::vector<float> dots(n*width);
      half_matrix(rows+first*dimensions,n,dimensions,panels,dots.data(),width);
      for (size_t r=0;r<n;++r) for (size_t c=0;c<lists;++c) out[(first+r)*lists+c]=dots[r*width+c]*scales[c]-half_norms[c];
    }
  }
  // Keeps the count lists of highest score (score[i] belongs to ids[i]; ties to the lower id), best first.
  static void keep_best(std::vector<uint32_t>& ids,std::vector<float>& score,size_t count) {
    std::vector<uint32_t> position(ids.size());
    std::iota(position.begin(),position.end(),0u);
    std::partial_sort(position.begin(),position.begin()+count,position.end(),[&](uint32_t a,uint32_t b) {
      return score[a]>score[b] || (score[a]==score[b] && ids[a]<ids[b]);
    });
    std::vector<uint32_t> kept(count);
    for (size_t i=0;i<count;++i) kept[i]=ids[position[i]];
    ids=std::move(kept);
    score.resize(count);
  }
  // Leading principal axes of the rounded centroids and the centroids' coordinates on them.
  void prepare_shortlist() {
    if (dimensions<=shortlist_axes || lists<=shortlist_axes) return;
    const int d=int(dimensions);
    std::vector<double> mean(dimensions,0.0),centered(product(lists,dimensions));
    for (size_t c=0;c<lists;++c) for (size_t k=0;k<dimensions;++k) mean[k]+=centroid(c,k);
    for (auto& value:mean) value/=double(lists);
    for (size_t c=0;c<lists;++c) for (size_t k=0;k<dimensions;++k) centered[c*dimensions+k]=centroid(c,k)-mean[k];
    const auto& linalg=TrainingLinalg::get();
    auto vectors=linalg.covariance_centered(centered,d,int(lists));
    std::vector<double> eigenvalues(dimensions);
    const char lower='L',want_vectors='V';
    int lwork=-1,liwork=-1,info=0,iwork_query=0; double work_query=0;
    linalg.eigen(&want_vectors,&lower,&d,vectors.data(),&d,eigenvalues.data(),&work_query,&lwork,&iwork_query,&liwork,&info);
    if (info) bad("IVF coarse PCA workspace failed");
    lwork=int(work_query); liwork=iwork_query;
    std::vector<double> work(static_cast<size_t>(lwork)); std::vector<int> iwork(static_cast<size_t>(liwork));
    linalg.eigen(&want_vectors,&lower,&d,vectors.data(),&d,eigenvalues.data(),work.data(),&lwork,iwork.data(),&liwork,&info);
    if (info) bad("IVF coarse PCA failed");
    // Eigenvalues ascend; axis j is the j-th largest, column d - 1 - j of the column-major eigenvectors.
    auto axis=[&](size_t k,size_t j) { return vectors[(dimensions-1-j)*dimensions+k]; };
    axes.depth=dimensions; axes.columns=shortlist_axes;
    axes.data=pack_panel_values<uint16_t>(dimensions,shortlist_axes,[&](size_t k,size_t j) { return half_encode(axis(k,j)); });
    std::vector<double> coordinates(product(lists,shortlist_axes),0.0);
#pragma omp parallel for schedule(static)
    for (size_t c=0;c<lists;++c) for (size_t j=0;j<shortlist_axes;++j) {
      double sum=0;
      for (size_t k=0;k<dimensions;++k) sum+=centered[c*dimensions+k]*axis(k,j);
      coordinates[c*shortlist_axes+j]=sum/scales[c];
    }
    reduced.depth=shortlist_axes; reduced.columns=lists;
    reduced.data=pack_panel_values<uint16_t>(shortlist_axes,lists,[&](size_t j,size_t c) {
      return half_encode(coordinates[c*shortlist_axes+j]);
    });
  }
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
    CoarseQuantizer::from(centroids,lists,dimensions,false).assign(sample,rows,ids.data());
    update_centroids(sample,rows,dimensions,ids.data(),centroids);
  }
  return CoarseQuantizer::from(centroids,lists,dimensions);
}
} // namespace dv::tieredq_detail
