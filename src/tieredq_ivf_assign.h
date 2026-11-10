// Coarse assignment at build time (#48). Exact assignment runs as FP32 BLAS sgemm on the rounded centroids, the way
// Faiss assigns (1.3-1.5x the binary16 panel kernel's single-thread throughput on AVX2 and AVX-512); the panels stay for
// query ranking, which streams the centroids once per query. Restricted assignment: each row's nearest list among the members of its probes nearest supers (k-means
// of the lists' centroids), instead of among every list. Rows probing the same super are scored together against that
// super's members through the coarse binary16 GEMM, so the work is about probes / supers of exact assignment while
// staying GEMM-shaped. Members keep the coarse quantizer's rounded centroids, scales and norms, so a row whose nearest
// list lies in a probed super gets the exact label up to floating-point near-ties (the panel kernel and BLAS sum in
// different orders).
#pragma once
#include "tieredq/tieredq.h"
#include "tieredq_encode_float.h"
#include "tieredq_ivf_coarse.h"
#include "tieredq_train_linalg.h"
#include <limits>
#include <numeric>
#include <optional>
#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace dv::tieredq_detail {
// OpenBLAS sizes its own thread pool; within a call it uses the OpenMP thread count, so builds keep their thread budget.
class BlasThreads {
 public:
  BlasThreads() {
#if defined(__unix__) || defined(__APPLE__)
    set_=reinterpret_cast<void(*)(int)>(dlsym(RTLD_DEFAULT,"openblas_set_num_threads"));
    auto get=reinterpret_cast<int(*)()>(dlsym(RTLD_DEFAULT,"openblas_get_num_threads"));
    if (!set_ || !get) { set_=nullptr; return; }
    previous_=get();
    set_(omp_get_max_threads());
#endif
  }
  ~BlasThreads() { if (set_) set_(previous_); }
  BlasThreads(const BlasThreads&)=delete;
  BlasThreads& operator=(const BlasThreads&)=delete;
 private:
  void (*set_)(int)=nullptr;
  int previous_=1;
};

// Exact nearest list per row (ties: lowest list id) by FP32 sgemm over row chunks and centroid tiles.
class BlasAssigner {
 public:
  explicit BlasAssigner(const CoarseQuantizer& coarse) : coarse_(coarse),centroids_(coarse.centroids()) {}

  void assign(const float* rows,size_t count,uint32_t* ids) const {
    const BlasThreads threads;
    const auto& linalg=TrainingLinalg::get();
    const int d=int(coarse_.dimensions);
    const size_t tile=std::min(tile_lists,coarse_.lists);
    std::vector<float> dots(product(std::min(chunk_rows,count),tile)),best(std::min(chunk_rows,count));
    for (size_t first=0;first<count;first+=chunk_rows) {
      const size_t n=std::min(chunk_rows,count-first);
      std::fill_n(best.begin(),n,-std::numeric_limits<float>::infinity());
      for (size_t list0=0;list0<coarse_.lists;list0+=tile) {
        const size_t m=std::min(tile,coarse_.lists-list0);
        // Column-major: dots (m x n) = centroids[list0..)^T x rows^T, so each row's scores are contiguous.
        const int mi=int(m),ni=int(n); const float one=1,zero=0; const char transposed='T',normal='N';
        linalg.gemm(&transposed,&normal,&mi,&ni,&d,&one,centroids_.data()+list0*coarse_.dimensions,&d,
                    rows+first*coarse_.dimensions,&d,&zero,dots.data(),&mi);
#pragma omp parallel for schedule(static) if(n>1 && !omp_in_parallel())
        for (size_t r=0;r<n;++r) {
          const float* score=dots.data()+r*m;
          for (size_t j=0;j<m;++j) {
            const float closeness=score[j]-coarse_.half_norms[list0+j];
            if (closeness>best[r]) { best[r]=closeness; ids[first+r]=uint32_t(list0+j); }
          }
        }
      }
    }
  }

 private:
  static constexpr size_t chunk_rows=4096,tile_lists=8192;
  const CoarseQuantizer& coarse_;
  std::vector<float> centroids_;  // the rounded centroids, row-major
};

class RestrictedAssigner {
 public:
  RestrictedAssigner(const CoarseQuantizer& coarse,size_t supers,size_t probes,uint64_t seed)
      : coarse_(coarse),supers_(cluster(coarse,std::min(supers,coarse.lists),seed)),
        probes_(std::clamp<size_t>(probes,1,supers_.lists)) {
    const size_t d=coarse.dimensions;
    std::vector<uint32_t> owner(coarse.lists);
    supers_.assign(coarse.centroids().data(),coarse.lists,owner.data());
    members_.assign(supers_.lists,{});
    for (uint32_t c=0;c<coarse.lists;++c) members_[owner[c]].push_back(c);
    panels_.resize(supers_.lists);
    for (size_t s=0;s<supers_.lists;++s) {
      const auto& members=members_[s];
      std::vector<uint16_t> halves(product(members.size(),d));
      for (size_t m=0;m<members.size();++m) coarse.half_row(members[m],halves.data()+m*d);
      panels_[s].depth=d; panels_[s].columns=members.size();
      panels_[s].data=pack_panel_values<uint16_t>(d,members.size(),[&](size_t k,size_t m) { return halves[m*d+k]; });
    }
  }

  // Nearest list per row among the probed supers' members; ties keep the lowest list id, as exact assignment does.
  void assign(const float* rows,size_t count,uint32_t* ids) const {
    std::vector<float> super_closeness,best_closeness;
    std::vector<uint32_t> probed,slot_order,best_list;
    for (size_t first=0;first<count;first+=chunk_rows) {
      const size_t n=std::min(chunk_rows,count-first);
      const float* chunk=rows+first*coarse_.dimensions;
      probe_supers(chunk,n,super_closeness,probed);
      // Slots (row * probes + p) grouped by super: a counting sort, then work items of whole GEMM blocks per super.
      std::vector<size_t> group_begin(supers_.lists+1,0);
      for (size_t slot=0;slot<n*probes_;++slot) ++group_begin[probed[slot]+1];
      std::partial_sum(group_begin.begin(),group_begin.end(),group_begin.begin());
      slot_order.resize(n*probes_);
      {
        auto fill=group_begin;
        for (size_t slot=0;slot<n*probes_;++slot) slot_order[fill[probed[slot]]++]=uint32_t(slot);
      }
      struct Work { uint32_t super; size_t first,count; };
      std::vector<Work> work;
      for (uint32_t s=0;s<supers_.lists;++s)
        for (size_t at=group_begin[s];at<group_begin[s+1];at+=CoarseQuantizer::block_rows)
          work.push_back({s,at,std::min(CoarseQuantizer::block_rows,group_begin[s+1]-at)});
      best_closeness.assign(n*probes_,-std::numeric_limits<float>::infinity());
      best_list.assign(n*probes_,std::numeric_limits<uint32_t>::max());
#pragma omp parallel for schedule(dynamic) if(work.size()>1 && !omp_in_parallel())
      for (size_t item=0;item<work.size();++item) score_block(chunk,work[item].super,slot_order.data()+work[item].first,
                                                              work[item].count,best_closeness,best_list);
      for (size_t row=0;row<n;++row) {
        float best=-std::numeric_limits<float>::infinity();
        uint32_t label=std::numeric_limits<uint32_t>::max();
        for (size_t p=0;p<probes_;++p) {
          const size_t slot=row*probes_+p;
          if (best_closeness[slot]>best || (best_closeness[slot]==best && best_list[slot]<label)) {
            best=best_closeness[slot]; label=best_list[slot];
          }
        }
        // Only empty probed supers: fall back to the exact label.
        if (label==std::numeric_limits<uint32_t>::max()) coarse_.assign(chunk+row*coarse_.dimensions,1,&label);
        ids[first+row]=label;
      }
    }
  }

 private:
  static constexpr size_t chunk_rows=32768,super_iterations=6;

  static CoarseQuantizer cluster(const CoarseQuantizer& coarse,size_t supers,uint64_t seed) {
    const auto centroids=coarse.centroids();
    return train_coarse(centroids.data(),coarse.lists,coarse.dimensions,supers,seed,super_iterations);
  }

  // probed[row * probes + p]: the row's p-th nearest super.
  void probe_supers(const float* chunk,size_t n,std::vector<float>& closeness,std::vector<uint32_t>& probed) const {
    closeness.resize(product(n,supers_.lists));
    supers_.closeness(chunk,n,closeness.data());
    probed.resize(n*probes_);
#pragma omp parallel for schedule(static) if(n>1 && !omp_in_parallel())
    for (size_t row=0;row<n;++row) {
      const float* score=closeness.data()+row*supers_.lists;
      std::vector<uint32_t> order(supers_.lists);
      std::iota(order.begin(),order.end(),0u);
      std::partial_sort(order.begin(),order.begin()+probes_,order.end(),[&](uint32_t a,uint32_t b) {
        return score[a]>score[b] || (score[a]==score[b] && a<b);
      });
      std::copy_n(order.begin(),probes_,probed.begin()+row*probes_);
    }
  }

  // The best member of super s for each of count slots.
  void score_block(const float* chunk,uint32_t s,const uint32_t* slots,size_t count,std::vector<float>& best_closeness,
                   std::vector<uint32_t>& best_list) const {
    const auto& members=members_[s];
    if (members.empty()) return;
    const size_t d=coarse_.dimensions,width=panels_[s].padded_columns();
    std::vector<float> gathered(product(count,d)),dots(product(count,width));
    for (size_t i=0;i<count;++i) std::copy_n(chunk+size_t(slots[i]/probes_)*d,d,gathered.data()+i*d);
    half_matrix(gathered.data(),count,d,panels_[s],dots.data(),width);
    for (size_t i=0;i<count;++i) {
      float best=-std::numeric_limits<float>::infinity();
      uint32_t label=std::numeric_limits<uint32_t>::max();
      for (size_t m=0;m<members.size();++m) {
        const uint32_t list=members[m];
        const float value=dots[i*width+m]*coarse_.scales[list]-coarse_.half_norms[list];
        if (value>best || (value==best && list<label)) { best=value; label=list; }
      }
      best_closeness[slots[i]]=best; best_list[slots[i]]=label;
    }
  }

  const CoarseQuantizer& coarse_;
  CoarseQuantizer supers_;
  std::vector<std::vector<uint32_t>> members_;  // list ids per super, ascending
  std::vector<HalfPanels> panels_;              // per super: its members' rounded centroids
  size_t probes_;
};

// Labels rows with the coarse quantizer's nearest list: exactly (BLAS), or restricted when training sets supers.
class ListAssigner {
 public:
  ListAssigner(const CoarseQuantizer& coarse,const TieredQCoarseTraining& training,uint64_t seed) {
    if (training.assignment_supers) restricted_.emplace(coarse,training.assignment_supers,training.assignment_probes,seed);
    else exact_.emplace(coarse);
  }
  void assign(const float* rows,size_t count,uint32_t* ids) const {
    if (restricted_) restricted_->assign(rows,count,ids);
    else exact_->assign(rows,count,ids);
  }
 private:
  std::optional<RestrictedAssigner> restricted_;
  std::optional<BlasAssigner> exact_;
};

inline CoarseQuantizer train_coarse(const float* sample,size_t rows,size_t dimensions,size_t lists,uint64_t seed,
                                    const TieredQCoarseTraining& training) {
  return train_coarse(sample,rows,dimensions,lists,seed,training.iterations,
                      [&](const CoarseQuantizer& q,const float* x,size_t n,uint32_t* ids) {
                        ListAssigner(q,training,seed).assign(x,n,ids);
                      });
}
} // namespace dv::tieredq_detail
