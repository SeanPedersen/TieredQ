// Restricted coarse assignment (#48): each row's nearest list among the members of its probes nearest supers (k-means
// of the lists' centroids), instead of among every list. Rows probing the same super are scored together against that
// super's members through the coarse binary16 GEMM, so the work is about probes / supers of exact assignment while
// staying GEMM-shaped. Members keep the coarse quantizer's rounded centroids, scales and norms, and every GEMM column
// accumulates independently, so a row whose nearest list lies in a probed super gets exactly the exact label.
#pragma once
#include "tieredq/tieredq.h"
#include "tieredq_encode_float.h"
#include "tieredq_ivf_coarse.h"
#include <limits>
#include <numeric>

namespace dv::tieredq_detail {
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

// Labels rows with the coarse quantizer's nearest list: exactly, or restricted when training sets assignment supers.
inline void assign_lists(const CoarseQuantizer& coarse,const TieredQCoarseTraining& training,uint64_t seed,
                         const float* rows,size_t count,uint32_t* ids) {
  if (!training.assignment_supers) return coarse.assign(rows,count,ids);
  RestrictedAssigner(coarse,training.assignment_supers,training.assignment_probes,seed).assign(rows,count,ids);
}

inline CoarseQuantizer train_coarse(const float* sample,size_t rows,size_t dimensions,size_t lists,uint64_t seed,
                                    const TieredQCoarseTraining& training) {
  return train_coarse(sample,rows,dimensions,lists,seed,training.iterations,
                      [&](const CoarseQuantizer& q,const float* x,size_t n,uint32_t* ids) { assign_lists(q,training,seed,x,n,ids); });
}
} // namespace dv::tieredq_detail
