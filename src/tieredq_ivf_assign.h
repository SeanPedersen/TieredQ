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
#include "tieredq_ivf_bound.h"
#include "tieredq_ivf_coarse.h"
#include "tieredq_train_linalg.h"
#include <cstdlib>
#include <limits>
#include <numeric>
#include <optional>
#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace tq::tieredq_detail {
// Thread cap for the calling thread's multi-threaded BLAS calls, set per index call from its config (0: unset).
inline int& blas_thread_cap() {
  static thread_local int cap=0;
  return cap;
}
class BlasThreadCap {
 public:
  explicit BlasThreadCap(int cap): previous_(blas_thread_cap()) { if (cap>0) blas_thread_cap()=cap; }
  ~BlasThreadCap() { blas_thread_cap()=previous_; }
  BlasThreadCap(const BlasThreadCap&)=delete;
  BlasThreadCap& operator=(const BlasThreadCap&)=delete;
 private:
  int previous_;
};

// Thread count for BLAS calls made outside an OpenMP region: the config cap, else TIEREDQ_BLAS_THREADS, else
// OPENBLAS_NUM_THREADS, else the OpenMP thread count (so builds keep their thread budget).
inline int default_blas_threads() {
  if (blas_thread_cap()>0) return blas_thread_cap();
  for (const char* name : {"TIEREDQ_BLAS_THREADS","OPENBLAS_NUM_THREADS"}) {
    const char* value=std::getenv(name);
    const long parsed=value ? std::strtol(value,nullptr,10) : 0;
    if (parsed>0) return int(std::min<long>(parsed,std::numeric_limits<int>::max()));
  }
  return omp_get_max_threads();
}

// OpenBLAS sizes its own thread pool; BlasThreads sets it for one scope. BlasThreads(1) keeps each call on its
// caller's thread, for GEMMs issued from inside an OpenMP parallel region.
class BlasThreads {
 public:
  explicit BlasThreads(int threads=default_blas_threads()) {
#if defined(__unix__) || defined(__APPLE__)
    set_=reinterpret_cast<void(*)(int)>(dlsym(RTLD_DEFAULT,"openblas_set_num_threads"));
    auto get=reinterpret_cast<int(*)()>(dlsym(RTLD_DEFAULT,"openblas_get_num_threads"));
    if (!set_ || !get) { set_=nullptr; return; }
    previous_=get();
    set_(threads);
#endif
  }
  ~BlasThreads() { if (set_) set_(previous_); }
  BlasThreads(const BlasThreads&)=delete;
  BlasThreads& operator=(const BlasThreads&)=delete;
 private:
  void (*set_)(int)=nullptr;
  int previous_=1;
};

// Exact nearest centroid per row (ties: lowest id) by FP32 sgemm over row chunks and centroid tiles: centroids
// (lists x d, row-major) rank by row.c - half_norms[c], half_norms[c] = |c|^2 / 2.
inline void nearest_centroids(const float* centroids,const float* half_norms,size_t lists,size_t dimensions,
                              const float* rows,size_t count,uint32_t* ids) {
  constexpr size_t chunk_rows=4096,tile_lists=8192;
  const BlasThreads threads;
  const auto& linalg=TrainingLinalg::get();
  const int d=int(dimensions);
  const size_t tile=std::min(tile_lists,lists);
  std::vector<float> dots(product(std::min(chunk_rows,count),tile)),best(std::min(chunk_rows,count));
  for (size_t first=0;first<count;first+=chunk_rows) {
    const size_t n=std::min(chunk_rows,count-first);
    std::fill_n(best.begin(),n,-std::numeric_limits<float>::infinity());
    for (size_t list0=0;list0<lists;list0+=tile) {
      const size_t m=std::min(tile,lists-list0);
      // Column-major: dots (m x n) = centroids[list0..)^T x rows^T, so each row's scores are contiguous.
      const int mi=int(m),ni=int(n); const float one=1,zero=0; const char transposed='T',normal='N';
      linalg.gemm(&transposed,&normal,&mi,&ni,&d,&one,centroids+list0*dimensions,&d,rows+first*dimensions,&d,&zero,
                  dots.data(),&mi);
#pragma omp parallel for schedule(static) if(n>1 && !omp_in_parallel())
      for (size_t r=0;r<n;++r) {
        const float* score=dots.data()+r*m;
        for (size_t j=0;j<m;++j) {
          const float closeness=score[j]-half_norms[list0+j];
          if (closeness>best[r]) { best[r]=closeness; ids[first+r]=uint32_t(list0+j); }
        }
      }
    }
  }
}

// nearest_centroids() on the coarse quantizer's rounded centroids.
class BlasAssigner {
 public:
  explicit BlasAssigner(const CoarseQuantizer& coarse) : coarse_(coarse),centroids_(coarse.centroids()) {}

  void assign(const float* rows,size_t count,uint32_t* ids) const {
    nearest_centroids(centroids_.data(),coarse_.half_norms.data(),coarse_.lists,coarse_.dimensions,rows,count,ids);
  }

 private:
  const CoarseQuantizer& coarse_;
  std::vector<float> centroids_;  // the rounded centroids, row-major
};

// Restricted assignment: each row's nearest list among the members of its probes nearest supers (a k-means of the
// lists' centroids). With bound_dimensions > 0 (tieredq_ivf_bound.h) the first probed super is scored in full, giving
// each row a distance to beat, and the other probes' members only where their exact lower bound is below it.
class RestrictedAssigner {
 public:
  RestrictedAssigner(const CoarseQuantizer& coarse,size_t supers,size_t probes,uint64_t seed,size_t bound_dimensions=0)
      : coarse_(coarse),supers_(cluster(coarse,std::min(supers,coarse.lists),seed)),
        probes_(std::clamp<size_t>(probes,1,supers_.lists)) {
    const size_t d=coarse.dimensions;
    std::vector<uint32_t> owner(coarse.lists);
    supers_.assign(coarse.centroids().data(),coarse.lists,owner.data());
    members_.assign(supers_.lists,{});
    for (uint32_t c=0;c<coarse.lists;++c) members_[owner[c]].push_back(c);
    // A bound as wide as the rows costs more than it saves, and one probe leaves nothing to prune.
    if (bound_dimensions && bound_dimensions<d && probes_>1) {
      bound_.emplace(coarse,bound_dimensions);
      bound_panels_.reserve(supers_.lists);
      member_bound_terms_.reserve(supers_.lists);
      for (const auto& members:members_) {
        bound_panels_.push_back(bound_->panels(members));
        member_bound_terms_.push_back(bound_terms(members));
      }
      bound_->release_projections();
      member_halves_.resize(supers_.lists);
    }
    panels_.resize(supers_.lists);
    for (size_t s=0;s<supers_.lists;++s) {
      const auto& members=members_[s];
      std::vector<uint16_t> halves(product(members.size(),d));
      for (size_t m=0;m<members.size();++m) coarse.half_row(members[m],halves.data()+m*d);
      panels_[s].depth=d; panels_[s].columns=members.size();
      panels_[s].data=pack_panel_values<uint16_t>(d,members.size(),[&](size_t k,size_t m) { return halves[m*d+k]; });
      if (bound_) member_halves_[s]=std::move(halves);
    }
  }

  // Nearest list per row among the probed supers' members; ties keep the lowest list id, as exact assignment does.
  void assign(const float* rows,size_t count,uint32_t* ids) const {
    ChunkState state;
    for (size_t first=0;first<count;first+=chunk_rows) {
      const size_t n=std::min(chunk_rows,count-first);
      const float* chunk=rows+first*coarse_.dimensions;
      probe_supers(chunk,n,state.closeness,state.probed);
      state.best.assign(n*probes_,-std::numeric_limits<float>::infinity());
      state.best_list.assign(n*probes_,std::numeric_limits<uint32_t>::max());
      if (bound_) assign_bounded(chunk,n,state);
      else score_slots(chunk,n,state,0,probes_);
#pragma omp parallel for schedule(static) if(n>1 && !omp_in_parallel())
      for (size_t row=0;row<n;++row) ids[first+row]=best_of_probes(state,row);
      // Only empty probed supers leave no label; those rows fall back to the exact one.
      for (size_t row=0;row<n;++row)
        if (ids[first+row]==std::numeric_limits<uint32_t>::max()) coarse_.assign(chunk+row*coarse_.dimensions,1,ids+first+row);
    }
  }

 private:
  static constexpr size_t chunk_rows=32768,super_iterations=6;
  // Relative slack on the bound for FP32 rounding of both expansions: pruning keeps exact labels up to near-ties.
  static constexpr float bound_slack=1e-5f;

  struct ChunkState {
    std::vector<float> closeness,best,projected,projected_sq,residual,limit;
    std::vector<uint32_t> probed,best_list,slots;
  };
  struct Work { uint32_t super; size_t first,count; };
  // A super's members' |Pc|^2 and |r_c| in member order, contiguous for the vectorized bound.
  struct BoundTerms { std::vector<float> projected_sq,residual; };

  BoundTerms bound_terms(const std::vector<uint32_t>& members) const {
    BoundTerms terms{std::vector<float>(members.size()),std::vector<float>(members.size())};
    for (size_t m=0;m<members.size();++m) {
      terms.projected_sq[m]=bound_->projected_sq(members[m]); terms.residual[m]=bound_->residual(members[m]);
    }
    return terms;
  }

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

  // The slots (row * probes + p) of probes [probe_begin, probe_end) grouped by super, a counting sort, as work items of
  // GEMM blocks per super. It runs on one thread while the others wait, so it walks rows and probes without a division.
  std::vector<Work> group_slots(size_t n,ChunkState& state,size_t probe_begin,size_t probe_end) const {
    std::vector<size_t> begin(supers_.lists+1,0);
    for (size_t row=0;row<n;++row)
      for (size_t p=probe_begin;p<probe_end;++p) ++begin[state.probed[row*probes_+p]+1];
    std::partial_sum(begin.begin(),begin.end(),begin.begin());
    state.slots.resize(begin[supers_.lists]);
    auto fill=begin;
    for (size_t row=0;row<n;++row)
      for (size_t p=probe_begin;p<probe_end;++p) {
        const size_t slot=row*probes_+p;
        state.slots[fill[state.probed[slot]]++]=uint32_t(slot);
      }
    std::vector<Work> work;
    for (uint32_t s=0;s<supers_.lists;++s)
      for (size_t at=begin[s];at<begin[s+1];at+=CoarseQuantizer::block_rows)
        work.push_back({s,at,std::min(CoarseQuantizer::block_rows,begin[s+1]-at)});
    return work;
  }

  void score_slots(const float* chunk,size_t n,ChunkState& state,size_t probe_begin,size_t probe_end) const {
    const auto work=group_slots(n,state,probe_begin,probe_end);
#pragma omp parallel for schedule(dynamic) if(work.size()>1 && !omp_in_parallel())
    for (size_t item=0;item<work.size();++item)
      score_block(chunk,work[item].super,state.slots.data()+work[item].first,work[item].count,state.best,state.best_list);
  }

  void assign_bounded(const float* chunk,size_t n,ChunkState& state) const {
    const size_t d=coarse_.dimensions,dp=bound_->dimensions();
    state.projected.resize(product(n,dp)); state.projected_sq.resize(n); state.residual.resize(n); state.limit.resize(n);
    { const BlasThreads threads; bound_->project(chunk,n,state.projected.data()); }
    score_slots(chunk,n,state,0,1);
#pragma omp parallel for schedule(static) if(n>1 && !omp_in_parallel())
    for (size_t row=0;row<n;++row) {
      const float row_sq=ProjectionBound::squared_norm(chunk+row*d,d);
      const float projected_sq=ProjectionBound::squared_norm(state.projected.data()+row*dp,dp);
      state.projected_sq[row]=projected_sq;
      state.residual[row]=std::sqrt(std::max(row_sq-projected_sq,0.0f));
      const size_t slot=row*probes_;
      const uint32_t list=state.best_list[slot];
      state.limit[row]=list==std::numeric_limits<uint32_t>::max() ? std::numeric_limits<float>::infinity()
          : row_sq-2*state.best[slot]+bound_slack*(row_sq+2*coarse_.half_norms[list]);
    }
    const auto work=group_slots(n,state,1,probes_);
#pragma omp parallel for schedule(dynamic) if(work.size()>1 && !omp_in_parallel())
    for (size_t item=0;item<work.size();++item)
      score_bounded_block(chunk,work[item].super,state.slots.data()+work[item].first,work[item].count,state);
  }

  uint32_t best_of_probes(const ChunkState& state,size_t row) const {
    float best=-std::numeric_limits<float>::infinity();
    uint32_t label=std::numeric_limits<uint32_t>::max();
    for (size_t p=0;p<probes_;++p) {
      const size_t slot=row*probes_+p;
      if (state.best[slot]>best || (state.best[slot]==best && state.best_list[slot]<label)) {
        best=state.best[slot]; label=state.best_list[slot];
      }
    }
    return label;
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

  // score_block for slots beyond the first probe: bounds on the projected panels, full rescoring of survivors only.
  void score_bounded_block(const float* chunk,uint32_t s,const uint32_t* slots,size_t count,ChunkState& state) const {
    const auto& members=members_[s];
    if (members.empty()) return;
    const size_t d=coarse_.dimensions,dp=bound_->dimensions(),width=bound_panels_[s].padded_columns();
    std::vector<float> gathered(product(count,dp)),dots(product(count,width));
    for (size_t i=0;i<count;++i) std::copy_n(state.projected.data()+size_t(slots[i]/probes_)*dp,dp,gathered.data()+i*dp);
    float_matrix(gathered.data(),count,dp,bound_panels_[s],dots.data(),width);
    const uint16_t* halves=member_halves_[s].data();
    const BoundTerms& terms=member_bound_terms_[s];
    std::vector<uint32_t> survivors(members.size());
    for (size_t i=0;i<count;++i) {
      const size_t row=slots[i]/probes_;
      const float* x=chunk+row*d;
      const BoundRow bound{state.projected_sq[row],state.residual[row],state.limit[row]};
      const size_t found=bound_survivors(bound,dots.data()+i*width,terms.projected_sq.data(),terms.residual.data(),
                                         members.size(),survivors.data());
      float best=-std::numeric_limits<float>::infinity();
      uint32_t label=std::numeric_limits<uint32_t>::max();
      for (size_t at=0;at<found;++at) {
        const uint32_t m=survivors[at],list=members[m];
        const float value=half_row_dot(x,halves+size_t(m)*d,d)*coarse_.scales[list]-coarse_.half_norms[list];
        if (value>best || (value==best && list<label)) { best=value; label=list; }
      }
      state.best[slots[i]]=best; state.best_list[slots[i]]=label;
    }
  }

  const CoarseQuantizer& coarse_;
  CoarseQuantizer supers_;
  std::vector<std::vector<uint32_t>> members_;  // list ids per super, ascending
  std::vector<HalfPanels> panels_;              // per super: its members' rounded centroids
  size_t probes_;
  std::optional<ProjectionBound> bound_;
  std::vector<TieredQPanels> bound_panels_;     // per super: its members' projections
  std::vector<BoundTerms> member_bound_terms_;
  // Per super, its members' binary16 rows contiguous for rescoring survivors: a panel column spans d / 2 cache lines
  // where a row spans d / 32 (gathering columns made the bounded assignment slower than the unbounded one).
  std::vector<std::vector<uint16_t>> member_halves_;
};

// Labels rows with the coarse quantizer's nearest list: exactly (BLAS), or restricted when training sets supers.
class ListAssigner {
 public:
  ListAssigner(const CoarseQuantizer& coarse,const TieredQCoarseTraining& training,uint64_t seed) {
    if (training.assignment_supers)
      restricted_.emplace(coarse,training.assignment_supers,training.assignment_probes,seed,
                          training.assignment_bound_dimensions);
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

} // namespace tq::tieredq_detail
