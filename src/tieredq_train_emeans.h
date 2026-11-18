// E-means (Bessa et al., "E-Means: Scaling Laws for Mini-Batch K-Means", arXiv 2611.09501): mini-batch k-means whose
// centroids move by a per-cluster exponential moving average, with a batch size and an initial step from fitted
// scaling laws, a cosine step schedule, revival of starved clusters by splitting heavy ones, and closing means from the
// last pass. A port of Faiss's reference (faiss/contrib/e_means, NumpyEmeans at commit fc1095a): the same laws,
// constants, epoch layout and revival rule; assignment is exact FP32 BLAS, as for Lloyd.
//
// Why here (#45): per pass it costs what a Lloyd iteration costs (one nearest-centroid GEMM over the rows), but it
// reaches a lower objective in fewer passes. For TieredQIVFFlatIndex on MS MARCO 1M x 1024 (1,024 lists), 2.5 epochs on
// 256 rows per list gave the knn_graph recall of 10 Lloyd iterations on those rows (+0.006-0.015 over the 64-row
// default at equal join time), at the 64-row default's training budget. Being mini-batch, it streams its batches from
// the row source (EmeansTrainer), so its RAM does not grow with the training rows: at 8.84M x 1024 and 64K lists a
// 64-row-per-list FP32 sample held 17 GB, 7.7 GB more peak build RSS than fast Lloyd's 40-row sample.
#pragma once
#include "tieredq_ivf_assign.h"
#include "tieredq_profile.h"
#include "training_rng.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <vector>
#include <omp.h>

namespace tq::tieredq_detail {
namespace emeans {
constexpr double alpha_min_factor=0.1;       // the cosine schedule ends at this fraction of the initial step
constexpr size_t check_unused_every=5;       // batch updates between revival checks
constexpr double initial_usage=1.0/65536;    // initial EMA mass per cluster (the paper uses 1e-10)
constexpr double split_epsilon=1e-4;         // relative coordinate offset between a split donor and its child
constexpr double unused_threshold=0.1;       // revive clusters below this fraction of the mean mass
constexpr size_t no_revival_max_dimensions=2;
constexpr size_t quality_law_max_dimensions=16;
// Batch law B = c * k^a * n^b * T^e, (c, a, b, e); the quality law has an eighth of the throughput coefficient.
constexpr double throughput_law[4]={0.035134,0.134749,0.876473,0.813344};
constexpr double quality_law[4]={0.00439175,0.134749,0.876473,0.813344};
// Step law alpha0 = c * k^a * B^b * n^e * T^f, (c, a, b, e, f), clamped to [alpha0_min, 1].
constexpr double step_law[5]={0.8244,-0.4617,0.9003,-0.4498,-0.1057};
constexpr double alpha0_min=1e-4;
constexpr uint64_t epoch_salt=0x6e1e3a5d2c4b7f19ULL;
constexpr uint64_t split_salt=0x2545f4914f6cdd1dULL;
constexpr double pi=3.14159265358979323846;
} // namespace emeans

// The resolved plan of one run.
struct EmeansRecipe {
  size_t batch_rows=0;
  double alpha0=0;
  std::vector<size_t> epoch_rows;  // rows per epoch: whole passes, a fractional prefix, then the closing pass
  size_t total_steps=0;
  double unused_threshold=0;

  // The step of update `step`: cosine annealing from alpha0 to alpha_min_factor of it.
  double alpha(size_t step) const {
    const double fraction=double(std::min(step,total_steps))/double(std::max<size_t>(1,total_steps));
    const double cosine=0.5*(1.0+std::cos(emeans::pi*fraction));
    return alpha0*(emeans::alpha_min_factor+(1.0-emeans::alpha_min_factor)*cosine);
  }
};

// Nearest power of two, at least 1 (ties go up, as in the reference).
inline size_t round_power_of_two(double value) {
  if (value<=1) return 1;
  const double low=std::exp2(std::floor(std::log2(value)));
  return size_t(value/low<2*low/value ? low : 2*low);
}

inline EmeansRecipe emeans_recipe(size_t rows,size_t lists,size_t dimensions,double epochs) {
  if (!(epochs>0)) bad("e-means needs a positive epoch budget");
  const double n=double(rows),k=double(lists);
  const bool quality=dimensions<=emeans::quality_law_max_dimensions;
  const double* law=quality ? emeans::quality_law : emeans::throughput_law;
  size_t batch=round_power_of_two(law[0]*std::pow(k,law[1])*std::pow(n,law[2])*std::pow(epochs,law[3]));
  batch=quality ? std::min(batch,rows/2) : std::min(batch,round_power_of_two(n/2));
  batch=std::max<size_t>(1,std::min(batch,rows));
  const auto& s=emeans::step_law;
  const double alpha0=s[0]*std::pow(k,s[1])*std::pow(double(batch),s[2])*std::pow(n,s[3])*std::pow(epochs,s[4]);

  EmeansRecipe recipe;
  recipe.batch_rows=batch;
  recipe.alpha0=std::clamp(alpha0,emeans::alpha0_min,1.0);
  const double whole=std::floor(epochs);
  const auto remainder=size_t(std::nearbyint((epochs-whole)*n));  // ties to even, as Python's round()
  if (whole==0) recipe.epoch_rows={remainder};
  else {
    recipe.epoch_rows.assign(size_t(whole)-1,rows);
    if (remainder) recipe.epoch_rows.push_back(remainder);
    recipe.epoch_rows.push_back(rows);
  }
  for (const size_t epoch_rows:recipe.epoch_rows) recipe.total_steps+=(epoch_rows+batch-1)/batch;
  recipe.unused_threshold=dimensions<=emeans::no_revival_max_dimensions ? 0.0 : emeans::unused_threshold;
  return recipe;
}

// Training rows by position: fills out (count x d, row-major) with the rows at positions[0..count). Called from one
// thread at a time, with ascending positions within a call.
using TrainingRowSource=std::function<void(const uint32_t* positions,size_t count,float* out)>;

inline TrainingRowSource sample_source(const float* sample,size_t dimensions) {
  return [sample,dimensions](const uint32_t* positions,size_t count,float* out) {
#pragma omp parallel for schedule(static) if(count>=1024 && !omp_in_parallel())
    for (size_t i=0;i<count;++i) std::copy_n(sample+size_t(positions[i])*dimensions,dimensions,out+i*dimensions);
  };
}

// Restricted batch assignment for many lists, as fast Lloyd training assigns (RestrictedAssigner): each row's nearest
// centroid among the members of its probes nearest supers, groups of the centroids. Unlike Lloyd's assigner it scores
// the moving centroids themselves: regroup() clusters the centroids into supers (k-means), follow() moves each super to
// the mean of its members' current positions after every update, and reassign() moves centroids to their nearest
// super every few updates. prepare() copies the centroids into member order once per batch, so each super's members
// are one contiguous sgemm operand for every block of rows probing it (gathering them per block re-copied each super
// about 64 times per batch at 64K lists).
class SuperRestriction {
 public:
  SuperRestriction(size_t supers,size_t probes) : supers_(supers),probes_(std::max<size_t>(1,probes)) {}

  void regroup(const float* centroids,size_t lists,size_t d,uint64_t seed) {
    lists_=lists; d_=d;
    supers_=std::min(supers_,lists); probes_=std::min(probes_,supers_);
    means_=train_coarse(centroids,lists,d,supers_,seed,regroup_iterations).centroids();
    owner_.resize(lists);
    reassign(centroids);
  }
  // Each centroid to its nearest super, then the supers to their members' means.
  void reassign(const float* centroids) {
    std::vector<float> half(supers_);
    half_norms(means_.data(),supers_,d_,half.data());
    nearest_centroids(means_.data(),half.data(),supers_,d_,centroids,lists_,owner_.data());
    member_offsets_.assign(supers_+1,0);
    for (const uint32_t s:owner_) ++member_offsets_[s+1];
    std::partial_sum(member_offsets_.begin(),member_offsets_.end(),member_offsets_.begin());
    members_.resize(lists_);
    auto cursor=member_offsets_;
    for (uint32_t c=0;c<lists_;++c) members_[cursor[owner_[c]]++]=c;
    follow(centroids);
  }
  void follow(const float* centroids) {
#pragma omp parallel for schedule(static)
    for (size_t s=0;s<supers_;++s) {
      const size_t begin=member_offsets_[s],count=member_offsets_[s+1]-begin;
      if (!count) continue;
      float* mean=means_.data()+s*d_;
      std::fill_n(mean,d_,0.0f);
      for (size_t m=begin;m<begin+count;++m) {
        const float* centroid=centroids+size_t(members_[m])*d_;
        for (size_t j=0;j<d_;++j) mean[j]+=centroid[j];
      }
      for (size_t j=0;j<d_;++j) mean[j]/=float(count);
    }
  }
  void prepare(const float* centroids,const float* centroid_half_norms) {
    grouped_.resize(product(lists_,d_));
    grouped_half_norms_.resize(lists_);
#pragma omp parallel for schedule(static)
    for (size_t m=0;m<lists_;++m) {
      std::copy_n(centroids+size_t(members_[m])*d_,d_,grouped_.data()+m*d_);
      grouped_half_norms_[m]=centroid_half_norms[members_[m]];
    }
  }

  // Nearest prepared centroid per row among its probed supers' members; ties keep the lowest centroid id.
  void assign(const float* rows,size_t n,uint32_t* ids) {
    probe(rows,n);
    // Slots (row * probes + p) grouped by probed super; work items of at most block_rows slots of one super.
    std::vector<size_t> begin(supers_+1,0);
    for (const uint32_t s:probed_) ++begin[s+1];
    std::partial_sum(begin.begin(),begin.end(),begin.begin());
    slots_.resize(probed_.size());
    {
      auto cursor=begin;
      for (size_t slot=0;slot<probed_.size();++slot) slots_[cursor[probed_[slot]]++]=uint32_t(slot);
    }
    struct Work { uint32_t super; size_t first,count; };
    std::vector<Work> work;
    for (uint32_t s=0;s<supers_;++s)
      for (size_t at=begin[s];at<begin[s+1];at+=block_rows) work.push_back({s,at,std::min(block_rows,begin[s+1]-at)});
    best_.assign(probed_.size(),-std::numeric_limits<float>::infinity());
    best_ids_.assign(probed_.size(),std::numeric_limits<uint32_t>::max());
    const BlasThreads single(1);  // each worker's sgemm stays on its own thread
#pragma omp parallel
    {
      std::vector<float> block,dots;
#pragma omp for schedule(dynamic,1)
      for (size_t item=0;item<work.size();++item)
        score(work[item].super,slots_.data()+work[item].first,work[item].count,rows,block,dots);
    }
#pragma omp parallel for schedule(static)
    for (size_t row=0;row<n;++row) {
      float best=-std::numeric_limits<float>::infinity();
      uint32_t label=std::numeric_limits<uint32_t>::max();
      for (size_t p=0;p<probes_;++p) {
        const size_t slot=row*probes_+p;
        if (best_[slot]>best || (best_[slot]==best && best_ids_[slot]<label)) { best=best_[slot]; label=best_ids_[slot]; }
      }
      ids[row]=label;
    }
  }

  static void half_norms(const float* centroids,size_t count,size_t d,float* out) {
#pragma omp parallel for schedule(static)
    for (size_t c=0;c<count;++c) {
      const float* centroid=centroids+c*d;
      double sum=0;
      for (size_t j=0;j<d;++j) sum+=double(centroid[j])*centroid[j];
      out[c]=float(sum/2);
    }
  }

 private:
  static constexpr size_t regroup_iterations=6,block_rows=256,chunk_rows=4096;

  // probed_[row * probes + p]: the row's p-th nearest super (nonempty supers only rank above empty ones).
  void probe(const float* rows,size_t n) {
    probed_.resize(n*probes_);
    std::vector<float> half(supers_),closeness(product(std::min(chunk_rows,n),supers_));
    half_norms(means_.data(),supers_,d_,half.data());
    for (size_t s=0;s<supers_;++s) if (member_offsets_[s]==member_offsets_[s+1]) half[s]=std::numeric_limits<float>::infinity();
    const BlasThreads threads;
    const auto& linalg=TrainingLinalg::get();
    for (size_t first=0;first<n;first+=chunk_rows) {
      const size_t m=std::min(chunk_rows,n-first);
      const int si=int(supers_),mi=int(m),di=int(d_); const float one=1,zero=0; const char transposed='T',normal='N';
      linalg.gemm(&transposed,&normal,&si,&mi,&di,&one,means_.data(),&di,rows+first*d_,&di,&zero,closeness.data(),&si);
#pragma omp parallel
      {
        std::vector<uint32_t> order(supers_);
#pragma omp for schedule(static)
        for (size_t r=0;r<m;++r) {
          float* score=closeness.data()+r*supers_;
          for (size_t s=0;s<supers_;++s) score[s]-=half[s];
          std::iota(order.begin(),order.end(),0u);
          std::partial_sort(order.begin(),order.begin()+ptrdiff_t(probes_),order.end(),[&](uint32_t a,uint32_t b) {
            return score[a]>score[b] || (score[a]==score[b] && a<b);
          });
          std::copy_n(order.begin(),probes_,probed_.begin()+ptrdiff_t((first+r)*probes_));
        }
      }
    }
  }

  void score(uint32_t s,const uint32_t* slots,size_t count,const float* rows,std::vector<float>& block,
             std::vector<float>& dots) {
    const size_t begin=member_offsets_[s],members=member_offsets_[s+1]-begin;
    if (!members) return;
    block.resize(product(count,d_));
    for (size_t i=0;i<count;++i) std::copy_n(rows+size_t(slots[i]/probes_)*d_,d_,block.data()+i*d_);
    dots.resize(product(count,members));
    const int mi=int(members),ni=int(count),di=int(d_); const float one=1,zero=0; const char transposed='T',normal='N';
    TrainingLinalg::get().gemm(&transposed,&normal,&mi,&ni,&di,&one,grouped_.data()+begin*d_,&di,block.data(),&di,&zero,
                               dots.data(),&mi);
    const float* half=grouped_half_norms_.data()+begin;
    for (size_t i=0;i<count;++i) {
      float best=-std::numeric_limits<float>::infinity();
      uint32_t label=std::numeric_limits<uint32_t>::max();
      for (size_t m=0;m<members;++m) {
        const uint32_t c=members_[begin+m];
        const float value=dots[i*members+m]-half[m];
        if (value>best || (value==best && c<label)) { best=value; label=c; }
      }
      best_[slots[i]]=best; best_ids_[slots[i]]=label;
    }
  }

  size_t supers_,probes_,lists_=0,d_=0;
  std::vector<float> means_,grouped_,grouped_half_norms_,best_;
  std::vector<uint32_t> owner_,members_,probed_,slots_,best_ids_;
  std::vector<size_t> member_offsets_;
};

// Streams every batch from a TrainingRowSource in chunks of chunk_rows, accumulating per-cluster sums, so training holds
// the centroids, the batch sums and the closing pass's sums (each lists x d floats) and one chunk, never the rows.
class EmeansTrainer {
 public:
  static constexpr size_t chunk_rows=32768;

  // supers > 0: restricted batch assignment (SuperRestriction with probes), else exact.
  EmeansTrainer(TrainingRowSource source,size_t rows,size_t dimensions,size_t lists,uint64_t seed,double epochs,
                size_t supers=0,size_t probes=0)
      : source_(std::move(source)),rows_(rows),d_(dimensions),k_(lists),seed_(seed),
        recipe_(emeans_recipe(rows,lists,dimensions,epochs)),split_rng_(seed^emeans::split_salt) {
    if (lists<1 || rows<lists) bad("e-means needs at least one row per list");
    if (rows>std::numeric_limits<uint32_t>::max()) bad("e-means training exceeds uint32 row positions");
    if (supers && supers<lists) restriction_.emplace(supers,probes);
  }

  // Centroids (lists x dimensions, row-major) from lists distinct seeded rows.
  std::vector<float> train() {
    initialize();
    profile_.mark("initialize");
    size_t step=0;
    std::vector<uint32_t> batch;
    for (size_t epoch=0;epoch<recipe_.epoch_rows.size();++epoch) {
      const bool closing=epoch+1==recipe_.epoch_rows.size();
      if (closing) { pass_sums_.assign(product(k_,d_),0.0f); pass_counts_.assign(k_,0); }
      const auto order=TrainingRng(seed_^emeans::epoch_salt^(epoch*0x9e3779b97f4a7c15ULL)).choice(rows_,rows_);
      const size_t epoch_rows=recipe_.epoch_rows[epoch];
      for (size_t first=0;first<epoch_rows;first+=recipe_.batch_rows) {
        const size_t n=std::min(recipe_.batch_rows,epoch_rows-first);
        batch.assign(order.begin()+ptrdiff_t(first),order.begin()+ptrdiff_t(first+n));
        std::sort(batch.begin(),batch.end());  // reads the source in order; a batch's update ignores row order
        profile_.mark("order");
        accumulate_batch(batch.data(),n);
        apply_batch(recipe_.alpha(step),closing);
        profile_.mark("apply");
        ++step;
        const bool check=step%emeans::check_unused_every==0;
        if (check && recipe_.unused_threshold>0) revive_unused();
        if (restriction_) {
          if (check) restriction_->reassign(centroids_.data());
          else restriction_->follow(centroids_.data());
        }
        profile_.mark("regroup");
      }
    }
    close();
    profile_.mark("close");
    return std::move(centroids_);
  }

  const EmeansRecipe& recipe() const { return recipe_; }

 private:
  void initialize() {
    TrainingRng rng(seed_);
    const auto initial=rng.choice(rows_,k_);
    centroids_.resize(product(k_,d_));
    // Unsorted: centroid c starts at row initial[c]. Read a chunk at a time so a lifting source stays bounded.
    std::vector<uint32_t> positions;
    for (size_t first=0;first<k_;first+=chunk_rows) {
      const size_t n=std::min(chunk_rows,k_-first);
      positions.assign(initial.begin()+ptrdiff_t(first),initial.begin()+ptrdiff_t(first+n));
      std::vector<uint32_t> order(n);
      std::iota(order.begin(),order.end(),0u);
      std::sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b) { return positions[a]<positions[b]; });
      std::vector<uint32_t> sorted(n);
      for (size_t i=0;i<n;++i) sorted[i]=positions[order[i]];
      chunk_.resize(product(n,d_));
      source_(sorted.data(),n,chunk_.data());
      for (size_t i=0;i<n;++i) std::copy_n(chunk_.data()+i*d_,d_,centroids_.data()+(first+order[i])*d_);
    }
    usage_.assign(k_,emeans::initial_usage);
    half_norms_.resize(k_);
    sums_.assign(product(k_,d_),0.0f);
    counts_.assign(k_,0);
    chunk_offsets_.resize(k_+1);
    if (restriction_) restriction_->regroup(centroids_.data(),k_,d_,seed_^emeans::split_salt);
  }

  // Labels the batch's rows against the current centroids a chunk at a time, adding each row to its cluster's sum.
  void accumulate_batch(const uint32_t* positions,size_t n) {
    SuperRestriction::half_norms(centroids_.data(),k_,d_,half_norms_.data());
    if (restriction_) restriction_->prepare(centroids_.data(),half_norms_.data());
    profile_.mark("prepare");
    for (size_t first=0;first<n;first+=chunk_rows) {
      const size_t m=std::min(chunk_rows,n-first);
      chunk_.resize(product(m,d_));
      source_(positions+first,m,chunk_.data());
      profile_.mark("load");
      labels_.resize(m);
      if (restriction_) restriction_->assign(chunk_.data(),m,labels_.data());
      else nearest_centroids(centroids_.data(),half_norms_.data(),k_,d_,chunk_.data(),m,labels_.data());
      profile_.mark("assign");
      add_chunk(m);
      profile_.mark("accumulate");
    }
  }

  void add_chunk(size_t m) {
    std::fill(chunk_offsets_.begin(),chunk_offsets_.end(),0);
    for (size_t i=0;i<m;++i) ++chunk_offsets_[labels_[i]+1];
    std::partial_sum(chunk_offsets_.begin(),chunk_offsets_.end(),chunk_offsets_.begin());
    members_.resize(m);
    {
      auto cursor=chunk_offsets_;
      for (size_t i=0;i<m;++i) members_[cursor[labels_[i]]++]=uint32_t(i);
    }
#pragma omp parallel for schedule(dynamic,64)
    for (size_t c=0;c<k_;++c) {
      const size_t begin=chunk_offsets_[c],count=chunk_offsets_[c+1]-begin;
      if (!count) continue;
      float* sum=sums_.data()+c*d_;
      for (size_t i=begin;i<begin+count;++i) {
        const float* row=chunk_.data()+size_t(members_[i])*d_;
        for (size_t j=0;j<d_;++j) sum[j]+=row[j];
      }
      counts_[c]+=uint32_t(count);
    }
  }

  // Equation 2 per cluster, with mass N and centroid c = S / N: N' = (1 - alpha) N + alpha n_j and
  // c' = c + (alpha n_j / N') (mean_j - c). Untouched clusters only decay. The closing pass also collects plain sums.
  // Clears the batch sums.
  void apply_batch(double alpha,bool closing) {
    const bool replace=alpha>=1;
    const double decay=1.0-alpha;
#pragma omp parallel for schedule(dynamic,64)
    for (size_t c=0;c<k_;++c) {
      const size_t count=counts_[c];
      const double retained=replace ? 0.0 : usage_[c]*decay;
      if (!count) { if (!replace) usage_[c]=retained; continue; }
      float* sum=sums_.data()+c*d_;
      const double mass=retained+alpha*double(count);
      const double weight=alpha*double(count)/mass;
      float* centroid=centroids_.data()+c*d_;
      for (size_t j=0;j<d_;++j) {
        const double mean=double(sum[j])/double(count);
        centroid[j]=weight==1 ? float(mean) : float(centroid[j]+weight*(mean-centroid[j]));
      }
      usage_[c]=mass;
      if (closing) {
        float* pass=pass_sums_.data()+c*d_;
        for (size_t j=0;j<d_;++j) pass[j]+=sum[j];
        pass_counts_[c]+=count;
      }
      std::fill_n(sum,d_,0.0f);
      counts_[c]=0;
    }
  }

  // Clusters below unused_threshold x the mean mass take half of a heavy donor: donors are drawn without replacement
  // in proportion to their mass (Gumbel top-k), each split into two centroids offset by +-split_epsilon per
  // coordinate (alternating signs).
  void revive_unused() {
    const double threshold=std::accumulate(usage_.begin(),usage_.end(),0.0)*recipe_.unused_threshold/double(k_);
    std::vector<uint32_t> unused,donors;
    for (uint32_t c=0;c<k_;++c) {
      if (usage_[c]<threshold) unused.push_back(c);
      else if (usage_[c]>0) donors.push_back(c);
    }
    const size_t splits=std::min(unused.size(),donors.size());
    if (!splits) return;
    std::uniform_real_distribution<double> uniform(0.0,1.0);
    std::vector<double> keys(k_,-std::numeric_limits<double>::infinity());
    for (const uint32_t c:donors) {
      const double u=std::max(uniform(split_rng_),std::numeric_limits<double>::min());
      keys[c]=std::log(usage_[c])-std::log(-std::log(u));
    }
    std::partial_sort(donors.begin(),donors.begin()+ptrdiff_t(splits),donors.end(),
                      [&](uint32_t a,uint32_t b) { return keys[a]>keys[b]; });
    for (size_t s=0;s<splits;++s) split(donors[s],unused[s]);
  }

  void split(uint32_t donor,uint32_t child) {
    float* from=centroids_.data()+size_t(donor)*d_;
    float* to=centroids_.data()+size_t(child)*d_;
    bool collapsed=true;
    for (size_t j=0;j<d_;++j) {
      const double offset=emeans::split_epsilon*(j%2 ? -1.0 : 1.0);
      const float value=from[j];
      from[j]=float(value*(1.0-offset));
      to[j]=float(value*(1.0+offset));
      collapsed=collapsed && from[j]==to[j];
    }
    if (collapsed) {  // a zero donor: separate the pair along the first coordinate
      const float next=std::nextafter(to[0],std::numeric_limits<float>::infinity());
      to[0]=std::isfinite(next) ? next : std::nextafter(to[0],-std::numeric_limits<float>::infinity());
    }
    usage_[donor]/=2;
    usage_[child]=usage_[donor];
  }

  // Algorithm 1, line 12: unweighted means of the closing pass's assignments; never-assigned centroids stay.
  void close() {
#pragma omp parallel for schedule(static)
    for (size_t c=0;c<k_;++c) {
      if (!pass_counts_[c]) continue;
      const float inverse=1.0f/float(pass_counts_[c]);
      for (size_t j=0;j<d_;++j) centroids_[c*d_+j]=pass_sums_[c*d_+j]*inverse;
    }
  }

  TrainingRowSource source_;
  size_t rows_,d_,k_;
  uint64_t seed_;
  EmeansRecipe recipe_;
  std::mt19937_64 split_rng_;
  std::vector<float> centroids_,half_norms_,chunk_,sums_,pass_sums_;
  std::vector<double> usage_;
  std::vector<size_t> chunk_offsets_,pass_counts_;
  std::vector<uint32_t> counts_,labels_,members_;
  std::optional<SuperRestriction> restriction_;
  StageProfile profile_{"tieredq_emeans_profile"};
};

inline std::vector<float> train_emeans(const float* sample,size_t rows,size_t dimensions,size_t lists,uint64_t seed,
                                       double epochs,size_t supers=0,size_t probes=0) {
  return EmeansTrainer(sample_source(sample,dimensions),rows,dimensions,lists,seed,epochs,supers,probes).train();
}

inline size_t emeans_probes(const TieredQCoarseTraining& training) {
  return training.emeans_probes ? training.emeans_probes : training.assignment_probes;
}

// List training as TieredQCoarseTraining says: e-means (emeans_epochs > 0) or Lloyd iterations, each with exact or
// restricted (assignment_supers > 0) assignment.
inline CoarseQuantizer train_coarse(const float* sample,size_t rows,size_t dimensions,size_t lists,uint64_t seed,
                                    const TieredQCoarseTraining& training) {
  if (training.emeans_epochs>0)
    return CoarseQuantizer::from(train_emeans(sample,rows,dimensions,lists,seed,training.emeans_epochs,
                                              training.assignment_supers,emeans_probes(training)),lists,dimensions);
  return train_coarse(sample,rows,dimensions,lists,seed,training.iterations,
                      [&](const CoarseQuantizer& q,const float* x,size_t n,uint32_t* ids) {
                        ListAssigner(q,training,seed).assign(x,n,ids);
                      });
}

// train_coarse on rows read through a source: e-means streams its batches from it and never holds the rows; Lloyd
// reads them into RAM once.
inline CoarseQuantizer train_coarse_rows(const TrainingRowSource& source,size_t rows,size_t dimensions,size_t lists,
                                         uint64_t seed,const TieredQCoarseTraining& training) {
  if (training.emeans_epochs>0)
    return CoarseQuantizer::from(EmeansTrainer(source,rows,dimensions,lists,seed,training.emeans_epochs,
                                               training.assignment_supers,emeans_probes(training)).train(),
                                 lists,dimensions);
  std::vector<float> sample(product(rows,dimensions));
  std::vector<uint32_t> positions;
  for (size_t first=0;first<rows;first+=EmeansTrainer::chunk_rows) {
    const size_t n=std::min(EmeansTrainer::chunk_rows,rows-first);
    positions.resize(n);
    std::iota(positions.begin(),positions.end(),uint32_t(first));
    source(positions.data(),n,sample.data()+first*dimensions);
  }
  return train_coarse(sample.data(),rows,dimensions,lists,seed,training);
}
} // namespace tq::tieredq_detail
