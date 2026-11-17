// Unquantized IVF (#56): k-means lists over FP32 rows and the list-major batch engine of tieredq_ivf_flat.h.
//
// A batch runs in three steps. route() ranks every query's nprobe nearest lists (one sgemm per query chunk against
// the centroids). Queries are then grouped by their nearest list, so a group's members probe mostly the same lists.
// Each group is one task: for every list any member probes, the members probing it are gathered and scored against
// the list's rows in one sgemm, and each member keeps its own top k. A task owns its members' results, so workers
// never share state, and each list's rows are read once per group instead of once per query.
#include "tieredq/tieredq_ivf_flat.h"
#include "tieredq_io.h"
#include "tieredq_ivf_assign.h"
#include "tieredq_ivf_coarse.h"
#include "training_rng.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <omp.h>

namespace tq {
using tieredq_detail::BlasAssigner;
using tieredq_detail::BlasThreads;
using tieredq_detail::CoarseQuantizer;
using tieredq_detail::TrainingLinalg;
using tieredq_detail::bad;
using tieredq_detail::product;

namespace {
constexpr uint64_t sample_salt=0x9e3779b97f4a7c15ULL;
constexpr size_t group_rows=256;             // queries per task: bounds the gathered block and the per-task results
constexpr size_t route_budget_floats=size_t(1)<<23;  // closeness buffer of one routing chunk (32 MiB)
constexpr uint32_t missing_id=std::numeric_limits<uint32_t>::max();
constexpr float missing_score=-std::numeric_limits<float>::infinity();

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
float dot(const float* a,const float* b,size_t d) {
  float sum=0;
  for (size_t i=0;i<d;++i) sum+=a[i]*b[i];
  return sum;
}
void normalize(float* row,size_t d) {
  const float norm=std::sqrt(dot(row,row,d));
  if (norm>0) for (size_t i=0;i<d;++i) row[i]/=norm;
}
// column-major out (m x n) = a^T b for row-major a (m x d) and b (n x d): out[j * m + i] = a_i . b_j.
void dots(const float* a,size_t m,const float* b,size_t n,size_t d,float* out) {
  const int mi=int(m),ni=int(n),di=int(d); const float one=1,zero=0; const char transposed='T',normal='N';
  TrainingLinalg::get().gemm(&transposed,&normal,&mi,&ni,&di,&one,a,&di,b,&di,&zero,out,&mi);
}

// The k best (closeness, id) of one query, best first.
struct TopK {
  float* scores; uint32_t* ids; size_t k;
  float threshold() const { return scores[k-1]; }
  void insert(float score,uint32_t id) {
    size_t slot=k-1;
    while (slot>0 && scores[slot-1]<score) { scores[slot]=scores[slot-1]; ids[slot]=ids[slot-1]; --slot; }
    scores[slot]=score; ids[slot]=id;
  }
};

// One task's reusable buffers.
struct Workspace {
  std::vector<uint64_t> probes;  // (list << 32) | member, sorted by list
  std::vector<uint32_t> members;
  std::vector<float> block,closeness,best_scores;
  std::vector<uint32_t> best_ids;
};
} // namespace

void TieredQIVFFlatConfig::validate() const {
  if (!iterations) bad("IVF-Flat training needs at least one iteration");
  if (!sample_rows_per_list) bad("IVF-Flat training needs at least one sample row per list");
}

uint32_t TieredQIVFFlatIndex::automatic_lists(size_t rows) {
  if (rows<2) return 1;
  const double exponent=std::round(std::log2(std::sqrt(double(rows))));
  return uint32_t(std::min<size_t>(rows,size_t(1)<<size_t(exponent)));
}

struct TieredQIVFFlatIndex::State {
  uint32_t dimensions;
  TieredQIVFFlatConfig config;
  TieredQIVFFlatBuildStats stats;
  std::shared_ptr<const CoarseQuantizer> coarse;
  std::vector<float> centroids;     // the coarse quantizer's rounded centroids, row-major, for sgemm routing
  std::vector<float> vectors;       // rows grouped by list (normalized for cosine)
  std::vector<float> half_norms;    // |x|^2 / 2 per stored row (l2), else 0: closeness = q.x - half_norm
  std::vector<uint32_t> ids;        // row number per stored row
  std::vector<size_t> offsets;      // list l holds stored rows [offsets[l], offsets[l + 1])

  size_t size() const { return ids.size(); }
  size_t lists() const { return coarse ? coarse->lists : 0; }
  bool l2() const { return config.metric==TieredQMetric::l2; }

  // probes[q * nprobe + p]: query q's p-th nearest list (by q.c - |c|^2/2).
  std::vector<uint32_t> route(const float* queries,size_t count,size_t nprobe) const {
    const size_t L=lists(),d=dimensions;
    const size_t chunk=std::max<size_t>(1,std::min(count,route_budget_floats/L));
    std::vector<uint32_t> probes(product(count,nprobe));
    std::vector<float> closeness(product(chunk,L));
    const BlasThreads threads;
    for (size_t first=0;first<count;first+=chunk) {
      const size_t n=std::min(chunk,count-first);
      dots(centroids.data(),L,queries+first*d,n,d,closeness.data());
#pragma omp parallel
      {
        std::vector<uint32_t> order(L);
#pragma omp for schedule(static)
        for (size_t r=0;r<n;++r) {
          float* score=closeness.data()+r*L;
          for (size_t c=0;c<L;++c) score[c]-=coarse->half_norms[c];
          std::iota(order.begin(),order.end(),0u);
          const auto better=[&](uint32_t a,uint32_t b) { return score[a]>score[b] || (score[a]==score[b] && a<b); };
          std::partial_sort(order.begin(),order.begin()+ptrdiff_t(nprobe),order.end(),better);
          std::copy_n(order.begin(),nprobe,probes.begin()+ptrdiff_t((first+r)*nprobe));
        }
      }
    }
    return probes;
  }

  // Batch engine. query_ids (optional): the stored row each query is, excluded from its results and naming its
  // output row; otherwise query q writes output row q.
  void join(const float* queries,size_t count,size_t k,size_t nprobe,const uint32_t* query_ids,float* scores,
            uint32_t* out_ids) const {
    const size_t d=dimensions,L=lists();
    if (!count || !k) return;
    if (!size()) bad("search on an empty IVF-Flat index");
    nprobe=std::min(nprobe,L);
    if (!nprobe) bad("nprobe must be positive");
    const auto probes=route(queries,count,nprobe);

    // Queries by nearest list, cut into tasks of at most group_rows; largest first for a short tail.
    std::vector<size_t> by_list(L+1,0);
    for (size_t q=0;q<count;++q) ++by_list[probes[q*nprobe]+1];
    for (size_t l=0;l<L;++l) by_list[l+1]+=by_list[l];
    std::vector<uint32_t> grouped(count);
    {
      auto cursor=by_list;
      for (size_t q=0;q<count;++q) grouped[cursor[probes[q*nprobe]]++]=uint32_t(q);
    }
    std::vector<std::pair<size_t,size_t>> tasks;  // [begin, end) in grouped
    for (size_t l=0;l<L;++l)
      for (size_t begin=by_list[l];begin<by_list[l+1];begin+=group_rows)
        tasks.emplace_back(begin,std::min(by_list[l+1],begin+group_rows));
    std::stable_sort(tasks.begin(),tasks.end(),[](const auto& a,const auto& b) { return a.second-a.first>b.second-b.first; });

    const int threads=omp_get_max_threads();
    const BlasThreads single(1);  // each worker's sgemm stays on its own thread
#pragma omp parallel num_threads(threads)
    {
      Workspace work;
#pragma omp for schedule(dynamic,1)
      for (size_t t=0;t<tasks.size();++t) {
        const auto [begin,end]=tasks[t];
        const size_t n=end-begin;
        work.best_scores.assign(n*k,missing_score);
        work.best_ids.assign(n*k,missing_id);
        work.probes.clear();
        for (size_t member=0;member<n;++member)
          for (size_t p=0;p<nprobe;++p)
            work.probes.push_back(uint64_t(probes[size_t(grouped[begin+member])*nprobe+p])<<32 | member);
        std::sort(work.probes.begin(),work.probes.end());

        for (size_t run=0;run<work.probes.size();) {
          const uint32_t list=uint32_t(work.probes[run]>>32);
          work.members.clear();
          for (;run<work.probes.size() && uint32_t(work.probes[run]>>32)==list;++run)
            work.members.push_back(uint32_t(work.probes[run]));
          const size_t first=offsets[list],m=offsets[list+1]-first,s=work.members.size();
          if (!m) continue;
          work.block.resize(s*d);
          for (size_t i=0;i<s;++i)
            std::copy_n(queries+size_t(grouped[begin+work.members[i]])*d,d,work.block.data()+i*d);
          work.closeness.resize(s*m);
          dots(vectors.data()+first*d,m,work.block.data(),s,d,work.closeness.data());
          for (size_t i=0;i<s;++i) {
            const uint32_t member=work.members[i];
            const uint32_t self=query_ids ? query_ids[grouped[begin+member]] : missing_id;
            TopK best{work.best_scores.data()+member*k,work.best_ids.data()+member*k,k};
            const float* row=work.closeness.data()+i*m;
            const float* bias=half_norms.data()+first;
            float threshold=best.threshold();
            for (size_t j=0;j<m;++j) {
              const float score=row[j]-bias[j];
              if (!(score>threshold) || ids[first+j]==self) continue;
              best.insert(score,ids[first+j]);
              threshold=best.threshold();
            }
          }
        }

        for (size_t member=0;member<n;++member) {
          const uint32_t q=grouped[begin+member];
          const size_t out=query_ids ? query_ids[q] : q;
          // l2 scores are -|q - x|^2 = 2 (q.x - |x|^2/2) - |q|^2.
          const float query_norm=l2() ? dot(queries+size_t(q)*d,queries+size_t(q)*d,d) : 0.0f;
          for (size_t j=0;j<k;++j) {
            const float closeness=work.best_scores[member*k+j];
            scores[out*k+j]=l2() ? 2*closeness-query_norm : closeness;
            out_ids[out*k+j]=work.best_ids[member*k+j];
          }
        }
      }
    }
  }
};

TieredQIVFFlatIndex::TieredQIVFFlatIndex(uint32_t dimensions,TieredQIVFFlatConfig config)
    : state_(std::make_unique<State>()) {
  if (!dimensions) bad("IVF-Flat needs at least one dimension");
  config.validate();
  state_->dimensions=dimensions;
  state_->config=config;
}
TieredQIVFFlatIndex::~TieredQIVFFlatIndex()=default;
TieredQIVFFlatIndex::TieredQIVFFlatIndex(TieredQIVFFlatIndex&&) noexcept=default;
TieredQIVFFlatIndex& TieredQIVFFlatIndex::operator=(TieredQIVFFlatIndex&&) noexcept=default;

void TieredQIVFFlatIndex::build(const float* input,size_t count) {
  State& s=*state_;
  const size_t d=s.dimensions;
  if (!input && count) bad("null vector data");
  if (count>size_t(missing_id)) bad("IVF-Flat holds at most 2^32 - 1 rows");
  const size_t lists=s.config.lists ? s.config.lists : automatic_lists(count);
  if (count<lists) bad("IVF-Flat training needs at least one row per list");
  const bool cosine=s.config.metric==TieredQMetric::cosine;
  std::vector<float> normalized;
  if (cosine) {
    normalized.assign(input,input+product(count,d));
#pragma omp parallel for schedule(static)
    for (size_t r=0;r<count;++r) normalize(normalized.data()+r*d,d);
    input=normalized.data();
  }

  auto started=std::chrono::steady_clock::now();
  const size_t sample_rows=std::min(count,product(lists,s.config.sample_rows_per_list));
  tieredq_detail::TrainingRng rng(s.config.seed^sample_salt);
  auto picked=rng.choice(count,sample_rows);
  std::sort(picked.begin(),picked.end());
  std::vector<float> sample(product(sample_rows,d));
  for (size_t i=0;i<sample_rows;++i) std::copy_n(input+picked[i]*d,d,sample.data()+i*d);
  const TieredQCoarseTraining training{s.config.iterations,0,TieredQCoarseTraining{}.assignment_probes,
                                       s.config.sample_rows_per_list};
  s.coarse=std::make_shared<const CoarseQuantizer>(
      tieredq_detail::train_coarse(sample.data(),sample_rows,d,lists,s.config.seed,training));
  s.centroids=s.coarse->centroids();
  s.stats.train_s=seconds_since(started);

  started=std::chrono::steady_clock::now();
  std::vector<uint32_t> labels(count);
  BlasAssigner(*s.coarse).assign(input,count,labels.data());
  s.stats.assign_s=seconds_since(started);

  started=std::chrono::steady_clock::now();
  s.offsets.assign(lists+1,0);
  for (const uint32_t l:labels) ++s.offsets[l+1];
  for (size_t l=0;l<lists;++l) s.offsets[l+1]+=s.offsets[l];
  s.ids.resize(count);
  {
    auto cursor=s.offsets;
    for (size_t r=0;r<count;++r) s.ids[cursor[labels[r]]++]=uint32_t(r);
  }
  s.vectors.resize(product(count,d));
  s.half_norms.resize(count);
  const bool l2=s.l2();
#pragma omp parallel for schedule(static)
  for (size_t pos=0;pos<count;++pos) {
    float* row=s.vectors.data()+pos*d;
    std::copy_n(input+size_t(s.ids[pos])*d,d,row);
    s.half_norms[pos]=l2 ? dot(row,row,d)/2 : 0.0f;
  }
  s.stats.store_s=seconds_since(started);
}

void TieredQIVFFlatIndex::search(const float* queries,size_t rows,size_t k,size_t nprobe,float* scores,
                                 uint32_t* ids) const {
  if (!queries && rows) bad("null query data");
  if (state_->config.metric!=TieredQMetric::cosine) {
    state_->join(queries,rows,k,nprobe,nullptr,scores,ids);
    return;
  }
  const size_t d=state_->dimensions;
  std::vector<float> normalized(queries,queries+product(rows,d));
  for (size_t r=0;r<rows;++r) normalize(normalized.data()+r*d,d);
  state_->join(normalized.data(),rows,k,nprobe,nullptr,scores,ids);
}

void TieredQIVFFlatIndex::knn_graph(size_t k,size_t nprobe,float* scores,uint32_t* ids) const {
  // Stored rows are the queries; each excludes itself and answers its own row number.
  state_->join(state_->vectors.data(),state_->size(),k,nprobe,state_->ids.data(),scores,ids);
}

size_t TieredQIVFFlatIndex::size() const { return state_->size(); }
uint32_t TieredQIVFFlatIndex::dimensions() const { return state_->dimensions; }
size_t TieredQIVFFlatIndex::lists() const { return state_->lists(); }
const TieredQIVFFlatConfig& TieredQIVFFlatIndex::config() const { return state_->config; }
const TieredQIVFFlatBuildStats& TieredQIVFFlatIndex::build_stats() const { return state_->stats; }

std::vector<uint32_t> TieredQIVFFlatIndex::list_assignment() const {
  std::vector<uint32_t> labels(size());
  for (size_t l=0;l<lists();++l)
    for (size_t pos=state_->offsets[l];pos<state_->offsets[l+1];++pos) labels[state_->ids[pos]]=uint32_t(l);
  return labels;
}

size_t TieredQIVFFlatIndex::storage_bytes() const {
  const State& s=*state_;
  return s.vectors.capacity()*sizeof(float)+s.half_norms.capacity()*sizeof(float)+s.ids.capacity()*sizeof(uint32_t)+
         s.offsets.capacity()*sizeof(size_t)+s.centroids.capacity()*sizeof(float)+
         (s.coarse ? s.coarse->allocated_bytes() : 0);
}
} // namespace tq
