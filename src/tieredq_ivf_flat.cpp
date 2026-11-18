// Flat IVF (#56): k-means lists over FP32 or SQ8 rows and the list-major batch engine of tieredq_ivf_flat.h.
//
// A batch runs in three steps. route() ranks every query's nprobe nearest lists (one sgemm per query chunk against
// the centroids). Queries are then grouped by their nearest list, so a group's members probe mostly the same lists.
// Each group is one task: for every list any member probes, the members probing it are gathered and scored against
// the list's rows in one sgemm, and each member keeps its own top k. A task owns its members' results, so workers
// never share state, and each list's rows are read once per group instead of once per query. Lists are scored in
// L2-sized tiles of rows; SQ8 rows are decoded into such an FP32 tile at that read, a cost of about one in (members
// probing the list) of the sgemm's.
#include "tieredq/tieredq_ivf_flat.h"
#include "tieredq_io.h"
#include "tieredq_ivf_assign.h"
#include "tieredq_ivf_coarse.h"
#include "tieredq_sq8.h"
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
using tieredq_detail::SQ8Codec;
using tieredq_detail::SQ8Range;
using tieredq_detail::TrainingLinalg;
using tieredq_detail::bad;
using tieredq_detail::product;

namespace {
constexpr uint64_t sample_salt=0x9e3779b97f4a7c15ULL;
constexpr size_t group_rows=256;             // queries per task: bounds the gathered block and the per-task results
constexpr size_t route_budget_floats=size_t(1)<<23;  // closeness buffer of one routing chunk (32 MiB)
constexpr size_t build_chunk_rows=size_t(1)<<14;     // rows decoded or normalized at a time for assignment
// FP32 floats of one list tile per sgemm (512 KiB): stays in L2 across the sgemm (and the SQ8 decode before it). On
// MS MARCO 1M x 1024 at 512 lists, SQ8 joins with such tiles took 83 s against 170 s for whole-list FP32 sgemms.
constexpr size_t list_tile_floats=size_t(1)<<17;
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

// Rows read as FP32: borrowed FP32 rows, or SQ8 codes decoded on read; build inputs may also normalize on read.
struct RowReader {
  size_t d=0;
  const float* rows=nullptr;
  const int8_t* codes=nullptr;
  const SQ8Codec* codec=nullptr;
  bool unit_rows=false;

  bool borrows() const { return rows && !unit_rows; }
  // Rows [first, first + count): borrowed when possible, else written to buffer (count x d).
  const float* read(size_t first,size_t count,float* buffer) const {
    if (borrows()) return rows+first*d;
    if (rows) std::copy_n(rows+first*d,count*d,buffer);
    else codec->decode(codes+first*d,count,buffer);
    if (unit_rows) for (size_t r=0;r<count;++r) normalize(buffer+r*d,d);
    return buffer;
  }
  void copy(size_t row,float* out) const {
    const float* source=read(row,1,out);
    if (source!=out) std::copy_n(source,d,out);
  }
  // read() split over the OpenMP workers.
  const float* read_parallel(size_t first,size_t count,std::vector<float>& buffer) const {
    if (borrows()) return rows+first*d;
    buffer.resize(product(count,d));
#pragma omp parallel for schedule(static)
    for (size_t r=0;r<count;++r) read(first+r,1,buffer.data()+r*d);
    return buffer.data();
  }
};

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
  std::vector<float> block,list_rows,closeness,best_scores,query;
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
  std::vector<float> vectors;       // fp32 storage: rows grouped by list (normalized for cosine)
  std::vector<int8_t> codes;        // sq8 storage: the same rows as codes
  SQ8Codec codec;
  std::vector<float> half_norms;    // |x|^2 / 2 per stored (decoded) row (l2), else 0: closeness = q.x - half_norm
  std::vector<uint32_t> ids;        // row number per stored row
  std::vector<size_t> offsets;      // list l holds stored rows [offsets[l], offsets[l + 1])

  size_t size() const { return ids.size(); }
  size_t lists() const { return coarse ? coarse->lists : 0; }
  bool l2() const { return config.metric==TieredQMetric::l2; }
  bool sq8() const { return config.storage==TieredQIVFFlatStorage::sq8; }
  RowReader stored() const {
    if (sq8()) return {dimensions,nullptr,codes.data(),&codec};
    return {dimensions,vectors.data()};
  }

  // input: count rows in row order, normalized for cosine. given: input's codec when its codes can be kept as-is.
  void build(const RowReader& input,size_t count,const SQ8Codec* given) {
    const size_t d=dimensions;
    if (count>size_t(missing_id)) bad("IVF-Flat holds at most 2^32 - 1 rows");
    const size_t list_count=config.lists ? config.lists : automatic_lists(count);
    if (count<list_count) bad("IVF-Flat training needs at least one row per list");

    auto started=std::chrono::steady_clock::now();
    const size_t sample_rows=std::min(count,product(list_count,config.sample_rows_per_list));
    tieredq_detail::TrainingRng rng(config.seed^sample_salt);
    auto picked=rng.choice(count,sample_rows);
    std::sort(picked.begin(),picked.end());
    std::vector<float> sample(product(sample_rows,d));
#pragma omp parallel for schedule(static)
    for (size_t i=0;i<sample_rows;++i) input.copy(picked[i],sample.data()+i*d);
    const TieredQCoarseTraining training{config.iterations,0,TieredQCoarseTraining{}.assignment_probes,
                                         config.sample_rows_per_list};
    coarse=std::make_shared<const CoarseQuantizer>(
        tieredq_detail::train_coarse(sample.data(),sample_rows,d,list_count,config.seed,training));
    centroids=coarse->centroids();
    sample=std::vector<float>();
    stats.train_s=seconds_since(started);

    started=std::chrono::steady_clock::now();
    std::vector<uint32_t> labels(count);
    {
      const BlasAssigner assigner(*coarse);
      const size_t chunk=input.borrows() ? count : build_chunk_rows;
      std::vector<float> buffer;
      for (size_t first=0;first<count;first+=chunk) {
        const size_t n=std::min(chunk,count-first);
        assigner.assign(input.read_parallel(first,n,buffer),n,labels.data()+first);
      }
    }
    stats.assign_s=seconds_since(started);

    started=std::chrono::steady_clock::now();
    offsets.assign(list_count+1,0);
    for (const uint32_t l:labels) ++offsets[l+1];
    for (size_t l=0;l<list_count;++l) offsets[l+1]+=offsets[l];
    ids.resize(count);
    {
      auto cursor=offsets;
      for (size_t r=0;r<count;++r) ids[cursor[labels[r]]++]=uint32_t(r);
    }
    half_norms.resize(count);
    if (sq8()) store_codes(input,given); else store_vectors(input);
    stats.store_s=seconds_since(started);
  }

  void store_vectors(const RowReader& input) {
    const size_t d=dimensions,count=size();
    codes=std::vector<int8_t>(); codec={};
    vectors.resize(product(count,d));
#pragma omp parallel for schedule(static)
    for (size_t pos=0;pos<count;++pos) {
      float* row=vectors.data()+pos*d;
      input.copy(ids[pos],row);
      half_norms[pos]=l2() ? dot(row,row,d)/2 : 0.0f;
    }
  }

  void store_codes(const RowReader& input,const SQ8Codec* given) {
    const size_t d=dimensions,count=size();
    vectors=std::vector<float>();
    codec=given ? *given : trained_codec(input,count);
    codes.resize(product(count,d));
#pragma omp parallel
    {
      std::vector<float> row(d);
#pragma omp for schedule(static)
      for (size_t pos=0;pos<count;++pos) {
        int8_t* code=codes.data()+pos*d;
        if (given) std::copy_n(input.codes+size_t(ids[pos])*d,d,code);
        else { input.copy(ids[pos],row.data()); codec.encode(row.data(),code); }
        if (!l2()) continue;
        codec.decode(code,1,row.data());
        half_norms[pos]=dot(row.data(),row.data(),d)/2;
      }
    }
    if (!l2()) std::fill(half_norms.begin(),half_norms.end(),0.0f);
  }

  SQ8Codec trained_codec(const RowReader& input,size_t count) const {
    const size_t d=dimensions;
    SQ8Range range(d);
#pragma omp parallel
    {
      SQ8Range local(d);
      std::vector<float> row(d);
#pragma omp for schedule(static) nowait
      for (size_t r=0;r<count;++r) { input.copy(r,row.data()); local.add(row.data()); }
#pragma omp critical
      range.merge(local);
    }
    return SQ8Codec::spanning(range.low,range.high);
  }

  // probes[q * nprobe + p]: query q's p-th nearest list (by q.c - |c|^2/2).
  std::vector<uint32_t> route(const RowReader& queries,size_t count,size_t nprobe) const {
    const size_t L=lists(),d=dimensions;
    const size_t chunk=std::max<size_t>(1,std::min(count,route_budget_floats/L));
    std::vector<uint32_t> probes(product(count,nprobe));
    std::vector<float> closeness(product(chunk,L)),buffer;
    const BlasThreads threads;
    for (size_t first=0;first<count;first+=chunk) {
      const size_t n=std::min(chunk,count-first);
      dots(centroids.data(),L,queries.read_parallel(first,n,buffer),n,d,closeness.data());
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
  void join(const RowReader& queries,size_t count,size_t k,size_t nprobe,const uint32_t* query_ids,float* scores,
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

    const RowReader rows=stored();
    const size_t list_tile_rows=std::max<size_t>(1,list_tile_floats/d);
    const int threads=omp_get_max_threads();
    const BlasThreads single(1);  // each worker's sgemm stays on its own thread
#pragma omp parallel num_threads(threads)
    {
      Workspace work;
      work.query.resize(d);
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
          const size_t list_first=offsets[list],list_rows=offsets[list+1]-list_first,s=work.members.size();
          if (!list_rows) continue;
          work.block.resize(s*d);
          for (size_t i=0;i<s;++i) queries.copy(grouped[begin+work.members[i]],work.block.data()+i*d);
          const size_t tile=std::min(list_rows,list_tile_rows);
          if (!rows.borrows()) work.list_rows.resize(tile*d);
          work.closeness.resize(s*tile);
          for (size_t tile_first=list_first;tile_first<list_first+list_rows;tile_first+=tile) {
            const size_t m=std::min(tile,list_first+list_rows-tile_first);
            dots(rows.read(tile_first,m,work.list_rows.data()),m,work.block.data(),s,d,work.closeness.data());
            for (size_t i=0;i<s;++i) {
              const uint32_t member=work.members[i];
              const uint32_t self=query_ids ? query_ids[grouped[begin+member]] : missing_id;
              TopK best{work.best_scores.data()+member*k,work.best_ids.data()+member*k,k};
              const float* row=work.closeness.data()+i*m;
              const float* bias=half_norms.data()+tile_first;
              const uint32_t* row_ids=ids.data()+tile_first;
              float threshold=best.threshold();
              for (size_t j=0;j<m;++j) {
                const float score=row[j]-bias[j];
                if (!(score>threshold) || row_ids[j]==self) continue;
                best.insert(score,row_ids[j]);
                threshold=best.threshold();
              }
            }
          }
        }

        for (size_t member=0;member<n;++member) {
          const uint32_t q=grouped[begin+member];
          const size_t out=query_ids ? query_ids[q] : q;
          // l2 scores are -|q - x|^2 = 2 (q.x - |x|^2/2) - |q|^2.
          float query_norm=0;
          if (l2()) {
            const float* query=queries.read(q,1,work.query.data());
            query_norm=dot(query,query,d);
          }
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
  if (!input && count) bad("null vector data");
  RowReader rows{state_->dimensions,input};
  rows.unit_rows=state_->config.metric==TieredQMetric::cosine;
  state_->build(rows,count,nullptr);
}

void TieredQIVFFlatIndex::build(const int8_t* codes,size_t count,const float* scales,const float* offsets) {
  if (!codes && count) bad("null code data");
  if (!scales || !offsets) bad("SQ8 build needs scales and offsets");
  const size_t d=state_->dimensions;
  const SQ8Codec input_codec{std::vector<float>(scales,scales+d),std::vector<float>(offsets,offsets+d)};
  RowReader rows{d,nullptr,codes,&input_codec};
  rows.unit_rows=state_->config.metric==TieredQMetric::cosine;
  const bool keep_codes=state_->sq8() && !rows.unit_rows;
  state_->build(rows,count,keep_codes ? &input_codec : nullptr);
}

void TieredQIVFFlatIndex::search(const float* queries,size_t rows,size_t k,size_t nprobe,float* scores,
                                 uint32_t* ids) const {
  if (!queries && rows) bad("null query data");
  const size_t d=state_->dimensions;
  if (state_->config.metric!=TieredQMetric::cosine) {
    state_->join(RowReader{d,queries},rows,k,nprobe,nullptr,scores,ids);
    return;
  }
  std::vector<float> normalized(queries,queries+product(rows,d));
  for (size_t r=0;r<rows;++r) normalize(normalized.data()+r*d,d);
  state_->join(RowReader{d,normalized.data()},rows,k,nprobe,nullptr,scores,ids);
}

void TieredQIVFFlatIndex::knn_graph(size_t k,size_t nprobe,float* scores,uint32_t* ids) const {
  // Stored rows are the queries; each excludes itself and answers its own row number.
  state_->join(state_->stored(),state_->size(),k,nprobe,state_->ids.data(),scores,ids);
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
  return s.vectors.capacity()*sizeof(float)+s.codes.capacity()+s.codec.allocated_bytes()+
         s.half_norms.capacity()*sizeof(float)+s.ids.capacity()*sizeof(uint32_t)+
         s.offsets.capacity()*sizeof(size_t)+s.centroids.capacity()*sizeof(float)+
         (s.coarse ? s.coarse->allocated_bytes() : 0);
}
} // namespace tq
