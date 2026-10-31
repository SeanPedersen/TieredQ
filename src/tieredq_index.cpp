// Index ownership, append transactions and exhaustive exact/fast query execution.
#include "tieredq/tieredq.h"
#include "tieredq_io.h"
#include "tieredq_layout.h"
#include "tieredq_search.h"
#include "tieredq_search_fast.h"
#include "tieredq_fast_storage.h"
#include "tieredq_fast_scan.h"
namespace dv {
using tieredq_detail::Layout;
using tieredq_detail::bad;
using tieredq_detail::product;
using tieredq_detail::TopK;
using tieredq_detail::better_hit;
using tieredq_detail::selection_block;
using tieredq_detail::prepare_search_queries;
using tieredq_detail::save_file;
using tieredq_detail::load_file;
std::shared_ptr<tieredq_detail::FastScanCache> TieredQIndex::make_fast_storage(
    const uint8_t* rows,size_t count,bool lookup,int level_max) const {
  if (lookup) {
    auto projection=tieredq_detail::build_fast_query_projection(codec_.model_dimensions(),
        tieredq_detail::FastQueryBasis::from(codec_),codec_.mean_);
    return tieredq_detail::build_fast_lookup_cache(codec_.levels_,codec_.tiers_,std::move(projection),
        count,codec_.bytes_per_vector(),codec_.code_bytes(),codec_.config().num_centroids,rows);
  }
  return tieredq_detail::build_fast_scan_cache(codec_.levels_,codec_.tiers_,codec_.coded_dimensions(),
      count,codec_.bytes_per_vector(),codec_.code_bytes(),codec_.config().num_centroids,level_max,rows);
}
std::shared_ptr<tieredq_detail::FastScanCache> TieredQIndex::make_fast_storage(const uint8_t* rows,size_t count) const {
  if (tieredq_detail::fast_code_lookup_supported(codec_.tiers_))
    return make_fast_storage(rows,count,false,tieredq_detail::fast_code_lookup_levels);
#ifdef DV_TIEREDQ_NEON
  return make_fast_storage(rows,count,false,tieredq_detail::arm_fast_level_max);
#else
  return make_fast_storage(rows,count,tieredq_detail::fast_lookup_supported(codec_.tiers_),tieredq_detail::x86_fast_level_max);
#endif
}
bool TieredQIndex::native_fast_scan_available() {
  return tieredq_detail::native_fast_scan_available();
}
bool TieredQIndex::uses_scalar_fast_scan() const {
  return fast_ && tieredq_detail::fast_scan_backend(*fast_)==tieredq_detail::FastScanBackend::scalar;
}
size_t TieredQIndex::payload_bytes() const {
  if (!count_) return 0;
  if (fast_mode_) return count_*bytes_per_vector();
  return count_*codec_.bytes_per_vector();
}
size_t TieredQIndex::storage_bytes() const {
  return payload_.capacity()+(fast_ ? tieredq_detail::fast_storage_bytes(*fast_) : 0);
}
size_t TieredQIndex::bytes_per_vector() const {
  if (!fast_mode_) return codec_.bytes_per_vector();
  if (fast_) return tieredq_detail::fast_vector_bytes(*fast_);
  return tieredq_detail::fast_vector_bytes(*make_fast_storage(nullptr,0));
}
// Rows encoded at a time when filling an empty fast index, which then holds one chunk of payload rows next to its
// storage instead of all of them.
constexpr size_t add_chunk_rows=16384;
void TieredQIndex::add(const float* rows,size_t count) {
  if (count>UINT32_MAX-size()) bad("index exceeds uint32 document ids");
  if (!fast_mode_ || size() || count<=add_chunk_rows) {
    const auto encoded=codec_.encode(rows,count);
    add_encoded(encoded.data(),count);
    return;
  }
  const size_t d=codec_.config().dimensions;
  try {
    for (size_t first=0;first<count;first+=add_chunk_rows) {
      const size_t n=std::min(add_chunk_rows,count-first);
      const auto encoded=codec_.encode(rows+first*d,n);
      add_encoded(encoded.data(),n);
      if (!first) tieredq_detail::reserve_fast_rows(*fast_,count);
    }
  } catch (...) { fast_.reset(); count_=0; throw; }
}
void TieredQIndex::add_encoded(const uint8_t* encoded,size_t count) {
  if (count>UINT32_MAX-size()) bad("index exceeds uint32 document ids");
  if (fast_mode_) {
    if (!count) return;
    if (!fast_) fast_=make_fast_storage(encoded,count);
    else {
      auto storage=fast_.use_count()==1 ? fast_ : tieredq_detail::copy_fast_storage(*fast_);
      tieredq_detail::append_fast_rows(*storage,codec_.levels_,codec_.tiers_,codec_.bytes_per_vector(),
          codec_.code_bytes(),codec_.config().num_centroids,encoded,count);
      fast_=std::move(storage);
    }
    count_+=count;
    return;
  }
  const size_t stride=codec_.bytes_per_vector(), total=size()+count;
  payload_.resize(product(total/16+(total%16!=0),product(16,stride)),0);
  for (size_t i=0;i<count;++i) {
    const size_t id=size()+i, block=(id/16)*16*stride, lane=id%16;
    for (size_t byte=0;byte<stride;++byte) payload_[block+byte*16+lane]=encoded[i*stride+byte];
  }
  count_=total;
  if (count) fast_.reset();
}
namespace {
// Model-space queries: the input itself for inner-product models, otherwise the metric's lift.
struct ModelQueries {
  std::vector<float> lifted;
  const float* data;
  ModelQueries(const TieredQ& codec,const float* queries,size_t rows) : data(queries) {
    if (codec.config().metric==TieredQMetric::inner_product) return;
    lifted.resize(product(rows,codec.model_dimensions()));
    codec.lift_queries(queries,rows,lifted.data());
    data=lifted.data();
  }
};
} // namespace
void TieredQIndex::finish_scores(const float* query,float* scores,size_t count) const {
  const auto [scale,offset]=codec_.score_map(query);
  if (scale==1 && offset==0) return;
  for (size_t i=0;i<count;++i) scores[i]=float(scale*scores[i]+offset);
}
std::vector<TieredQHit> TieredQIndex::search(const float* vector,size_t k,TieredQKernel kernel) const {
  if (!k || !size()) return {};
  if (fast_mode_ && kernel!=TieredQKernel::automatic) bad("fast-only search does not support exact kernel overrides");
  const ModelQueries query(codec_,vector,1);
  auto hits=fast_mode_ ? search_fast_model(query.data,k) : search_model(query.data,k,kernel);
  for (auto& hit:hits) finish_scores(vector,&hit.score,1);
  return hits;
}
std::vector<TieredQHit> TieredQIndex::search_model(const float* vector,size_t k,TieredQKernel kernel) const {
  auto q=codec_.model_query(vector); k=std::min(k,size());
  TopK top(k);
  std::array<float,selection_block> scores;
  const size_t stride=codec_.bytes_per_vector();
  for (size_t begin=0;begin<size();begin+=scores.size()) {
    const size_t n=std::min(scores.size(),size()-begin);
    codec_.score_planes(payload_.data()+begin*stride,n,q,scores.data(),kernel,false);
    top.add(scores.data(),n,begin);
  }
  return top.finish();
}
TieredQIndex::TieredQIndex(TieredQ codec,TieredQIndexMode mode)
    : codec_(std::move(codec)),fast_mode_(mode==TieredQIndexMode::fast) {
  if (mode!=TieredQIndexMode::fast && mode!=TieredQIndexMode::full)
    throw std::invalid_argument("TieredQ: invalid index mode");
}
TieredQIndex::~TieredQIndex() = default;
void TieredQIndex::build_fast_scan() {
  if (fast_mode_) return;
  fast_.reset();
#if defined(DV_TIEREDQ_NEON) || defined(DV_TIEREDQ_X86)
  if (!count_ || !tieredq_detail::fast_scan_available()) return;
  const size_t stride=codec_.bytes_per_vector();
  std::vector<uint8_t> rows(product(count_,stride));
  codec_.unpack(payload_.data(),count_,rows.data());
  fast_=make_fast_storage(rows.data(),count_);
#endif
}
namespace {
#ifdef DV_TIEREDQ_X86
// Keep checkpoint preparation and selection scheduling separate from CPU/layout dispatch.
template<class Finish>
void fast_range_search(const tieredq_detail::FastScanCache& cache,const TieredQ& codec,
    const float* query,size_t k,Finish finish) {
  const auto prepared=codec.model_query_weights(query,1);
  const auto bounds=tieredq_detail::prepare_fast_range_bounds(cache,codec.tiers(),prepared[0]);
  const size_t prefix_rank=codec.tiers()[tieredq_detail::fast_range_prefix_tiers].start;
  auto scan=[&](const TieredQQuery* queries,size_t,size_t first,size_t documents,
      float* scores,size_t,TopK* top) {
    tieredq_detail::scan_fast_range_avx512(cache,*queries,bounds,prefix_rank,first,documents,scores,top->threshold());
  };
  tieredq_detail::search_fast_tiles(prepared,cache.count,k,1,scan,finish);
}
#endif
template<class Finish>
void fast_search(const tieredq_detail::FastScanCache& cache,const TieredQ& codec,
    const float* queries,size_t rows,size_t k,Finish finish) {
#ifdef DV_TIEREDQ_X86
  if (rows==1 && tieredq_detail::fast_scan_backend(cache)==tieredq_detail::FastScanBackend::avx512 &&
      tieredq_detail::fast_range_checkpoint_supported(cache,codec.tiers())) {
    fast_range_search(cache,codec,queries,k,finish);
    return;
  }
#endif
  tieredq_detail::with_fast_scan(cache,codec,queries,rows,[&](const auto& prepared,size_t query_tile,auto scan) {
    tieredq_detail::search_fast_tiles(prepared,cache.count,k,query_tile,scan,finish);
  });
}
} // namespace
std::vector<TieredQHit> TieredQIndex::search_fast(const float* vector,size_t k) const {
  if (!k || !size()) return {};
  const ModelQueries query(codec_,vector,1);
  auto hits=search_fast_model(query.data,k);
  for (auto& hit:hits) finish_scores(vector,&hit.score,1);
  return hits;
}
std::vector<TieredQHit> TieredQIndex::search_fast_model(const float* vector,size_t k) const {
  if (!fast_) return fast_mode_ ? std::vector<TieredQHit>{} : search_model(vector,k,TieredQKernel::automatic);
  std::vector<TieredQHit> result;
  fast_search(*fast_,codec_,vector,1,std::min(k,size()),
      [&](size_t,std::vector<TieredQHit> hits) { result=std::move(hits); });
  return result;
}
void TieredQIndex::search_batch_fast(const float* queries,size_t rows,size_t k,
    float* scores,uint32_t* ids) const {
  if (!rows || !k || !size()) return;
  if (!queries || !scores || !ids) bad("null search buffer");
  const ModelQueries model(codec_,queries,rows);
  search_batch_fast_model(model.data,rows,k,scores,ids);
  const size_t found=std::min(k,size()),d=codec_.config().dimensions;
  for (size_t row=0;row<rows;++row) finish_scores(queries+row*d,scores+row*found,found);
}
void TieredQIndex::search_batch_fast_model(const float* queries,size_t rows,size_t k,
    float* scores,uint32_t* ids) const {
  if (!fast_) { search_batch_model(queries,rows,k,scores,ids); return; }
  const size_t found=std::min(k,size());
  fast_search(*fast_,codec_,queries,rows,found,[&](size_t row,const std::vector<TieredQHit>& hits) {
    for (size_t hit=0;hit<hits.size();++hit) {
      scores[row*found+hit]=hits[hit].score;
      ids[row*found+hit]=hits[hit].id;
    }
  });
}
void TieredQIndex::search_batch(const float* queries,size_t rows,size_t k,
    float* scores,uint32_t* ids) const {
  if (!rows || !k || !size()) return;
  if (!queries || !scores || !ids) bad("null search buffer");
  const ModelQueries model(codec_,queries,rows);
  if (fast_mode_) search_batch_fast_model(model.data,rows,k,scores,ids);
  else search_batch_model(model.data,rows,k,scores,ids);
  const size_t found=std::min(k,size()),d=codec_.config().dimensions;
  for (size_t row=0;row<rows;++row) finish_scores(queries+row*d,scores+row*found,found);
}
void TieredQIndex::search_batch_model(const float* queries,size_t rows,size_t k,
    float* scores,uint32_t* ids) const {
  const size_t dimensions=codec_.model_dimensions(),stride=codec_.bytes_per_vector();
  const size_t found=std::min(k,size());
  const size_t available=std::max<size_t>(1,size_t(omp_get_max_threads()));
  const size_t chunk=std::min<size_t>(4,rows),query_chunks=(rows+chunk-1)/chunk;
  size_t segments=1;
  if (rows>1 && query_chunks<2*available)
    segments=std::min<size_t>(16,(2*available+query_chunks-1)/query_chunks);
  const size_t total_blocks=(size()+selection_block-1)/selection_block;
  size_t segment_blocks=(total_blocks+segments-1)/segments;
  segments=(total_blocks+segment_blocks-1)/segment_blocks;
  const size_t tasks=query_chunks*segments,workers=std::min(tasks,available);
  auto prepared=prepare_search_queries(queries,rows,dimensions,
      [&](const float* vector) { return codec_.model_query(vector); });
  std::vector<float> partial_scores(tasks*chunk*found);
  std::vector<uint32_t> partial_ids(tasks*chunk*found),partial_counts(tasks*chunk);
  std::exception_ptr failure; std::mutex failure_mutex;
#pragma omp parallel for schedule(dynamic) num_threads(workers) if(tasks>1 && !omp_in_parallel())
  for (size_t task=0;task<tasks;++task) {
    try {
      const size_t first=(task/segments)*chunk,count=std::min(chunk,rows-first);
      const size_t begin=std::min((task%segments)*segment_blocks*selection_block,size());
      const size_t end=std::min(begin+segment_blocks*selection_block,size());
      std::vector<TopK> heaps; heaps.reserve(count);
      for (size_t i=0;i<count;++i) heaps.emplace_back(found);
      std::vector<float> block_scores(count*selection_block);
      for (size_t doc=begin;doc<end;doc+=selection_block) {
        const size_t n=std::min(selection_block,end-doc);
        codec_.score_planes_batch(payload_.data()+doc*stride,n,prepared.data()+first,count,
                                  block_scores.data(),n,TieredQKernel::automatic,false);
        for (size_t i=0;i<count;++i) heaps[i].add(block_scores.data()+i*n,n,doc);
      }
      for (size_t i=0;i<count;++i) {
        const auto hits=heaps[i].finish();
        const size_t offset=(task*chunk+i)*found;
        for (size_t hit=0;hit<hits.size();++hit) {
          partial_scores[offset+hit]=hits[hit].score;
          partial_ids[offset+hit]=hits[hit].id;
        }
        partial_counts[task*chunk+i]=uint32_t(hits.size());
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(failure_mutex);
      if (!failure) failure=std::current_exception();
    }
  }
  if (failure) std::rethrow_exception(failure);
#pragma omp parallel for schedule(static) num_threads(std::min(rows,available)) if(rows>1 && !omp_in_parallel())
  for (size_t row=0;row<rows;++row) {
    const size_t chunk_index=row/chunk,local=row%chunk;
    std::vector<TieredQHit> hits; hits.reserve(segments*found);
    for (size_t segment=0;segment<segments;++segment) {
      const size_t task=chunk_index*segments+segment,offset=(task*chunk+local)*found;
      for (size_t hit=0;hit<partial_counts[task*chunk+local];++hit)
        hits.push_back({partial_scores[offset+hit],partial_ids[offset+hit]});
    }
    std::sort(hits.begin(),hits.end(),better_hit);
    for (size_t hit=0;hit<found;++hit) {
      scores[row*found+hit]=hits[hit].score;
      ids[row*found+hit]=hits[hit].id;
    }
  }
}
} // namespace dv
