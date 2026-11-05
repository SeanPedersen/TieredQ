// IVF list grouping at build, distance-ranked probing, and persistence. Lists come from the codec's centroids or
// from a separately trained coarse quantizer (any list count). Every list is a contiguous, aligned range of one fast
// representation, so list scans reuse the flat kernels.
#include "tieredq/tieredq.h"
#include "tieredq_fast_scan.h"
#include "tieredq_index_read.h"
#include "tieredq_ivf_coarse.h"
#include "tieredq_fast_fine.h"
#include <bit>
#include <numeric>
namespace dv {
using tieredq_detail::bad;
using tieredq_detail::product;
using tieredq_detail::TopK;
using tieredq_detail::selection_block;
using tieredq_detail::fast_list_alignment;
using tieredq_detail::CoarseQuantizer;
namespace {
// Version 2 adds the coarse quantizer section; version 3 stores the ids as Elias-Fano lists instead of raw.
constexpr uint32_t raw_ids_payload_version=2,ivf_payload_version=3;
constexpr uint32_t padding_id=UINT32_MAX;
// Scored documents per scan call at most; calls start on list starts or chunk multiples, both aligned, and chunk
// multiples are whole lookup blocks.
constexpr size_t ivf_scan_chunk=1024;
static_assert(ivf_scan_chunk%fast_list_alignment==0 && ivf_scan_chunk%selection_block==0);
// Coarse training sample: at least the codec's sample size and ~64 rows per list.
constexpr size_t coarse_minimum_sample=32768,coarse_rows_per_list=64;
// Rows lifted into model space at a time while assigning coarse lists.
constexpr size_t coarse_assign_chunk=65536;
// Documents encoded (coarse lists) or copied into the fast storage at a time during build.
constexpr size_t ivf_build_chunk=8192;

size_t aligned(size_t documents) { return (documents+fast_list_alignment-1)/fast_list_alignment*fast_list_alignment; }

// The nprobe nonempty lists with the highest closeness q.c - |c|^2/2 (nearest in model space); ties go to the lower
// list id.
std::vector<uint32_t> probe_order(const float* closeness,const std::vector<uint32_t>& list_rows,size_t nprobe) {
  std::vector<uint32_t> order;
  for (uint32_t list=0;list<list_rows.size();++list) if (list_rows[list]) order.push_back(list);
  nprobe=std::min(nprobe,order.size());
  std::partial_sort(order.begin(),order.begin()+nprobe,order.end(),[&](uint32_t a,uint32_t b) {
    return closeness[a]>closeness[b] || (closeness[a]==closeness[b] && a<b);
  });
  order.resize(nprobe);
  return order;
}
} // namespace

TieredQIVFIndex::TieredQIVFIndex(TieredQ codec,uint32_t lists,uint64_t seed)
    : documents_(std::move(codec),TieredQIndexMode::fast),
      coarse_lists_(lists),seed_(seed) {
  list_begin_.assign(lists ? lists : documents_.codec().config().num_centroids,0);
  list_rows_.assign(list_begin_.size(),0);
  ids_=tieredq_detail::EliasFanoLists::encode({},list_rows_.size());
}

// Model-space rows [first, first + count): the input itself for inner-product models, otherwise the metric's lift.
const float* TieredQIVFIndex::model_rows(const float* rows,size_t first,size_t count,std::vector<float>& buffer) const {
  const TieredQ& model=codec();
  const float* input=rows+first*model.config().dimensions;
  if (model.config().metric==TieredQMetric::inner_product) return input;
  buffer.resize(product(count,model.model_dimensions()));
  model.lift(input,count,false,buffer.data());
  return buffer.data();
}

void TieredQIVFIndex::train_lists(const float* sample,size_t rows) {
  if (!coarse_lists_) bad("train_lists needs an index with coarse lists");
  if (!sample && rows) bad("null vector data");
  std::vector<float> lifted;
  coarse_=std::make_shared<const CoarseQuantizer>(tieredq_detail::train_coarse(
      model_rows(sample,0,rows,lifted),rows,codec().model_dimensions(),coarse_lists_,seed_));
  fixed_lists_=true;
}

// Coarse lists: k-means on a seeded sample (unless train_lists fixed them), then every row's nearest list.
std::vector<uint32_t> TieredQIVFIndex::coarse_lists(const float* rows,size_t count) {
  const TieredQ& model=codec();
  const size_t d=model.config().dimensions,md=model.model_dimensions();
  std::vector<float> lifted;
  if (!fixed_lists_) {
    const size_t sample_rows=std::min(count,std::max(coarse_minimum_sample,product(coarse_rows_per_list,coarse_lists_)));
    tieredq_detail::TrainingRng rng(seed_);
    const auto picked=rng.choice(count,sample_rows);
    std::vector<float> sample(product(sample_rows,d));
    for (size_t i=0;i<sample_rows;++i) std::copy_n(rows+picked[i]*d,d,sample.data()+i*d);
    const float* model_sample=model_rows(sample.data(),0,sample_rows,lifted);
    coarse_=std::make_shared<const CoarseQuantizer>(tieredq_detail::train_coarse(model_sample,sample_rows,md,coarse_lists_,seed_));
  }
  std::vector<uint32_t> lists(count);
  for (size_t first=0;first<count;first+=coarse_assign_chunk) {
    const size_t n=std::min(coarse_assign_chunk,count-first);
    coarse_->assign(model_rows(rows,first,n,lifted),n,lists.data()+first);
  }
  return lists;
}

void TieredQIVFIndex::build(const float* rows,size_t count) {
  if (count>=padding_id) bad("index exceeds uint32 document ids");
  const TieredQ& model=codec();
  const size_t stride=model.bytes_per_vector(),centroid_byte=model.code_bytes(),lists=list_rows_.size();
  const size_t d=model.config().dimensions;
  if (count && !rows) bad("null vector data");
  // Codec-centroid lists come from the codes; coarse lists are known first, so rows are encoded in list order below.
  std::vector<uint8_t> encoded;
  std::vector<uint32_t> list_of=coarse_lists_ && count ? coarse_lists(rows,count) : std::vector<uint32_t>(count);
  if (!coarse_lists_) {
    encoded=model.encode(rows,count);
    for (size_t row=0;row<count;++row) list_of[row]=encoded[row*stride+centroid_byte];
  }
  std::vector<uint32_t> list_rows(lists,0),list_begin(lists,0);
  for (size_t row=0;row<count;++row) ++list_rows[list_of[row]];
  size_t padded=0;
  for (size_t list=0;list<lists;++list) { list_begin[list]=uint32_t(padded); padded+=aligned(list_rows[list]); }
  if (padded>=padding_id) bad("index exceeds uint32 document ids");
  // source[doc]: the row stored at doc; padding repeats its list's last row (valid codes for every kernel, never
  // selected: scans stop at list_rows). Rows fill each list in input order, so every list's ids ascend.
  std::vector<uint32_t> source(padded),filled(lists,0);
  for (size_t row=0;row<count;++row) {
    const size_t list=list_of[row];
    source[list_begin[list]+filled[list]++]=uint32_t(row);
  }
  for (size_t list=0;list<lists;++list) {
    const size_t first=list_begin[list],live=list_rows[list];
    for (size_t doc=first+live;live && doc<first+aligned(live);++doc) source[doc]=source[first+live-1];
  }
  // Documents go to the fast storage a chunk at a time, so a build holds one chunk of codes besides it.
  TieredQIndex documents(model,TieredQIndexMode::fast);
  std::vector<uint8_t> chunk_codes;
  std::vector<float> chunk_rows;
  for (size_t first=0;first<padded;first+=ivf_build_chunk) {
    const size_t n=std::min(ivf_build_chunk,padded-first);
    if (coarse_lists_) {
      chunk_rows.resize(product(n,d));
      for (size_t i=0;i<n;++i) std::copy_n(rows+size_t(source[first+i])*d,d,chunk_rows.data()+i*d);
      chunk_codes.resize(product(n,stride));
      model.encode(chunk_rows.data(),n,chunk_codes.data());
    } else {
      chunk_codes.resize(product(n,stride));
      for (size_t i=0;i<n;++i) std::copy_n(encoded.data()+size_t(source[first+i])*stride,stride,chunk_codes.data()+i*stride);
    }
    documents.add_encoded(chunk_codes.data(),n);
    if (!first) documents.reserve_rows(padded);
  }
  documents_=std::move(documents);
  ids_=tieredq_detail::EliasFanoLists::encode(list_of,lists);
  list_begin_=std::move(list_begin);list_rows_=std::move(list_rows);size_=count;
}

std::vector<uint32_t> TieredQIVFIndex::list_assignment() const {
  std::vector<uint32_t> lists(size_);
  for (uint32_t list=0;list<list_rows_.size();++list) ids_.for_each(list,[&](size_t,uint32_t id) { lists[id]=list; });
  return lists;
}

// The last list starting at or before document holds it (empty lists share their successor's start). The search is
// branch-free: hits land in arbitrary lists, so a branching search mispredicts at every level.
TieredQIVFIndex::Place TieredQIVFIndex::place_of(size_t document) const {
  const size_t last=list_begin_.size()-1;
  size_t list=0;
  for (size_t step=std::bit_floor(last);step;step/=2) {
    const size_t probe=std::min(list+step,last);
    list=list_begin_[probe]<=document ? probe : list;
  }
  return {uint32_t(list),uint32_t(document-list_begin_[list])};
}

bool TieredQIVFIndex::live(size_t document) const {
  const Place place=place_of(document);
  return document<stored_documents() && place.position<list_rows_[place.list];
}

// Hits are live documents; their rows are decoded together so that the lookups overlap in memory.
std::vector<uint32_t> TieredQIVFIndex::hit_rows(const std::vector<TieredQHit>& hits) const {
  std::vector<uint32_t> lists(hits.size()),positions(hits.size()),rows(hits.size());
  for (size_t hit=0;hit<hits.size();++hit) {
    const Place place=place_of(hits[hit].id);
    lists[hit]=place.list;positions[hit]=place.position;
  }
  ids_.ids(lists.data(),positions.data(),hits.size(),rows.data());
  return rows;
}

size_t TieredQIVFIndex::stored_documents() const {
  return list_begin_.empty() ? 0 : list_begin_.back()+aligned(list_rows_.back());
}

size_t TieredQIVFIndex::storage_bytes() const {
  return documents_.storage_bytes()+(list_begin_.capacity()+list_rows_.capacity())*sizeof(uint32_t)+
      ids_.allocated_bytes()+(coarse_ ? coarse_->allocated_bytes() : 0);
}

std::vector<TieredQHit> TieredQIVFIndex::search(const float* query,size_t k,size_t nprobe,TieredQPruning pruning) const {
  return search_hits(query,nullptr,k,nprobe,pruning);
}

std::vector<TieredQHit> TieredQIVFIndex::search(const float* query,const TieredQQuery& weights,size_t k,size_t nprobe,
                                                TieredQPruning pruning) const {
  return search_hits(query,&weights,k,nprobe,pruning);
}

std::vector<TieredQHit> TieredQIVFIndex::search_hits(const float* query,const TieredQQuery* weights,size_t k,size_t nprobe,
                                                     TieredQPruning pruning) const {
  const size_t found=std::min(k,size_);
  std::vector<float> scores(found);
  std::vector<uint32_t> ids(found);
  search_rows(query,weights,1,k,nprobe,scores.data(),ids.data(),pruning);
  std::vector<TieredQHit> hits;
  for (size_t hit=0;hit<found && ids[hit]!=padding_id;++hit) hits.push_back({scores[hit],ids[hit]});
  return hits;
}

void TieredQIVFIndex::search_batch(const float* queries,size_t rows,size_t k,size_t nprobe,
    float* scores,uint32_t* ids,TieredQPruning pruning) const {
  search_rows(queries,nullptr,rows,k,nprobe,scores,ids,pruning);
}

void TieredQIVFIndex::search_batch(const float* queries,const TieredQQuery* weights,size_t rows,size_t k,size_t nprobe,
    float* scores,uint32_t* ids,TieredQPruning pruning) const {
  if (rows && !weights) bad("null query weights");
  search_rows(queries,weights,rows,k,nprobe,scores,ids,pruning);
}

void TieredQIVFIndex::search_rows(const float* queries,const TieredQQuery* weights,size_t rows,size_t k,size_t nprobe,
    float* scores,uint32_t* ids,TieredQPruning pruning) const {
  if (!(pruning.confidence>=0) || !std::isfinite(pruning.confidence)) bad("invalid pruning");
  if (!rows || !k || !size_) return;
  if (!queries || !scores || !ids) bad("null search buffer");
  const TieredQ& model=codec();
  for (size_t row=0;weights && row<rows;++row)
    if (weights[row].weights.size()!=model.coded_dimensions() || weights[row].centroid_scores.size()!=model.config().num_centroids ||
        weights[row].correction_weights.size()!=model.config().num_centroids)
      bad("query weight shape mismatch");
  const size_t found=std::min(k,size_),probes=std::clamp<size_t>(nprobe,1,lists()),d=model.config().dimensions;
  std::vector<float> lifted;
  const float* model_queries=queries;
  if (model.config().metric!=TieredQMetric::inner_product) {
    lifted.resize(product(rows,model.model_dimensions()));
    model.lift_queries(queries,rows,lifted.data());
    model_queries=lifted.data();
  }
  // Coarse closeness for all queries at once through the GEMM; codec lists use the prepared centroid scores.
  std::vector<float> coarse_closeness;
  if (coarse_) { coarse_closeness.resize(product(rows,lists())); coarse_->closeness(model_queries,rows,coarse_closeness.data()); }
  const auto& cache=*documents_.fast_;
  // Fine-tier models scan their prefix view, which shares the model's projection, so one set of query weights serves
  // the list scans and the rescoring of their candidates.
  const TieredQ& scan_model=documents_.scan_codec();
  const bool fine=bool(documents_.fine_tiers_);
  std::vector<TieredQQuery> fine_weights;
  if (fine && !weights) fine_weights=documents_.fine_query_weights(model_queries,rows);
  const TieredQQuery* query_weights=fine && !weights ? fine_weights.data() : weights;
  const size_t depth=fine ? tieredq_detail::fine_candidate_depth(found,size_) : found;
  std::vector<std::vector<TieredQHit>> candidates(fine ? rows : 0);
  // Hits in stored documents -> the row's ids and scores in the metric's units; unfilled slots are marked.
  auto emit=[&](size_t row,const std::vector<TieredQHit>& hits) {
    const auto rows_of_hits=hit_rows(hits);
    for (size_t hit=0;hit<found;++hit) {
      const bool filled=hit<hits.size();
      scores[row*found+hit]=filled ? hits[hit].score : -std::numeric_limits<float>::infinity();
      ids[row*found+hit]=filled ? rows_of_hits[hit] : padding_id;
    }
    documents_.finish_scores(queries+row*d,scores+row*found,hits.size());
  };
#ifdef DV_TIEREDQ_X86
  const size_t code_prefix_tiers=pruning.prefix_tiers==TieredQPruning::automatic_prefix ?
      tieredq_detail::fast_range_prefix_tiers : pruning.prefix_tiers;
  const bool range=tieredq_detail::fast_scan_backend(cache)==tieredq_detail::FastScanBackend::avx512 &&
      tieredq_detail::fast_range_checkpoint_supported(cache,scan_model.tiers(),code_prefix_tiers);
  const size_t prefix_rank=range ? scan_model.tiers()[code_prefix_tiers].start : 0;
  const auto segments=range ? tieredq_detail::code_table_segments(cache) : std::vector<tieredq_detail::CodeTableSegment>();
#endif
  const bool lookup_range=tieredq_detail::lookup_range_scan_supported(cache,scan_model.tiers(),pruning.prefix_tiers,
      pruning.confidence);
  tieredq_detail::with_fast_scan(cache,scan_model,model_queries,rows,[&](const auto& prepared,size_t,auto scan) {
    const size_t workers=std::max<size_t>(1,std::min(rows,size_t(omp_get_max_threads())));
    std::exception_ptr failure; std::mutex failure_mutex;
#pragma omp parallel for schedule(dynamic) num_threads(workers) if(workers>1 && !omp_in_parallel())
    for (size_t row=0;row<rows;++row) {
      try {
        std::vector<float> codec_closeness;
        const float* closeness=coarse_closeness.data()+row*lists();
        if (!coarse_) {
          const auto& centroid=tieredq_detail::prepared_centroid_scores(prepared[row]);
          codec_closeness.resize(lists());
          for (size_t list=0;list<lists();++list) codec_closeness[list]=centroid[list]-model.centroid_half_norms_[list];
          closeness=codec_closeness.data();
        }
        TopK top(depth);
        std::vector<float> chunk(ivf_scan_chunk);
        // Rebound inside the OpenMP region: Clang rejects the nested generic lambda naming the enclosing local.
        const size_t probed_lists=probes;
        // Lists in probe order, nearest first, so the running top-k threshold tightens early for pruned scans.
        auto scan_lists=[&](auto score) {
          for (const uint32_t list:probe_order(closeness,list_rows_,probed_lists)) {
            const size_t first=list_begin_[list],live=list_rows_[list];
            // Chunks end on absolute chunk boundaries, so block kernels never rescan a block across calls.
            for (size_t doc=first,end=first+live;doc<end;) {
              const size_t documents=std::min(end,(doc/ivf_scan_chunk+1)*ivf_scan_chunk)-doc;
              score(doc,documents,chunk.data(),top.threshold());
              for (size_t block=0;block<documents;block+=selection_block)
                top.add(chunk.data()+block,std::min(selection_block,documents-block),doc+block);
              doc+=documents;
            }
          }
        };
        bool scanned=false;
#ifdef DV_TIEREDQ_X86
        // Range checkpoint: documents whose prefix score bound cannot reach the threshold skip their suffix tiers
        // and score -inf; every other score is exact, so with the exact bound hits equal the unpruned scan.
        if constexpr (requires { prepared[row].codes.weights; }) if (range) {
          const auto& query=prepared[row].codes;
          // Bind inside the OpenMP region so Clang can capture this in the nested scan lambda.
          const auto& query_segments=segments;
          const tieredq_detail::CentroidTiles centroids(1,[&](size_t)->const TieredQQuery& { return query; });
          tieredq_detail::FastRangeCheckpoint checkpoint{prefix_rank,tieredq_detail::prepare_fast_range_bounds(cache,scan_model.tiers(),
              query,code_prefix_tiers,pruning.confidence),0};
          scan_lists([&](size_t first,size_t documents,float* out,float threshold) {
            checkpoint.threshold=threshold;
            tieredq_detail::scan_range_code_blocks_avx512(cache,query_segments,query.weights.data(),centroids.scores(0),
                centroids.corrections(0),first,documents,out,checkpoint);
          });
          scanned=true;
        }
#endif
        // The same checkpoint on nibble byte tables: rejected chunks score -inf, the rest equal the unpruned scan.
        if constexpr (requires { prepared[row].lookup.tables; }) if (lookup_range) {
          const auto& query=prepared[row].lookup;
          const auto bounds=tieredq_detail::prepare_lookup_range_bounds(cache,scan_model.tiers(),query,pruning.prefix_tiers,
              pruning.confidence);
          scan_lists([&](size_t first,size_t documents,float* out,float threshold) {
            tieredq_detail::scan_range_lookup(cache,query,first,documents,out,{bounds,threshold});
          });
          scanned=true;
        }
        if (!scanned) scan_lists([&](size_t first,size_t documents,float* out,float) {
          scan(&prepared[row],1,first,documents,out,documents);
        });
        if (fine) candidates[row]=top.finish();
        else emit(row,top.finish());
      } catch (...) {
        std::lock_guard lock(failure_mutex);
        if (!failure) failure=std::current_exception();
      }
    }
    if (failure) std::rethrow_exception(failure);
  },query_weights);
  if (!fine) return;
  documents_.rescore_candidates(query_weights,candidates,found);
  for (size_t row=0;row<rows;++row) emit(row,candidates[row]);
}

// Payload: version, list count, coarse flag (+ seed, trained flag and centroids), size, per-list offset and rows,
// Elias-Fano id words, fast payload (with the fine rows of fine-tier models). An unbuilt coarse index has no centroids
// yet.
void TieredQIVFIndex::write_lists(tieredq_detail::Writer& writer) const {
  writer.u32(ivf_payload_version);writer.u32(uint32_t(lists()));writer.u32(coarse_lists_ ? 1 : 0);
  if (coarse_lists_) {
    writer.u64(seed_);writer.u32(coarse_ ? 1 : 0);
    if (coarse_) for (float value:coarse_->centroids()) writer.f32(value);
  }
  writer.u64(size_);
  for (size_t list=0;list<lists();++list) { writer.u32(list_begin_[list]);writer.u32(list_rows_[list]); }
  writer.u64(ids_.words().size());
  for (uint64_t word:ids_.words()) writer.u64(word);
}

std::vector<uint8_t> TieredQIVFIndex::serialize() const {
  tieredq_detail::Writer writer;
  const auto model=codec().serialize();
  writer.u64(model.size());
  writer.data.insert(writer.data.end(),model.begin(),model.end());
  write_lists(writer);
  const auto fast=documents_.fast_payload();
  writer.data.insert(writer.data.end(),fast.begin(),fast.end());
  return std::move(writer.data);
}

// Reads the list section into an index built from the payload's model and validates it.
template<class Reader>
TieredQIVFIndex TieredQIVFIndex::read_lists(TieredQ model,Reader& reader) {
  const uint32_t version=reader.u32();
  if (version!=raw_ids_payload_version && version!=ivf_payload_version) bad("unsupported IVF payload version");
  const uint32_t lists=reader.u32(),coarse=reader.u32();
  if (coarse>1 || (!coarse && lists!=model.config().num_centroids) || !lists) bad("invalid IVF list count");
  const uint64_t seed=coarse ? reader.u64() : 0;
  const uint32_t trained=coarse ? reader.u32() : 0;
  if (trained>1) bad("invalid IVF coarse flag");
  TieredQIVFIndex index(std::move(model),coarse ? lists : 0,seed);
  if (trained) {
    const size_t md=index.codec().model_dimensions();
    std::vector<float> centroids(product(lists,md));
    tieredq_detail::read_f32_array(reader,centroids.data(),centroids.size());
    index.coarse_=std::make_shared<const CoarseQuantizer>(CoarseQuantizer::from(centroids,lists,md));
    index.fixed_lists_=true;
  }
  index.size_=size_t(reader.u64());
  size_t padded=0,live=0;
  for (size_t list=0;list<lists;++list) {
    index.list_begin_[list]=reader.u32();index.list_rows_[list]=reader.u32();
    if (index.list_begin_[list]!=padded) bad("invalid IVF list offsets");
    padded+=aligned(index.list_rows_[list]);live+=index.list_rows_[list];
  }
  if (live!=index.size_) bad("invalid IVF list sizes");
  if (version==raw_ids_payload_version) index.read_raw_ids(reader,padded);
  else {
    const uint64_t words=reader.u64();
    if (words>reader.remaining()/sizeof(uint64_t)) bad("truncated IVF ids");
    std::vector<uint64_t> stream(words);
    for (auto& word:stream) word=reader.u64();
    index.ids_=tieredq_detail::EliasFanoLists::decode(index.list_rows_,std::move(stream));
  }
  index.documents_.read_fast_payload(reader,padded);
  return index;
}

template<class Reader>
void TieredQIVFIndex::read_raw_ids(Reader& reader,size_t padded) {
  if (reader.u64()!=padded) bad("invalid IVF list sizes");
  std::vector<uint32_t> list_of(size_,padding_id);
  for (uint32_t list=0;list<lists();++list) {
    const size_t live=list_rows_[list];
    for (size_t position=0,previous=0;position<aligned(live);++position) {
      const uint32_t id=reader.u32();
      if (position>=live) { if (id!=padding_id) bad("invalid IVF ids"); continue; }
      // Elias-Fano keeps ids sorted: a list stored in another order would no longer match its documents.
      if (id>=size_ || list_of[id]!=padding_id || (position && id<=previous)) bad("invalid IVF ids");
      list_of[id]=list;previous=id;
    }
  }
  ids_=tieredq_detail::EliasFanoLists::encode(list_of,lists());
}

TieredQIVFIndex TieredQIVFIndex::deserialize(const std::vector<uint8_t>& bytes) {
  tieredq_detail::Reader reader{bytes};
  const uint64_t model_length=reader.u64();
  if (model_length>reader.remaining()) bad("truncated IVF model");
  std::vector<uint8_t> model(model_length);
  reader.bytes(model.data(),model.size());
  return read_lists(TieredQ::deserialize(model),reader);
}

void TieredQIVFIndex::save(const std::string& path) const {
  tieredq_detail::Writer writer;
  write_lists(writer);
  const auto fast=documents_.fast_payload();
  writer.data.insert(writer.data.end(),fast.begin(),fast.end());
  tieredq_detail::save_file(path,codec().serialize(),writer.data,stored_documents(),tieredq_detail::ivf_index_file_kind);
}

TieredQIVFIndex TieredQIVFIndex::load(const std::string& path) {
  auto file=tieredq_detail::open_file(path,tieredq_detail::ivf_index_file_kind);
  tieredq_detail::StreamReader reader(file.in,size_t(file.header.payload_length));
  auto parse=[&] {
    auto model=TieredQ::deserialize(file.model);
    file.model=std::vector<uint8_t>();
    auto index=read_lists(std::move(model),reader);
    if (index.stored_documents()!=file.header.rows) bad("IVF row count mismatch");
    return index;
  };
  try {
    auto index=parse();
    if (!reader.verify(file.header.payload_hash)) bad("checksum mismatch");
    return index;
  } catch (...) {
    if (!reader.verify(file.header.payload_hash)) bad("checksum mismatch");
    throw;
  }
}
} // namespace dv
