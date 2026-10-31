// Issue #6: TieredQ fast-mode IVF (TieredQIVFIndex) vs RaBitQ's IVF on the same partition, recall/storage/QPS per nprobe.
// The partition is TieredQ's k-means at NLIST centroids (training sample of up to 32,768 rows, seed 0): RaBitQ gets
// every row's TieredQ list and each list's row mean as its centroid, so both scan identical lists; only the codec
// and the list ranking (TieredQ: query-centroid score, RaBitQ: its own centroid distance) differ. RaBitQ encodes
// with its accurate configuration (faster_config off) and high-accuracy scoring. Disk is the serialized index.
// Modes: tieredq CODE_BITS | rabitq BITS_PER_DIM | tieredq-match TARGET_DISK_BITS (largest code budget whose disk
// bits/vector fit, storage only) | tieredq-profile CODE_BITS (single-query loops only, for perf; prints qps_1t_b1). Rows: codec budget nlist nprobe disk_bits code_bits recall qps_1t_b1 qps_8t_b32
// build_s list_hash. LISTS gives TieredQ a coarse quantizer with that many lists (the codec keeps NLIST centroids) and
// RaBitQ the same partition. PRUNE_Z and PRUNE_PREFIX set TieredQPruning. MAX_BITS caps TieredQ bits per coordinate (TieredQConfig::max_bits). Data as ab_eval: DATA_DIR/{base.f32,queries.f32,truth.u32}, inner-product truth.
// RERANK=K1,K2,... adds exact rerank: each codec returns its top-k' and one shared rescoring (inner product on the raw
// float base, as the truth) keeps the top 10. Rows then read: codec budget nlist nprobe disk_bits code_bits rerank_k
// cand_recall recall reads fp32_bits qps_1t_b1 qps_8t_b32 build_s list_hash, where cand_recall counts the true top 10
// among the k' candidates, reads the full vectors rescored per query and fp32_bits the rerank tier per vector (not in
// disk_bits). QPS includes the rerank. RESCORE=T1,T2,... rescores the same candidates with each rerank tier: fp32 (the
// default, exact) | fp16 | int8 (per-dimension min/max) | tieredq:B (flat TieredQ trained at B*d code bits) | rabitq:B
// (RaBitQ B-bit codes on its IVF's rotation and list centroids, high-accuracy estimator: its 1-bit IVF code plus
// B-1 ex-bits). tieredq-residual:B refines a flat TieredQ prefix with a residual TieredQ at B*d total bits;
// tieredq-refinement:B nests a TieredQRefinement suffix on the flat prefix (B*d total code bits). PREFIX_BITS sets
// the prefix budget of both (default 1416). Rerank rows end with rescore and rescore_bytes (the tier per vector; fp32_bits stays the FP32 size).
// Mode tieredq-nesting B1,B2,... (storage only) prints whether each budget's model refines the previous one.
// Usage: issue6_ivf_eval MODE BUDGET DATA_DIR D NLIST TIMED(0|1) NPROBE...   (RaBitQ needs BENCH_SEED set)
#include "tieredq/tieredq.h"
#include "tieredq/tieredq_refinement.h"
#include <rabitqlib/index/estimator.hpp>
#include <rabitqlib/index/ivf/ivf.hpp>
#include <rabitqlib/index/query.hpp>
#include <rabitqlib/quantization/rabitq.hpp>
#include <rabitqlib/utils/rotator.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <omp.h>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>
namespace {
template<class T> std::vector<T> load(const std::string& path) {
  std::ifstream in(path,std::ios::binary|std::ios::ate);
  std::vector<T> out(size_t(in.tellg())/sizeof(T));
  in.seekg(0); in.read(reinterpret_cast<char*>(out.data()),std::streamsize(out.size()*sizeof(T)));
  return out;
}
constexpr size_t top_k=10,training_limit=32768,batch=32,threads=8,repeats=3,match_resolution=8;
constexpr uint64_t seed=0;
struct Data { std::vector<float> base,queries; std::vector<uint32_t> truth; size_t d,n,nq; };
double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
template<class F> double best_seconds(F run) {
  double best=1e30;
  for (size_t r=0;r<repeats;++r) { const auto start=std::chrono::steady_clock::now(); run(); best=std::min(best,seconds_since(start)); }
  return best;
}
// Fraction of the true top 10 among each query's first `stride` ids.
double recall(const Data& data,const std::vector<uint32_t>& ids,size_t stride=top_k) {
  size_t hits=0;
  for (size_t q=0;q<data.nq;++q) for (size_t j=0;j<stride;++j)
    hits+=std::count(data.truth.begin()+q*top_k,data.truth.begin()+(q+1)*top_k,ids[q*stride+j]);
  return double(hits)/double(data.nq*top_k);
}
std::vector<size_t> rerank_depths() {
  std::vector<size_t> depths;
  const char* list=std::getenv("RERANK");
  for (const char* at=list;at && *at;) { char* end; depths.push_back(std::strtoul(at,&end,10)); at=*end ? end+1 : end; }
  return depths;
}
// Independent lanes so the reduction vectorizes without reassociation flags.
float inner_product(const float* a,const float* b,size_t d) {
  constexpr size_t lanes=16;
  std::array<float,lanes> sums{};
  size_t j=0;
  for (;j+lanes<=d;j+=lanes) for (size_t l=0;l<lanes;++l) sums[l]+=a[j+l]*b[j+l];
  for (;j<d;++j) sums[0]+=a[j]*b[j];
  return std::accumulate(sums.begin(),sums.end(),0.0f);
}
// A rerank tier: scores (larger is better) of `count` valid base ids for one query; called from parallel workers.
struct Rescorer {
  std::string name;
  double bytes;  // per vector
  std::function<void(const float*,const uint32_t*,size_t,float*)> score;
};
Rescorer fp32_rescorer(const Data& data) {
  return {"fp32",4.0*double(data.d),[&data](const float* query,const uint32_t* ids,size_t count,float* scores) {
    for (size_t i=0;i<count;++i) scores[i]=inner_product(query,data.base.data()+size_t(ids[i])*data.d,data.d);
  }};
}
// Rescoring shared by every codec: top_k of `count` candidates (UINT32_MAX = empty slot) into out; returns reads.
size_t rerank(const Rescorer& tier,const float* query,const uint32_t* candidates,size_t count,uint32_t* out) {
  thread_local std::vector<uint32_t> ids;
  thread_local std::vector<float> scores;
  thread_local std::vector<std::pair<float,uint32_t>> scored;
  ids.clear(); scored.clear();
  for (size_t i=0;i<count;++i) if (candidates[i]!=UINT32_MAX) ids.push_back(candidates[i]);
  scores.resize(ids.size());
  tier.score(query,ids.data(),ids.size(),scores.data());
  for (size_t i=0;i<ids.size();++i) scored.emplace_back(scores[i],ids[i]);
  const size_t kept=std::min(top_k,scored.size());
  std::partial_sort(scored.begin(),scored.begin()+kept,scored.end(),[](const auto& a,const auto& b) { return a.first>b.first; });
  for (size_t i=0;i<top_k;++i) out[i]=i<kept ? scored[i].second : UINT32_MAX;
  return scored.size();
}
// Up to training_limit rows in a seeded shuffle.
std::vector<float> training_sample(const Data& data) {
  const size_t count=std::min(data.n,training_limit);
  std::vector<size_t> order(data.n); std::iota(order.begin(),order.end(),0);
  std::mt19937_64 rng(seed); std::shuffle(order.begin(),order.end(),rng);
  std::vector<float> sample(count*data.d);
  for (size_t i=0;i<count;++i) std::copy_n(data.base.data()+order[i]*data.d,data.d,sample.data()+i*data.d);
  return sample;
}
dv::TieredQ train(const Data& data,uint32_t code_bits,size_t nlist,bool rotate_tiers=true) {
  const auto sample=training_sample(data);
  const size_t count=sample.size()/data.d;
  auto config=dv::TieredQConfig::target_code_bits(uint32_t(data.d),code_bits);
  config.num_centroids=uint32_t(nlist);
  if (const char* max_bits=std::getenv("MAX_BITS")) config.max_bits=uint32_t(std::stoul(max_bits));
  config.rotate_tiers=rotate_tiers;
  return dv::TieredQ::train(config,sample.data(),count,seed);
}

Rescorer fp16_rescorer(const Data& data) {
  auto codes=std::make_shared<std::vector<_Float16>>(data.base.size());
  std::transform(data.base.begin(),data.base.end(),codes->begin(),[](float x) { return static_cast<_Float16>(x); });
  return {"fp16",2.0*double(data.d),[&data,codes](const float* query,const uint32_t* ids,size_t count,float* scores) {
    thread_local std::vector<float> row;
    row.resize(data.d);
    for (size_t i=0;i<count;++i) {
      std::copy_n(codes->data()+size_t(ids[i])*data.d,data.d,row.data());
      scores[i]=inner_product(query,row.data(),data.d);
    }
  }};
}
// Per-dimension min/max scalar quantization; a float query scores uint8 codes as q.(low + scale * code).
Rescorer int8_rescorer(const Data& data) {
  constexpr float levels=255;
  std::vector<float> low(data.d,INFINITY),high(data.d,-INFINITY);
  for (size_t i=0;i<data.n;++i) for (size_t j=0;j<data.d;++j) {
    low[j]=std::min(low[j],data.base[i*data.d+j]); high[j]=std::max(high[j],data.base[i*data.d+j]);
  }
  std::vector<float> scale(data.d);
  for (size_t j=0;j<data.d;++j) scale[j]=high[j]>low[j] ? (high[j]-low[j])/levels : 1;
  auto codes=std::make_shared<std::vector<uint8_t>>(data.n*data.d);
#pragma omp parallel for
  for (size_t i=0;i<data.n;++i) for (size_t j=0;j<data.d;++j)
    (*codes)[i*data.d+j]=uint8_t(std::clamp(std::nearbyint((data.base[i*data.d+j]-low[j])/scale[j]),0.0f,levels));
  return {"int8",double(data.d),[&data,codes,low,scale](const float* query,const uint32_t* ids,size_t count,float* scores) {
    thread_local std::vector<float> weights,row;
    weights.resize(data.d); row.resize(data.d);
    for (size_t j=0;j<data.d;++j) weights[j]=query[j]*scale[j];
    const float offset=inner_product(query,low.data(),data.d);
    for (size_t i=0;i<count;++i) {
      std::copy_n(codes->data()+size_t(ids[i])*data.d,data.d,row.data());
      scores[i]=offset+inner_product(weights.data(),row.data(),data.d);
    }
  }};
}
// A separately trained flat TieredQ (same seed and sample) at bits*d code bits; candidate rows are gathered and
// scored with the codec's row scorer.
Rescorer tieredq_rescorer(const Data& data,uint32_t bits,size_t nlist) {
  auto codec=std::make_shared<dv::TieredQ>(train(data,bits*uint32_t(data.d),nlist));
  const size_t row_bytes=codec->bytes_per_vector();
  auto codes=std::make_shared<std::vector<uint8_t>>(data.n*row_bytes);
  codec->encode(data.base.data(),data.n,codes->data());
  return {"tieredq:"+std::to_string(bits),double(row_bytes),
          [codec,codes,row_bytes](const float* query,const uint32_t* ids,size_t count,float* scores) {
    thread_local std::vector<uint8_t> rows;
    rows.resize(count*row_bytes);
    for (size_t i=0;i<count;++i) std::copy_n(codes->data()+size_t(ids[i])*row_bytes,row_bytes,rows.data()+i*row_bytes);
    codec->score(rows.data(),count,codec->query(query),scores);
  }};
}
uint32_t prefix_bits() { return std::getenv("PREFIX_BITS") ? uint32_t(std::stoul(std::getenv("PREFIX_BITS"))) : 1416; }
// Residual refinement of a flat TieredQ prefix (PREFIX_BITS code bits, default 1416): a second TieredQ, one
// centroid, trained on x - decode(prefix) at bits*d minus the prefix's code bits. Scores add q.decode(prefix) and the
// residual estimate; bytes count the residual rows only (the prefix is the in-RAM first stage).
Rescorer tieredq_residual_rescorer(const Data& data,uint32_t bits,size_t nlist) {
  constexpr size_t decode_block=256;
  const auto prefix=train(data,prefix_bits(),nlist);
  const auto prefix_codes=prefix.encode(data.base.data(),data.n);
  auto decoded=std::make_shared<std::vector<float>>(data.n*data.d);
#pragma omp parallel for schedule(dynamic)
  for (size_t first=0;first<data.n;first+=decode_block)
    prefix.decode(prefix_codes.data()+first*prefix.bytes_per_vector(),std::min(decode_block,data.n-first),
                  decoded->data()+first*data.d);
  Data residual{std::vector<float>(data.n*data.d),{},{},data.d,data.n,0};
  for (size_t i=0;i<residual.base.size();++i) residual.base[i]=data.base[i]-(*decoded)[i];
  auto codec=std::make_shared<dv::TieredQ>(train(residual,bits*uint32_t(data.d)-prefix.config().code_bits,1));
  const size_t row_bytes=codec->bytes_per_vector();
  auto codes=std::make_shared<std::vector<uint8_t>>(data.n*row_bytes);
  codec->encode(residual.base.data(),data.n,codes->data());
  std::fprintf(stderr,"residual tier %u: prefix %u code bits, residual %u code bits\n",bits,prefix.config().code_bits,
               codec->config().code_bits);
  return {"tieredq-residual:"+std::to_string(bits),double(row_bytes),
          [&data,decoded,codec,codes,row_bytes](const float* query,const uint32_t* ids,size_t count,float* scores) {
    thread_local std::vector<uint8_t> rows;
    rows.resize(count*row_bytes);
    for (size_t i=0;i<count;++i) std::copy_n(codes->data()+size_t(ids[i])*row_bytes,row_bytes,rows.data()+i*row_bytes);
    codec->score(rows.data(),count,codec->query(query),scores);
    for (size_t i=0;i<count;++i) scores[i]+=inner_product(query,decoded->data()+size_t(ids[i])*data.d,data.d);
  }};
}
// TieredQRefinement of the flat PREFIX_BITS model to bits*d total code bits (same sample and seed); bytes are the
// suffix rows. Candidates' prefix and suffix rows are gathered and scored together.
Rescorer tieredq_refinement_rescorer(const Data& data,uint32_t bits,size_t nlist) {
  auto prefix=std::make_shared<dv::TieredQ>(train(data,prefix_bits(),nlist));
  const auto sample=training_sample(data);
  auto refinement=std::make_shared<dv::TieredQRefinement>(
      dv::TieredQRefinement::train(*prefix,bits*uint32_t(data.d),sample.data(),sample.size()/data.d,seed));
  const size_t prefix_bytes=prefix->bytes_per_vector(),suffix_bytes=refinement->bytes_per_vector();
  auto prefix_codes=std::make_shared<std::vector<uint8_t>>(prefix->encode(data.base.data(),data.n));
  auto suffix_codes=std::make_shared<std::vector<uint8_t>>(data.n*suffix_bytes);
  refinement->encode(data.base.data(),data.n,prefix_codes->data(),suffix_codes->data());
  std::fprintf(stderr,"refinement tier %u: prefix %u code bits, suffix %zu bits, %zu bytes\n",bits,prefix->config().code_bits,
               refinement->suffix_bits(),suffix_bytes);
  return {"tieredq-refinement:"+std::to_string(bits),double(suffix_bytes),
          [refinement,prefix_codes,suffix_codes,prefix_bytes,suffix_bytes](const float* query,const uint32_t* ids,size_t count,
                                                                            float* scores) {
    thread_local std::vector<uint8_t> prefix_rows,suffix_rows;
    prefix_rows.resize(count*prefix_bytes); suffix_rows.resize(count*suffix_bytes);
    for (size_t i=0;i<count;++i) {
      std::copy_n(prefix_codes->data()+size_t(ids[i])*prefix_bytes,prefix_bytes,prefix_rows.data()+i*prefix_bytes);
      std::copy_n(suffix_codes->data()+size_t(ids[i])*suffix_bytes,suffix_bytes,suffix_rows.data()+i*suffix_bytes);
    }
    refinement->score(prefix_rows.data(),suffix_rows.data(),count,refinement->query(query),scores);
  }};
}
// Mean row of every list: RaBitQ's centroids on a shared partition.
std::vector<float> list_means(const Data& data,const std::vector<uint32_t>& partition,size_t nlist) {
  std::vector<float> centroids(nlist*data.d,0.0f);
  std::vector<size_t> sizes(nlist,0);
  for (size_t i=0;i<data.n;++i) {
    ++sizes[partition[i]];
    for (size_t j=0;j<data.d;++j) centroids[partition[i]*data.d+j]+=data.base[i*data.d+j];
  }
  for (size_t list=0;list<nlist;++list) if (sizes[list])
    for (size_t j=0;j<data.d;++j) centroids[list*data.d+j]/=float(sizes[list]);
  return centroids;
}
// RaBitQ's IVF codes as one row: the BENCH_SEED rotation (as in its IVF), the residual to the row's list mean, the
// 1-bit code and the B-1 ex-bits, scored like its IVF's high-accuracy ex-bit stage. Bytes count the bin factors.
Rescorer rabitq_rescorer(const Data& data,size_t bits,const std::vector<uint32_t>& partition,size_t nlist) {
  struct Tier {
    std::unique_ptr<rabitqlib::Rotator<float>> rotator;
    size_t padded,ex_bits,bin_bytes,ex_bytes;
    std::vector<float> centroids;   // rotated
    std::vector<uint32_t> lists;
    std::vector<char> codes;        // bin then ex per row
    float (*ip)(const float*,const uint8_t*,size_t);
  };
  auto tier=std::make_shared<Tier>();
  tier->rotator.reset(rabitqlib::choose_rotator<float>(data.d,rabitqlib::RotatorType::FhtKacRotator,
                                                       rabitqlib::round_up_to_multiple(data.d,64)));
  tier->padded=tier->rotator->size(); tier->ex_bits=bits-1;
  tier->bin_bytes=rabitqlib::BinDataMap<float>::data_bytes(tier->padded);
  tier->ex_bytes=rabitqlib::ExDataMap<float>::data_bytes(tier->padded,tier->ex_bits);
  tier->ip=rabitqlib::select_excode_ipfunc(tier->ex_bits);
  tier->lists=partition;
  const auto means=list_means(data,partition,nlist);
  tier->centroids.resize(nlist*tier->padded);
  for (size_t list=0;list<nlist;++list) tier->rotator->rotate(means.data()+list*data.d,tier->centroids.data()+list*tier->padded);
  const size_t row_bytes=tier->bin_bytes+tier->ex_bytes;
  tier->codes.resize(data.n*row_bytes);
#pragma omp parallel for schedule(dynamic,256)
  for (size_t i=0;i<data.n;++i) {
    thread_local std::vector<float> rotated;
    rotated.resize(tier->padded);
    tier->rotator->rotate(data.base.data()+i*data.d,rotated.data());
    char* row=tier->codes.data()+i*row_bytes;
    rabitqlib::quant::quantize_split_single(rotated.data(),tier->centroids.data()+size_t(partition[i])*tier->padded,
                                            tier->padded,tier->ex_bits,row,row+tier->bin_bytes,rabitqlib::METRIC_IP);
  }
  return {"rabitq:"+std::to_string(bits),double(row_bytes),
          [tier,row_bytes](const float* query,const uint32_t* ids,size_t count,float* scores) {
    thread_local std::vector<float> rotated;
    rotated.resize(tier->padded);
    tier->rotator->rotate(query,rotated.data());
    const rabitqlib::SplitSingleQuery<float> q(rotated.data(),tier->padded,tier->ex_bits,rabitqlib::quant::RabitqConfig(),
                                               rabitqlib::METRIC_IP);
    for (size_t i=0;i<count;++i) {
      const char* row=tier->codes.data()+size_t(ids[i])*row_bytes;
      const float centroid_ip=rabitqlib::dot_product<float>(rotated.data(),
                                                            tier->centroids.data()+size_t(tier->lists[ids[i]])*tier->padded,tier->padded);
      float distance,low,ip_x0_qr;
      rabitqlib::split_single_fulldist(row,row+tier->bin_bytes,tier->ip,q,tier->padded,tier->ex_bits,distance,low,
                                       ip_x0_qr,-centroid_ip,0);
      scores[i]=-distance;
    }
  }};
}
// RESCORE tiers (default fp32) over the run's partition, which the RaBitQ tier uses for its centroids.
std::vector<Rescorer> rescorers(const Data& data,const std::vector<uint32_t>& partition,size_t lists,size_t codec_lists) {
  std::vector<Rescorer> tiers;
  const std::string spec=std::getenv("RESCORE") ? std::getenv("RESCORE") : "fp32";
  for (size_t at=0;at<spec.size();) {
    const size_t end=std::min(spec.find(',',at),spec.size());
    const std::string name=spec.substr(at,end-at),kind=name.substr(0,name.find(':'));
    const uint32_t bits=name.find(':')==std::string::npos ? 0 : uint32_t(std::stoul(name.substr(name.find(':')+1)));
    std::fprintf(stderr,"building rerank tier %s\n",name.c_str());
    if (kind=="fp32") tiers.push_back(fp32_rescorer(data));
    else if (kind=="fp16") tiers.push_back(fp16_rescorer(data));
    else if (kind=="int8") tiers.push_back(int8_rescorer(data));
    else if (kind=="tieredq") tiers.push_back(tieredq_rescorer(data,bits,codec_lists));
    else if (kind=="tieredq-residual") tiers.push_back(tieredq_residual_rescorer(data,bits,codec_lists));
    else if (kind=="tieredq-refinement") tiers.push_back(tieredq_refinement_rescorer(data,bits,codec_lists));
    else if (kind=="rabitq") tiers.push_back(rabitq_rescorer(data,bits,partition,lists));
    else { std::fprintf(stderr,"unknown rerank tier %s\n",name.c_str()); std::exit(2); }
    at=end+1;
  }
  return tiers;
}
// FNV-1a over list sizes: equal hashes mean both codecs scan the same partition.
uint64_t list_hash(const std::vector<size_t>& sizes) {
  uint64_t hash=14695981039346656037ULL;
  for (size_t size:sizes) { hash^=size; hash*=1099511628211ULL; }
  return hash;
}
struct Searcher {
  std::function<void(const float*,size_t,size_t,size_t,uint32_t*)> batch;  // queries, rows, k, nprobe, ids (8 threads)
  std::function<void(const float*,size_t,size_t,uint32_t*)> single;        // one query on the calling thread
};
void sweep_rerank(const char* codec,double budget,size_t nlist,const Data& data,const Searcher& search,double disk_bits,
                  double code_bits,double build_s,uint64_t lists,bool timed,size_t nprobe,size_t depth,
                  const std::vector<Rescorer>& tiers) {
  std::vector<uint32_t> candidates(data.nq*depth),ids(data.nq*top_k);
  omp_set_num_threads(threads);
  search.batch(data.queries.data(),data.nq,depth,nprobe,candidates.data());
  const double candidate_recall=recall(data,candidates,depth);
  for (const auto& tier:tiers) {
    auto rerank_rows=[&](size_t first,size_t rows) {
      size_t reads=0;
#pragma omp parallel for schedule(dynamic) reduction(+:reads)
      for (size_t q=first;q<first+rows;++q)
        reads+=rerank(tier,data.queries.data()+q*data.d,candidates.data()+q*depth,depth,ids.data()+q*top_k);
      return reads;
    };
    omp_set_num_threads(threads);
    const double reads=double(rerank_rows(0,data.nq))/double(data.nq);
    const double r=recall(data,ids);
    double qps1=0,qps8=0;
    if (timed) {
      qps8=double(data.nq)/best_seconds([&] {
        for (size_t q=0;q<data.nq;q+=batch) {
          const size_t rows=std::min(batch,data.nq-q);
          search.batch(data.queries.data()+q*data.d,rows,depth,nprobe,candidates.data()+q*depth);
          rerank_rows(q,rows);
        }
      });
      omp_set_num_threads(1);
      qps1=double(data.nq)/best_seconds([&] {
        for (size_t q=0;q<data.nq;++q) {
          search.single(data.queries.data()+q*data.d,depth,nprobe,candidates.data()+q*depth);
          rerank(tier,data.queries.data()+q*data.d,candidates.data()+q*depth,depth,ids.data()+q*top_k);
        }
      });
    }
    std::printf("%s\t%g\t%zu\t%zu\t%.1f\t%.0f\t%zu\t%.4f\t%.4f\t%.1f\t%zu\t%.0f\t%.0f\t%.2f\t%016llx\t%s\t%.0f\n",codec,
                budget,nlist,nprobe,disk_bits,code_bits,depth,candidate_recall,r,reads,32*data.d,qps1,qps8,build_s,
                static_cast<unsigned long long>(lists),tier.name.c_str(),tier.bytes);
    std::fflush(stdout);
  }
}
void sweep(const char* codec,double budget,size_t nlist,const Data& data,const Searcher& search,double disk_bits,
           double code_bits,double build_s,uint64_t lists,bool timed,const std::vector<size_t>& nprobes,
           const std::vector<Rescorer>& tiers) {
  const auto depths=rerank_depths();
  std::vector<uint32_t> ids(data.nq*top_k);
  for (size_t nprobe:nprobes) {
    if (!depths.empty()) {
      for (size_t depth:depths)
        sweep_rerank(codec,budget,nlist,data,search,disk_bits,code_bits,build_s,lists,timed,nprobe,depth,tiers);
      continue;
    }
    omp_set_num_threads(threads);
    search.batch(data.queries.data(),data.nq,top_k,nprobe,ids.data());
    const double r=recall(data,ids);
    double qps1=0,qps8=0;
    if (timed) {
      qps8=double(data.nq)/best_seconds([&] {
        for (size_t q=0;q<data.nq;q+=batch)
          search.batch(data.queries.data()+q*data.d,std::min(batch,data.nq-q),top_k,nprobe,ids.data()+q*top_k);
      });
      omp_set_num_threads(1);
      qps1=double(data.nq)/best_seconds([&] {
        for (size_t q=0;q<data.nq;++q) search.single(data.queries.data()+q*data.d,top_k,nprobe,ids.data()+q*top_k);
      });
    }
    std::printf("%s\t%g\t%zu\t%zu\t%.1f\t%.0f\t%.4f\t%.0f\t%.0f\t%.2f\t%016llx\n",codec,budget,nlist,nprobe,disk_bits,
                code_bits,r,qps1,qps8,build_s,static_cast<unsigned long long>(lists));
    std::fflush(stdout);
  }
}
double tieredq_disk_bits(const dv::TieredQIVFIndex& index) { return 8.0*double(index.serialize().size())/double(index.size()); }

size_t coarse_lists() { return std::getenv("LISTS") ? std::stoul(std::getenv("LISTS")) : 0; }
// PRUNE_Z / PRUNE_PREFIX: TieredQPruning confidence and prefix tiers (defaults: exact bound, 4 tiers).
dv::TieredQPruning pruning() {
  dv::TieredQPruning options;
  if (const char* z=std::getenv("PRUNE_Z")) options.confidence=std::stod(z);
  if (const char* prefix=std::getenv("PRUNE_PREFIX")) options.prefix_tiers=uint32_t(std::stoul(prefix));
  return options;
}

void run_tieredq(const Data& data,uint32_t code_bits,size_t nlist,bool timed,const std::vector<size_t>& nprobes,double label) {
  const auto start=std::chrono::steady_clock::now();
  omp_set_num_threads(threads);
  dv::TieredQIVFIndex index(train(data,code_bits,nlist),uint32_t(coarse_lists()));
  index.build(data.base.data(),data.n);
  const double build_s=seconds_since(start);
  std::vector<size_t> sizes(index.lists());
  for (size_t list=0;list<sizes.size();++list) sizes[list]=index.list_size(list);
  Searcher search{
    [&](const float* q,size_t rows,size_t k,size_t nprobe,uint32_t* ids) {
      std::vector<float> scores(rows*k);
      index.search_batch(q,rows,k,nprobe,scores.data(),ids,pruning());
    },
    [&](const float* q,size_t k,size_t nprobe,uint32_t* ids) {
      const auto hits=index.search(q,k,nprobe,pruning());
      for (size_t i=0;i<k;++i) ids[i]=i<hits.size() ? hits[i].id : UINT32_MAX;
    }};
  const auto tiers=rescorers(data,index.list_assignment(),index.lists(),nlist);
  sweep("tieredq",label,index.lists(),data,search,tieredq_disk_bits(index),index.codec().config().code_bits,build_s,
        list_hash(sizes),timed,nprobes,tiers);
}

// Repeated single-query loops on the calling thread, nothing else timed: a clean perf target.
void profile_tieredq(const Data& data,uint32_t code_bits,size_t nlist,const std::vector<size_t>& nprobes) {
  const size_t profile_rounds=std::getenv("PROFILE_ROUNDS") ? std::stoul(std::getenv("PROFILE_ROUNDS")) : 10;
  omp_set_num_threads(threads);
  dv::TieredQIVFIndex index(train(data,code_bits,nlist),uint32_t(coarse_lists()));
  index.build(data.base.data(),data.n);
  omp_set_num_threads(1);
  std::fprintf(stderr,"coded ranks %zu of %zu, tiers (bits x dims):",index.codec().coded_dimensions(),index.codec().model_dimensions());
  for (const auto& tier:index.codec().tiers()) std::fprintf(stderr," %ux%u",tier.bits,tier.dimensions);
  std::fprintf(stderr,"\nprofile ready\n");
  for (size_t nprobe:nprobes) {
    const auto start=std::chrono::steady_clock::now();
    size_t found=0;
    for (size_t round=0;round<profile_rounds;++round)
      for (size_t q=0;q<data.nq;++q) found+=index.search(data.queries.data()+q*data.d,top_k,nprobe,pruning()).size();
    std::printf("profile\t%u\t%zu\t%zu\t%.0f\t%zu\n",code_bits,nlist,nprobe,
                double(profile_rounds*data.nq)/seconds_since(start),found);
    std::fflush(stdout);
  }
}

// Largest code budget (to match_resolution) whose serialized IVF fits target bits/vector; storage only.
uint32_t match_code_bits(const Data& data,double target,size_t nlist) {
  auto fits=[&](uint32_t code_bits) {
    dv::TieredQIVFIndex index(train(data,code_bits,nlist),uint32_t(coarse_lists()));
    index.build(data.base.data(),data.n);
    const double bits=tieredq_disk_bits(index);
    std::fprintf(stderr,"match target %.1f: code_bits %u -> %.1f bits/vector\n",target,code_bits,bits);
    return bits<=target;
  };
  const double max_bits=std::getenv("MAX_BITS") ? std::stod(std::getenv("MAX_BITS")) : 8.0;
  // MATCH_SPAN narrows the search to [target - span, target] when the per-vector overhead is known roughly.
  const double span=std::getenv("MATCH_SPAN") ? std::stod(std::getenv("MATCH_SPAN")) : target;
  uint32_t lo=uint32_t(std::max(1.0,target-span)),hi=uint32_t(std::min<double>(target,max_bits*double(data.d)));
  if (fits(hi)) lo=hi;
  while (hi-lo>match_resolution) { const uint32_t mid=lo+(hi-lo)/2; (fits(mid) ? lo : hi)=mid; }
  std::fprintf(stderr,"matched code budget %u\n",lo);
  return lo;
}

// Bits per position of the unrotated basis (positions are PCA ranks grouped by tier, widest first).
std::vector<uint32_t> position_bits(const dv::TieredQ& codec) {
  std::vector<uint32_t> bits;
  for (const auto& tier:codec.tiers()) bits.insert(bits.end(),tier.dimensions,tier.bits);
  return bits;
}
// Whether each budget's model refines the previous one, as a nested code needs: equal centroids and basis
// positions (compared on rotate_tiers=false models, whose query weights are projections times sigma), bits that
// never drop per position, and tiers that line up, since each tier is rotated as one block.
void check_nesting(const Data& data,const std::vector<uint32_t>& budgets,size_t nlist) {
  omp_set_num_threads(threads);
  std::vector<dv::TieredQ> models,plain;
  for (uint32_t budget:budgets) {
    models.push_back(train(data,budget,nlist)); plain.push_back(train(data,budget,nlist,false));
    std::printf("nesting\tmodel\t%u\tcode_bits %u\tbytes %zu\ttiers",budget,models.back().config().code_bits,
                models.back().bytes_per_vector());
    for (const auto& tier:models.back().tiers()) std::printf(" %ux%u",tier.bits,tier.dimensions);
    std::printf("\n");
  }
  auto max_difference=[](const std::vector<float>& a,const std::vector<float>& b,size_t count) {
    float diff=0,scale=0;
    for (size_t i=0;i<count;++i) { diff=std::max(diff,std::abs(a[i]-b[i])); scale=std::max(scale,std::abs(a[i])); }
    return diff/std::max(scale,1e-30f);
  };
  for (size_t m=1;m<models.size();++m) {
    const auto coarse=plain[m-1].query_weights(data.queries.data()),fine=plain[m].query_weights(data.queries.data());
    const auto bits_coarse=position_bits(models[m-1]),bits_fine=position_bits(models[m]);
    size_t lower=0,equal_tiers=0,inside=0;
    for (size_t i=0;i<bits_coarse.size();++i) lower+=i>=bits_fine.size() || bits_fine[i]<bits_coarse[i];
    for (const auto& tier:models[m-1].tiers()) for (const auto& other:models[m].tiers()) {
      equal_tiers+=tier.start==other.start && tier.dimensions==other.dimensions;
      inside+=tier.start>=other.start && tier.start+tier.dimensions<=other.start+other.dimensions;
    }
    std::printf("nesting\t%u->%u\tcentroid_diff %.2e\tbasis_diff %.2e (positions %zu)\tpositions_losing_bits %zu of %zu"
                "\ttiers_equal %zu inside_one %zu of %zu\n",budgets[m-1],budgets[m],
                max_difference(coarse.centroid_scores,fine.centroid_scores,coarse.centroid_scores.size()),
                max_difference(coarse.weights,fine.weights,std::min(coarse.weights.size(),fine.weights.size())),
                std::min(coarse.weights.size(),fine.weights.size()),lower,bits_coarse.size(),equal_tiers,inside,
                models[m-1].tiers().size());
  }
}

// TieredQ's partition: the codec's k-means lists, or with LISTS its coarse quantizer's; both are independent of the
// code budget at a fixed seed (the coarse quantizer trains in model space, which for inner product is the input).
std::vector<uint32_t> tieredq_partition(const Data& data,size_t nlist) {
  if (!coarse_lists()) {
    const auto codec=train(data,uint32_t(data.d),nlist);
    const auto encoded=codec.encode(data.base.data(),data.n);
    std::vector<uint32_t> lists(data.n);
    for (size_t i=0;i<data.n;++i) lists[i]=encoded[i*codec.bytes_per_vector()+codec.code_bytes()];
    return lists;
  }
  dv::TieredQIVFIndex index(train(data,uint32_t(data.d),nlist),uint32_t(coarse_lists()));
  index.build(data.base.data(),data.n);
  return index.list_assignment();
}

void run_rabitq(const Data& data,size_t bits,size_t codec_lists,bool timed,const std::vector<size_t>& nprobes) {
  omp_set_num_threads(threads);
  const size_t nlist=coarse_lists() ? coarse_lists() : codec_lists;
  const auto partition=tieredq_partition(data,codec_lists);
  const std::vector<rabitqlib::PID> assignment(partition.begin(),partition.end());
  const auto centroids=list_means(data,partition,nlist);
  std::vector<size_t> sizes(nlist,0);
  for (uint32_t list:partition) ++sizes[list];
  const auto start=std::chrono::steady_clock::now();
  rabitqlib::ivf::IVF index(data.n,data.d,nlist,bits,rabitqlib::METRIC_IP);
  index.construct(data.base.data(),centroids.data(),assignment.data(),false);
  const double build_s=seconds_since(start);
  const auto path=std::filesystem::temp_directory_path()/("issue6-rabitq-"+std::to_string(::getpid())+".bin");
  index.save(path.c_str());
  const double disk_bits=8.0*double(std::filesystem::file_size(path))/double(data.n);
  std::filesystem::remove(path);
  Searcher search{
    // RaBitQ writes only the hits it found; empty slots stay UINT32_MAX.
    [&](const float* q,size_t rows,size_t k,size_t nprobe,uint32_t* ids) {
      std::fill_n(ids,rows*k,UINT32_MAX);
#pragma omp parallel for schedule(dynamic)
      for (size_t row=0;row<rows;++row) index.search(q+row*data.d,k,nprobe,ids+row*k,true);
    },
    [&](const float* q,size_t k,size_t nprobe,uint32_t* ids) { std::fill_n(ids,k,UINT32_MAX); index.search(q,k,nprobe,ids,true); }};
  sweep("rabitq",double(bits),nlist,data,search,disk_bits,double(bits*data.d),build_s,list_hash(sizes),timed,nprobes,
        rescorers(data,partition,nlist,codec_lists));
}
} // namespace

int main(int argc,char** argv) {
  if (argc<8) { std::fprintf(stderr,"usage: %s MODE BUDGET DATA_DIR D NLIST TIMED NPROBE...\n",argv[0]); return 2; }
  const std::string mode=argv[1],dir=argv[3];
  const double budget=std::stod(argv[2]);
  Data data{load<float>(dir+"/base.f32"),load<float>(dir+"/queries.f32"),load<uint32_t>(dir+"/truth.u32"),std::stoul(argv[4]),0,0};
  data.n=data.base.size()/data.d; data.nq=data.queries.size()/data.d;
  const size_t nlist=std::stoul(argv[5]);
  const bool timed=std::stoi(argv[6]);
  std::vector<size_t> nprobes;
  for (int a=7;a<argc;++a) nprobes.push_back(std::stoul(argv[a]));
  if (mode=="tieredq") run_tieredq(data,uint32_t(budget),nlist,timed,nprobes,budget);
  else if (mode=="tieredq-match") run_tieredq(data,match_code_bits(data,budget,nlist),nlist,timed,nprobes,budget);
  else if (mode=="tieredq-profile") profile_tieredq(data,uint32_t(budget),nlist,nprobes);
  else if (mode=="rabitq") run_rabitq(data,size_t(budget),nlist,timed,nprobes);
  else if (mode=="tieredq-nesting") {
    std::vector<uint32_t> budgets;
    for (const char* at=argv[2];*at;) { char* end; budgets.push_back(uint32_t(std::strtoul(at,&end,10))); at=*end ? end+1 : end; }
    check_nesting(data,budgets,nlist);
  }
  else { std::fprintf(stderr,"unknown mode %s\n",mode.c_str()); return 2; }
}
