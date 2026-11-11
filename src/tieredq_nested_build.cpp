// Nested IVF builds (#52): trains the prefix codec, the refinement and the lists from one seeded sample and builds the
// lists through a row loader, so a base held as an SQ8 store (sq8_row_loader) needs no float copy; write_tier()
// streams the tier rows into a row file the same way.
#include "tieredq/tieredq_nested.h"
#include "tieredq_io.h"
#include "training_rng.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace dv {
using tieredq_detail::bad;
using tieredq_detail::product;
namespace {
// Codec and refinement training rows; recall saturates at 8-16K training rows on 1M-row sets (#19.2).
constexpr size_t training_sample_rows=32768;
constexpr uint64_t training_sample_salt=0xbf58476d1ce4e5b9ULL;
constexpr uint32_t maximum_centroids=256;
constexpr uint32_t default_prefix_bits_per_dimension=1,default_total_bits_per_dimension=8;
// Rows decoded per loader call while writing a tier (bounds the float buffer to a few hundred MB at d = 1024).
constexpr size_t tier_write_chunk_rows=65536;
// SQ8 rows decoded per worker block.
constexpr size_t sq8_parallel_rows=1024;

// The training rows: all of them up to the cap, else a seeded sample, in id order.
std::vector<float> training_rows(const TieredQIVFIndex::RowLoader& rows,size_t count,size_t width,uint64_t seed) {
  std::vector<uint32_t> ids(std::min(count,training_sample_rows));
  if (count>ids.size()) {
    tieredq_detail::TrainingRng rng(seed^training_sample_salt);  // independent of the codec's own seeded k-means
    const auto picked=rng.choice(count,ids.size());
    std::copy(picked.begin(),picked.end(),ids.begin());
    std::sort(ids.begin(),ids.end());
  } else std::iota(ids.begin(),ids.end(),0u);
  std::vector<float> out(product(ids.size(),width));
  rows(ids.data(),ids.size(),out.data());
  return out;
}

TieredQConfig prefix_config(const TieredQNestedConfig& config,uint32_t prefix_bits,size_t sampled) {
  auto prefix=TieredQConfig::target_code_bits(config.dimensions,prefix_bits);
  prefix.metric=config.metric;
  prefix.fine_tiers=false;  // the IVF prefix stays independently scannable
  prefix.num_centroids=std::min(maximum_centroids,uint32_t(sampled/2));
  prefix.max_scan_bits=config.max_scan_bits;
  prefix.max_scan_bits=TieredQIVFIndex::automatic_scan_bits(prefix);
  prefix.validate();
  return prefix;
}

// The lists' training and router as the config asks; unset settings keep the IVF's automatic ones.
void configure_lists(TieredQIVFIndex& index,const TieredQNestedConfig& config) {
  if (config.coarse_training) index.set_coarse_training(*config.coarse_training);
  const bool automatic=index.lists()>=TieredQRouterConfig::automatic_lists;
  if (config.router.value_or(automatic)) index.build_router(config.router_config);
  else index.remove_router();
}
} // namespace

TieredQIVFIndex::RowLoader sq8_row_loader(const int8_t* codes,size_t rows,std::vector<float> scales,
                                          std::vector<float> offsets) {
  if (!codes && rows) bad("null SQ8 codes");
  if (scales.empty() || offsets.size()!=scales.size()) bad("SQ8 needs one scale and one offset per dimension");
  for (size_t j=0;j<scales.size();++j)
    if (!std::isfinite(scales[j]) || !std::isfinite(offsets[j])) bad("SQ8 scales and offsets must be finite");
  return [codes,rows,scales=std::move(scales),offsets=std::move(offsets)](const uint32_t* ids,size_t count,float* out) {
    const size_t d=scales.size();
    for (size_t i=0;i<count;++i) if (ids[i]>=rows) bad("SQ8 row id out of range");
#pragma omp parallel for schedule(static) if(count>=sq8_parallel_rows)
    for (size_t i=0;i<count;++i) {
      const int8_t* code=codes+size_t(ids[i])*d;
      float* row=out+i*d;
      for (size_t j=0;j<d;++j) row[j]=float(code[j])*scales[j]+offsets[j];
    }
  };
}

TieredQNestedIVFIndex TieredQNestedIVFIndex::build(const TieredQNestedConfig& config,
                                                   const TieredQIVFIndex::RowLoader& rows,size_t count,
                                                   const float* sample,size_t sample_rows) {
  const size_t d=config.dimensions;
  if (!d) bad("nested build needs the dimensions");
  if (sample && sample_rows<2) bad("training needs at least 2 rows");
  if (!sample && count<2) bad("training needs at least 2 rows");
  const TieredQIVFIndex::RowLoader sample_loader=[sample,d](const uint32_t* ids,size_t n,float* out) {
    for (size_t i=0;i<n;++i) std::copy_n(sample+size_t(ids[i])*d,d,out+i*d);
  };
  const auto training=sample ? training_rows(sample_loader,sample_rows,d,config.seed) : training_rows(rows,count,d,config.seed);
  const size_t sampled=training.size()/d;
  const uint32_t prefix_bits=config.prefix_bits ? config.prefix_bits : default_prefix_bits_per_dimension*uint32_t(d);
  const uint32_t total_bits=config.total_bits ? config.total_bits : default_total_bits_per_dimension*uint32_t(d);
  auto codec=TieredQ::train(prefix_config(config,prefix_bits,sampled),training.data(),sampled,config.seed);
  auto refinement=TieredQRefinement::train(codec,total_bits,training.data(),sampled,config.seed,config.fine_tiers,
                                           config.max_tier_bits);
  const uint32_t lists=config.lists==TieredQNestedConfig::automatic_lists ? TieredQIVFIndex::automatic_lists(count) : config.lists;
  TieredQIVFIndex index(std::move(codec),lists,config.seed);
  configure_lists(index,config);
  if (sample) index.train_lists(sample,sample_rows);
  index.build(rows,count);
  return TieredQNestedIVFIndex(std::move(index),std::move(refinement),config.tier_rows);
}

void TieredQNestedIVFIndex::write_tier(const std::string& path,const TieredQIVFIndex::RowLoader& rows) const {
  if (!rows) bad("null row loader");
  const size_t d=index_.codec().config().dimensions,stride=tier_bytes_per_vector();
  const size_t chunk=std::min(size(),tier_write_chunk_rows);
  TieredQRowFileWriter writer(path,stride);
  std::vector<uint32_t> ids(chunk);
  std::vector<float> decoded(product(chunk,d));
  std::vector<uint8_t> encoded(product(chunk,stride));
  for (size_t first=0;first<size();first+=chunk) {
    const size_t n=std::min(chunk,size()-first);
    std::iota(ids.begin(),ids.begin()+n,uint32_t(first));
    rows(ids.data(),n,decoded.data());
    encode_tier(decoded.data(),first,n,encoded.data());
    writer.append(encoded.data(),n);
  }
  writer.finish();
}
} // namespace dv
