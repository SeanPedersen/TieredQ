// Two-level compressed coarse router (#47): build, probe order and persistence. The member index is an ordinary L2
// TieredQIVFIndex over the routed centroids (k-means supers as its coarse lists), so routing reuses the IVF scan
// kernels and the router persists as that index's own bytes. The exact head re-rank packs the head's binary16 rows
// into one small panel set for the coarse GEMM kernel, so its scores equal the exact ranking's bit for bit.
#include "tieredq_ivf_router.h"
#include "tieredq_encode_float.h"
#include <algorithm>
#include <cmath>

namespace dv {
using tieredq_detail::bad;

void TieredQRouterConfig::validate() const {
  if (!lists_per_super) bad("router lists_per_super must be positive");
  if (!(bits_per_dimension>0 && bits_per_dimension<=16)) bad("router bits_per_dimension must be in (0, 16]");
  if (!(expansion>0) || !std::isfinite(expansion)) bad("router expansion must be positive");
  if (!min_super_probes) bad("router min_super_probes must be positive");
}

namespace tieredq_detail {
namespace {
// The router's scan cap: the 4-bit nibble kernels serve single queries about 20x faster than L2's default cap (#44).
constexpr uint32_t router_scan_bits=4;
// Member centroids per codec centroid when the codec is trained on few routed lists (training needs 2 rows each).
constexpr size_t members_per_codec_centroid=4;
constexpr uint32_t max_codec_centroids=256;

std::vector<uint32_t> routed_lists(const std::vector<uint32_t>& list_rows) {
  std::vector<uint32_t> lists;
  for (uint32_t list=0;list<list_rows.size();++list) if (list_rows[list]) lists.push_back(list);
  return lists;
}

TieredQ train_member_codec(const float* rows,size_t count,size_t dimensions,const TieredQRouterConfig& config) {
  const auto bits=uint32_t(std::max(1.0,std::round(config.bits_per_dimension*double(dimensions))));
  auto codec_config=TieredQConfig::target_code_bits(uint32_t(dimensions),bits);
  codec_config.metric=TieredQMetric::l2;
  codec_config.max_scan_bits=std::max(router_scan_bits,uint32_t(std::ceil(config.bits_per_dimension)));
  codec_config.fine_tiers=false;
  codec_config.num_centroids=uint32_t(std::clamp<size_t>(count/members_per_codec_centroid,1,max_codec_centroids));
  return TieredQ::train(codec_config,rows,count,config.seed);
}
} // namespace

CoarseRouter::CoarseRouter(const CoarseQuantizer& coarse,TieredQRouterConfig config,TieredQIVFIndex members,
                           std::vector<uint32_t> lists)
    : config_(config),members_(std::move(members)),lists_(std::move(lists)),halves_(product(lists_.size(),coarse.dimensions)),
      dimensions_(coarse.dimensions) {
  for (size_t member=0;member<lists_.size();++member) coarse.half_row(lists_[member],halves_.data()+member*dimensions_);
}

CoarseRouter CoarseRouter::build(const CoarseQuantizer& coarse,const std::vector<uint32_t>& list_rows,
                                 const TieredQRouterConfig& config) {
  config.validate();
  auto lists=routed_lists(list_rows);
  const size_t count=lists.size(),d=coarse.dimensions;
  if (count<2) bad("the router needs at least two nonempty lists");
  // Members are the rounded centroids the exact ranking scores, so both rank the same vectors.
  std::vector<float> rows(product(count,d));
  std::vector<uint16_t> half(d);
  for (size_t member=0;member<count;++member) {
    coarse.half_row(lists[member],half.data());
    const float scale=coarse.scales[lists[member]];
    for (size_t k=0;k<d;++k) rows[member*d+k]=half_decode(half[k])*scale;
  }
  const auto supers=uint32_t(std::max<size_t>(1,count/config.lists_per_super));
  TieredQIVFIndex members(train_member_codec(rows.data(),count,d,config),supers,config.seed);
  members.build(rows.data(),count);
  return CoarseRouter(coarse,config,std::move(members),std::move(lists));
}

size_t CoarseRouter::automatic_super_probes(size_t nprobe) const {
  const size_t supers=members_.lists();
  const double lists_per_probe=double(lists_.size())/double(supers);
  const auto wanted=size_t(std::ceil(double(nprobe)*config_.expansion/lists_per_probe));
  return std::clamp<size_t>(std::max<size_t>(wanted,config_.min_super_probes),1,supers);
}

std::vector<uint32_t> CoarseRouter::probe_order(const CoarseQuantizer& coarse,const float* query,size_t nprobe,
                                                uint32_t super_probes) const {
  const size_t routed=lists_.size(),wanted=std::min(nprobe,routed),head_target=std::min<size_t>(config_.rerank,routed);
  const size_t probes=super_probes ? std::min<size_t>(super_probes,members_.lists()) : automatic_super_probes(wanted);
  const auto hits=members_.search(query,std::max(wanted,head_target),probes,{},{TieredQRouting::Mode::exact});
  const size_t head=std::min(head_target,hits.size());
  std::vector<std::pair<float,uint32_t>> ranked(head);  // (closeness, list)
  if (head) {
    HalfPanels panels; panels.depth=dimensions_; panels.columns=head;
    panels.data=pack_panel_values<uint16_t>(dimensions_,head,[&](size_t k,size_t c) {
      return halves_[size_t(hits[c].id)*dimensions_+k];
    });
    std::vector<float> dots(panels.padded_columns());
    half_matrix(query,1,dimensions_,panels,dots.data(),dots.size());
    for (size_t c=0;c<head;++c) {
      const uint32_t list=lists_[hits[c].id];
      ranked[c]={dots[c]*coarse.scales[list]-coarse.half_norms[list],list};
    }
    std::sort(ranked.begin(),ranked.end(),[](const auto& a,const auto& b) {
      return a.first>b.first || (a.first==b.first && a.second<b.second);
    });
  }
  std::vector<uint32_t> order;
  order.reserve(hits.size());
  for (const auto& [closeness,list]:ranked) order.push_back(list);
  for (size_t hit=head;hit<hits.size();++hit) order.push_back(lists_[hits[hit].id]);
  order.resize(std::min(order.size(),wanted));
  return order;
}

size_t CoarseRouter::allocated_bytes() const {
  return sizeof(*this)+members_.storage_bytes()+members_.codec().memory_bytes()+lists_.capacity()*sizeof(uint32_t)+
      halves_.capacity()*sizeof(uint16_t);
}

void CoarseRouter::write_config(Writer& writer,const TieredQRouterConfig& config) {
  writer.u32(config.lists_per_super);
  writer.u64(std::bit_cast<uint64_t>(config.bits_per_dimension));
  writer.u32(config.rerank);
  writer.u64(std::bit_cast<uint64_t>(config.expansion));
  writer.u32(config.min_super_probes);
  writer.u64(config.seed);
}

// The routed lists are the index's nonempty lists, so only the member index is stored.
std::vector<uint8_t> CoarseRouter::serialize() const { return members_.serialize(); }

CoarseRouter CoarseRouter::deserialize(const CoarseQuantizer& coarse,const std::vector<uint32_t>& list_rows,
                                       const TieredQRouterConfig& config,const std::vector<uint8_t>& bytes) {
  auto lists=routed_lists(list_rows);
  auto members=TieredQIVFIndex::deserialize(bytes);
  if (members.size()!=lists.size() || members.codec().config().dimensions!=coarse.dimensions ||
      members.codec().config().metric!=TieredQMetric::l2 || !members.lists())
    bad("invalid IVF router");
  return CoarseRouter(coarse,config,std::move(members),std::move(lists));
}
} // namespace tieredq_detail
} // namespace dv
