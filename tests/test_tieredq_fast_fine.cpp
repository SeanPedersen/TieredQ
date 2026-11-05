// Fast mode with fine tiers (#26): refined prefix books, the automatic policy, model round trips, exact rescoring of
// the scanned prefix candidates at every prefix width, batch/single agreement, rejected targets and indexes, the
// flat and IVF indexes' agreement, and persistence.
#include "tieredq/tieredq.h"
#include "tieredq/tieredq_blocks.h"
#include "tieredq/tieredq_refinement.h"
#include "tieredq_train_codebook.h"
#include "tieredq_lookup_layout.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <unistd.h>
#include <vector>
namespace {
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }

constexpr uint32_t widest_fast_bits=12,dimensions=64,centroids=4;
constexpr size_t rows=2048,queries=16,k=10;
constexpr double score_tolerance=1e-4;

// Every refined level lies inside its prefix level's nearest-level cell, sorted and cell major.
void refined_book_test() {
  for (unsigned width=dv::tieredq_detail::additive_wide_max_bits;width<=dv::tieredq_detail::widest_code_bits;++width) {
    const auto& prefix=dv::tieredq_detail::additive_wide_codebook(width);
    for (unsigned fine=1;width+fine<=widest_fast_bits;++fine) {
      const auto book=dv::tieredq_detail::refined_prefix_codebook(prefix,fine);
      require(book.size()==prefix.size()<<fine,"refined book size");
      require(std::is_sorted(book.begin(),book.end()),"refined book is not sorted");
      for (size_t code=0;code<book.size();++code) {
        const size_t cell=code>>fine;
        const double lo=cell ? (prefix[cell-1]+double(prefix[cell]))/2 : -INFINITY;
        const double hi=cell+1<prefix.size() ? (prefix[cell]+double(prefix[cell+1]))/2 : INFINITY;
        require(book[code]>=lo && book[code]<=hi,"refined level leaves its prefix cell");
      }
    }
  }
}

// Rows whose spectrum decays by `decay` per dimension; steep decays allocate codes wider than the prefix.
constexpr double steep_decay=0.9;
std::vector<float> decaying_rows(size_t count,std::mt19937& random,double decay=steep_decay) {
  std::normal_distribution<float> normal;
  std::vector<float> data(count*dimensions);
  for (size_t i=0;i<count;++i) for (uint32_t j=0;j<dimensions;++j) data[i*dimensions+j]=float(normal(random)*std::pow(decay,j));
  return data;
}
dv::TieredQ train(const std::vector<float>& data,uint32_t bits_per_dim,bool fine,
                  dv::TieredQIndexMode target=dv::TieredQIndexMode::fast,dv::TieredQMetric metric=dv::TieredQMetric::inner_product) {
  dv::TieredQConfig config{dimensions,bits_per_dim*dimensions,centroids};
  config.fine_tiers=fine; config.target_mode=target; config.metric=metric;
  return dv::TieredQ::train(config,data.data(),rows);
}
uint32_t widest(const dv::TieredQ& codec) {
  uint32_t out=0;
  for (const auto& t:codec.tiers()) out=std::max(out,t.bits);
  return out;
}

// Training adds fine tiers to steep spectra at the automatic prefix width unless switched off: fast targets over the
// additive prefix, ip full targets (exact fast scores) over the 8-bit Lloyd-Max prefix. L2 ignores the flag.
void policy_test() {
  std::mt19937 random(26);
  const auto steep=decaying_rows(rows,random);
  for (uint32_t bits_per_dim=dv::tieredq_detail::additive_wide_max_bits;bits_per_dim<=dv::tieredq_detail::widest_code_bits;++bits_per_dim) {
    const auto codec=train(steep,bits_per_dim,true);
    require(codec.fast_prefix_bits()==bits_per_dim && widest(codec)>bits_per_dim,"training skipped fine tiers");
    const auto off=train(steep,bits_per_dim,false);
    require(!off.fast_prefix_bits() && widest(off)<=bits_per_dim,"fine tiers switched off were added");
    const auto restored=dv::TieredQ::deserialize(codec.serialize());
    require(restored.fast_prefix_bits()==codec.fast_prefix_bits() && restored.serialize()==codec.serialize(),
            "fine-tier model round trip");
  }
  const auto exact=train(steep,dv::tieredq_detail::widest_code_bits,true,dv::TieredQIndexMode::full);
  require(exact.fast_prefix_bits()==dv::tieredq_detail::widest_code_bits && widest(exact)>dv::tieredq_detail::widest_code_bits,
          "full-target training skipped fine tiers");
  require(!train(steep,dv::tieredq_detail::widest_code_bits,false,dv::TieredQIndexMode::full).fast_prefix_bits(),
          "full-target fine tiers switched off were added");
  for (auto target:{dv::TieredQIndexMode::fast,dv::TieredQIndexMode::full})
    require(!train(steep,dv::tieredq_detail::widest_code_bits,true,target,dv::TieredQMetric::l2).fast_prefix_bits(),
            "fine tiers reached an L2 target");
}

// IVF over a fine-tier model: probing every list scans and rescores like the flat index, so its hits equal the flat
// index's; single, batched and prepared-weight searches agree, and the index round-trips through bytes and files.
void ivf_test(const dv::TieredQ& codec,const std::vector<float>& data,const std::vector<float>& probe,
              const std::vector<float>& flat_scores,const std::vector<uint32_t>& flat_ids) {
  for (const uint32_t lists:{0u,8u}) {
    dv::TieredQIVFIndex ivf(codec,lists); ivf.build(data.data(),rows);
    std::vector<float> scores(queries*k); std::vector<uint32_t> ids(queries*k);
    ivf.search_batch(probe.data(),queries,k,ivf.lists(),scores.data(),ids.data());
    require(ids==flat_ids && scores==flat_scores,"fine-tier IVF full probe differs from the flat index");
    const auto weights=codec.query_weights(probe.data(),queries);
    std::vector<float> prepared_scores(queries*k); std::vector<uint32_t> prepared_ids(queries*k);
    ivf.search_batch(probe.data(),weights.data(),queries,k,ivf.lists(),prepared_scores.data(),prepared_ids.data());
    for (size_t q=0;q<queries;++q) {
      const auto hits=ivf.search(probe.data()+q*dimensions,k,ivf.lists());
      require(hits.size()==k,"fine-tier IVF single search size");
      for (size_t i=0;i<k;++i) {
        require(hits[i].id==ids[q*k+i] && hits[i].score==scores[q*k+i],"fine-tier IVF single and batch differ");
        require(std::fabs(hits[i].score-prepared_scores[q*k+i])<=score_tolerance*(1+std::fabs(hits[i].score)),
                "fine-tier IVF prepared-weight scores differ");
      }
    }
    std::vector<float> partial_scores(queries*k); std::vector<uint32_t> partial_ids(queries*k);
    ivf.search_batch(probe.data(),queries,k,1,partial_scores.data(),partial_ids.data());
    for (size_t q=0;q<queries;++q) for (size_t i=1;i<k;++i)
      require(partial_scores[q*k+i]<=partial_scores[q*k+i-1],"fine-tier IVF partial probe is not sorted");
    const auto round_trip=dv::TieredQIVFIndex::deserialize(ivf.serialize());
    const auto path=std::filesystem::temp_directory_path()/("tieredq-ivf-fine-"+std::to_string(getpid()));
    ivf.save(path.string());
    const auto loaded=dv::TieredQIVFIndex::load(path.string());
    std::filesystem::remove(path);
    for (const auto* reloaded:{&round_trip,&loaded}) {
      std::vector<float> again_scores(queries*k); std::vector<uint32_t> again_ids(queries*k);
      reloaded->search_batch(probe.data(),queries,k,ivf.lists(),again_scores.data(),again_ids.data());
      require(again_ids==ids && again_scores==scores,"fine-tier IVF changed hits on reload");
    }
  }
}

// Fine-tier fast search of a fast target (nibble prefix scan) or a full target (exact prefix scan): batch equals single
// search, rescored scores equal the full-width codes' exact scores, and the index round-trips.
void rescoring_test(uint32_t bits_per_dim,dv::TieredQIndexMode target) {
  std::mt19937 random(26);
  const auto data=decaying_rows(rows,random),probe=decaying_rows(queries,random);
  const auto codec=train(data,bits_per_dim,true,target);
  const uint32_t prefix=target==dv::TieredQIndexMode::fast ? bits_per_dim : dv::tieredq_detail::widest_code_bits;
  require(codec.fast_prefix_bits()==prefix && widest(codec)>prefix+2,"fixture allocates no tier 3 bits past its prefix");
  require(codec.config().code_bits<=bits_per_dim*dimensions,"fine-tier model exceeds its code budget");

  dv::TieredQIndex fast(codec),full(codec,dv::TieredQIndexMode::full);
  fast.add(data.data(),rows); full.add(data.data(),rows);
  std::vector<float> batch_scores(queries*k); std::vector<uint32_t> batch_ids(queries*k);
  fast.search_batch(probe.data(),queries,k,batch_scores.data(),batch_ids.data());
  size_t agree=0;
  for (size_t q=0;q<queries;++q) {
    const auto hits=fast.search(probe.data()+q*dimensions,k);
    const auto exact=full.search(probe.data()+q*dimensions,full.size());
    require(hits.size()==k,"fine-tier fast search returned too few hits");
    for (size_t i=0;i<k;++i) {
      require(hits[i].id==batch_ids[q*k+i] && hits[i].score==batch_scores[q*k+i],"fine-tier batch and single search differ");
      if (i) require(hits[i-1].score>=hits[i].score,"fine-tier hits are not sorted");
      // Rescored scores are the full-width codes' exact scores, as full mode computes them.
      const auto match=std::find_if(exact.begin(),exact.end(),[&](const auto& hit){ return hit.id==hits[i].id; });
      require(match!=exact.end() && std::fabs(match->score-hits[i].score)<=score_tolerance*(1+std::fabs(match->score)),
              "fine-tier rescored score differs from the exact score");
      agree+=std::any_of(exact.begin(),exact.begin()+k,[&](const auto& hit){ return hit.id==hits[i].id; });
    }
  }
  constexpr double min_agreement=0.9;
  require(double(agree)/double(queries*k)>=min_agreement,"fine-tier fast top-k strays from the exact top-k");
  require(fast.bytes_per_vector()<=codec.code_bytes()+codec.metadata_bytes_per_vector()+dimensions/2,
          "fine-tier fast storage exceeds its code budget");

  // Only the flat and IVF indexes serve fine-tier models.
  const auto rejects=[](auto&& build) { try { build(); } catch (const std::runtime_error&) { return true; } return false; };
  require(rejects([&] { dv::TieredQFastBlocks blocks(codec); }),"fast blocks accepted a fine-tier model");
  require(rejects([&] { dv::TieredQRefinement::train(codec,codec.config().code_bits+dimensions,data.data(),rows); }),
          "refinement accepted a fine-tier prefix");

  const auto path=std::filesystem::temp_directory_path()/("tieredq-fast-fine-"+std::to_string(getpid()));
  fast.save(path.string());
  const auto loaded=dv::TieredQIndex::load(path.string());
  std::filesystem::remove(path);
  require(loaded.size()==fast.size() && loaded.bytes_per_vector()==fast.bytes_per_vector() &&
          loaded.codec().fast_prefix_bits()==codec.fast_prefix_bits(),"fine-tier index changed shape on reload");
  std::vector<float> loaded_scores(queries*k); std::vector<uint32_t> loaded_ids(queries*k);
  loaded.search_batch(probe.data(),queries,k,loaded_scores.data(),loaded_ids.data());
  require(loaded_ids==batch_ids && loaded_scores==batch_scores,"fine-tier index changed hits on reload");
  // Reloaded fine rows are allocated once, without growth slack, and the scan's prefix view of the model shares its
  // basis and centroids, so neither index holds a second copy of them.
  require(loaded.storage_bytes()<=fast.storage_bytes(),"reloaded fine-tier index allocates more than the built one");
  // An empty index stores no rows, so its storage is the view's own state alone.
  const size_t basis_bytes=size_t(codec.coded_dimensions())*dimensions*sizeof(uint16_t);
  require(dv::TieredQIndex(codec).storage_bytes()<basis_bytes,"fine-tier prefix view copies the model");
  if (target==dv::TieredQIndexMode::fast) ivf_test(codec,data,probe,batch_scores,batch_ids);
}
} // namespace

void tieredq_fast_fine_tests() {
  refined_book_test();
  policy_test();
  for (uint32_t bits_per_dim=dv::tieredq_detail::additive_wide_max_bits;bits_per_dim<=dv::tieredq_detail::widest_code_bits;++bits_per_dim)
    rescoring_test(bits_per_dim,dv::TieredQIndexMode::fast);
  rescoring_test(dv::tieredq_detail::widest_code_bits,dv::TieredQIndexMode::full);
}
