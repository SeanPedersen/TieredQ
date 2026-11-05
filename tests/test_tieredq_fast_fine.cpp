// Fast mode with fine tiers (#26): refined prefix books, the automatic policy, model round trips, exact rescoring of
// the scanned prefix candidates at every prefix width, batch/single agreement, rejected targets and indexes, and
// persistence.
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
constexpr double steep_decay=0.9,flat_decay=1.0;
std::vector<float> decaying_rows(size_t count,std::mt19937& random,double decay=steep_decay) {
  std::normal_distribution<float> normal;
  std::vector<float> data(count*dimensions);
  for (size_t i=0;i<count;++i) for (uint32_t j=0;j<dimensions;++j) data[i*dimensions+j]=float(normal(random)*std::pow(decay,j));
  return data;
}
dv::TieredQ train(const std::vector<float>& data,uint32_t bits_per_dim,dv::TieredQFineTiers fine,
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

// Automatic training adds fine tiers to steep spectra at the automatic prefix width and leaves flat ones alone.
void policy_test() {
  std::mt19937 random(26);
  const auto steep=decaying_rows(rows,random),flat=decaying_rows(rows,random,flat_decay);
  for (uint32_t bits_per_dim=dv::tieredq_detail::additive_wide_max_bits;bits_per_dim<=dv::tieredq_detail::widest_code_bits;++bits_per_dim) {
    const auto codec=train(steep,bits_per_dim,dv::TieredQFineTiers::automatic);
    require(codec.fast_prefix_bits()==bits_per_dim && widest(codec)>bits_per_dim,"automatic training skipped fine tiers");
    const auto plain=train(flat,bits_per_dim,dv::TieredQFineTiers::automatic);
    require(!plain.fast_prefix_bits() && widest(plain)<=bits_per_dim,"automatic training added fine tiers to a flat spectrum");
    const auto off=train(steep,bits_per_dim,dv::TieredQFineTiers::off);
    require(!off.fast_prefix_bits() && widest(off)<=bits_per_dim,"fine tiers switched off were added");
    const auto restored=dv::TieredQ::deserialize(codec.serialize());
    require(restored.fast_prefix_bits()==codec.fast_prefix_bits() && restored.serialize()==codec.serialize(),
            "fine-tier model round trip");
  }
  for (auto [target,metric]:{std::pair{dv::TieredQIndexMode::full,dv::TieredQMetric::inner_product},
                             std::pair{dv::TieredQIndexMode::fast,dv::TieredQMetric::l2}}) {
    bool rejected=false;
    try { train(steep,dv::tieredq_detail::widest_code_bits,dv::TieredQFineTiers::on,target,metric); } catch (const std::runtime_error&) { rejected=true; }
    require(rejected,"a full or L2 target accepted fine tiers");
    require(!train(steep,dv::tieredq_detail::widest_code_bits,dv::TieredQFineTiers::automatic,target,metric).fast_prefix_bits(),
            "automatic fine tiers reached a full or L2 target");
  }
}

void rescoring_test(uint32_t bits_per_dim) {
  std::mt19937 random(26);
  const auto data=decaying_rows(rows,random),probe=decaying_rows(queries,random);
  const auto codec=train(data,bits_per_dim,dv::TieredQFineTiers::on);
  require(codec.fast_prefix_bits()==bits_per_dim && widest(codec)>bits_per_dim+2,
          "fixture allocates no tier 3 bits past its prefix");
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

  // Only the flat index serves fine-tier models.
  const auto rejects=[](auto&& build) { try { build(); } catch (const std::runtime_error&) { return true; } return false; };
  require(rejects([&] { dv::TieredQIVFIndex ivf(codec); }),"IVF accepted a fine-tier model");
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
}
} // namespace

void tieredq_fast_fine_tests() {
  refined_book_test();
  policy_test();
  for (uint32_t bits_per_dim=dv::tieredq_detail::additive_wide_max_bits;bits_per_dim<=dv::tieredq_detail::widest_code_bits;++bits_per_dim)
    rescoring_test(bits_per_dim);
}
