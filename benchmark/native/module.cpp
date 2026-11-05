// Benchmark adapters: own compressed indexes, expose full native batch calls and storage.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <numeric>
#include <random>
#include <tieredq/tieredq.h>
#include <tieredq/tieredq_nested_flat.h>
#include "rabitq_flat.h"
#include <filesystem>
#include <chrono>
#include <memory>
#include <omp.h>
#if defined(__OPTIMIZE_SIZE__)
#error "Codec benchmarks require -O3, not size optimization"
#endif
namespace nb = nanobind;
using Matrix = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Ids = nb::ndarray<uint32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
constexpr size_t top_k = 10;
constexpr bool rabitq_high_accuracy = benchmark::RaBitQFlat::high_accuracy;
constexpr size_t training_limit = 32768;
class Index {
  std::unique_ptr<dv::TieredQIndex> tq_;
  std::unique_ptr<benchmark::RaBitQFlat> rq_;
  size_t width_, count_;
  double encode_s_ = 0;
  Index(std::unique_ptr<dv::TieredQIndex> tq, std::unique_ptr<benchmark::RaBitQFlat> rq, size_t width, size_t count)
      : tq_(std::move(tq)), rq_(std::move(rq)), width_(width), count_(count) {}
 public:
  static Index load(const std::string& path, bool tieredq) {
    if (!tieredq) {
      auto rq = benchmark::RaBitQFlat::load(path);
      const size_t width = rq->dimensions(), count = rq->size();
      return Index(nullptr, std::move(rq), width, count);
    }
    auto tq = std::make_unique<dv::TieredQIndex>(dv::TieredQIndex::load(path));
    const size_t width = tq->codec().config().dimensions, count = tq->size();
    return Index(std::move(tq), nullptr, width, count);
  }
  // code_bits overrides the TieredQ per-vector code budget; 0 keeps bits per effective dimension. native_metric
  // passes raw L2 rows to TieredQ's own L2 metric; otherwise L2 rows arrive lifted by the caller (one extra column).
  // exact serves a full-target model through the fast index (exact_fast_scores in the Python bindings). Full mode
  // trains without fine tiers, which only the fast index serves.
  Index(Matrix base, size_t bits, uint64_t seed, bool tieredq, bool l2, bool fast = true, size_t code_bits = 0,
        bool native_metric = false, bool exact = false, bool fine_tiers = true)
      : width_(base.shape(1)), count_(base.shape(0)) {
    if (count_ < top_k || !width_ || bits < 1 || bits > 8) throw std::invalid_argument("invalid index shape/bits");
    if (exact && !fast) throw std::invalid_argument("exact scores need the fast index");
    if (tieredq) {
      const bool lifted = l2 && !native_metric;
      const size_t budget = code_bits ? code_bits : bits * (width_ - (lifted ? 1 : 0));
      auto config = dv::TieredQConfig::target_code_bits(width_, budget);
      if (l2 && native_metric) config.metric = dv::TieredQMetric::l2;
      config.num_centroids = std::min<size_t>(256, count_ / 2);
      config.target_mode = fast && !exact ? dv::TieredQIndexMode::fast : dv::TieredQIndexMode::full;
      config.fine_tiers = fine_tiers && fast;
      const size_t train_count = std::min(count_, training_limit);
      std::vector<uint32_t> order(count_);
      std::iota(order.begin(), order.end(), 0);
      std::mt19937_64 rng(seed);
      std::shuffle(order.begin(), order.end(), rng);
      // The first train_count shuffled rows, read in place (the same model as training on them gathered).
      tq_ = std::make_unique<dv::TieredQIndex>(dv::TieredQ::train_sampled(config, base.data(), count_, order.data(), train_count, seed), fast ? dv::TieredQIndexMode::fast : dv::TieredQIndexMode::full);
      const auto start = std::chrono::steady_clock::now();
      tq_->add(base.data(), count_);
      encode_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      return;
    }
    rq_ = std::make_unique<benchmark::RaBitQFlat>(base.data(),count_,width_,bits,l2);
    encode_s_ = rq_->encode_seconds();
  }
  double encode_seconds() const { return encode_s_; }
  nb::dict diagnostics() const {
    if (!tq_) throw std::invalid_argument("TieredQ diagnostics require a TieredQ index");
    const auto& codec = tq_->codec();
    nb::list tiers;
    for (const auto& tier : codec.tiers()) {
      nb::dict entry;
      entry["start"] = tier.start;
      entry["dimensions"] = tier.dimensions;
      entry["bits"] = tier.bits;
      tiers.append(entry);
    }
    nb::dict result;
    result["tiers"] = tiers;
    result["coded_dimensions"] = codec.coded_dimensions();
    result["packed_code_bytes"] = codec.code_bytes();
    result["vector_bytes"] = tq_->bytes_per_vector();
    result["allocated_vector_bytes"] = tq_->storage_bytes();
    result["model_serialized_bytes"] = codec.model_bytes();
    result["model_memory_bytes"] = codec.memory_bytes();
    result["model_serving_memory_bytes"] = dv::TieredQ::deserialize(codec.serialize()).memory_bytes();
    result["scalar_fast_scan"] = tq_->uses_scalar_fast_scan();
    return result;
  }
  // The ids width is the search depth k.
  void search(Matrix queries, Ids ids, int threads) const {
    const size_t rows=queries.shape(0), k=ids.shape(1);
    if (queries.shape(1)!=width_ || ids.shape(0)!=rows || k<1 || k>count_ || threads<1)
      throw std::invalid_argument("invalid search shapes/threads");
    omp_set_num_threads(threads);
    if (tq_) {
      std::vector<float> scores(rows*k);
      tq_->search_batch(queries.data(), rows, k, scores.data(), ids.data());
      return;
    }
    #pragma omp parallel for schedule(static) if(rows > 1)
    for (size_t i=0; i<rows; ++i) rq_->search(queries.data()+i*width_, k, ids.data()+i*k);
  }
  size_t save(const std::string& path) const {
    if (rq_) rq_->save(path); else tq_->save(path);
    return std::filesystem::file_size(path);
  }
  // Component allocation accounting of this instance; not RSS, excludes allocator bookkeeping.
  size_t resident_bytes() const {
    return rq_ ? rq_->storage_bytes() : tq_->storage_bytes()+tq_->codec().memory_bytes();
  }
  size_t storage(const std::string& path) const {
    if (rq_) { rq_->save(path.c_str()); return std::max<size_t>(std::filesystem::file_size(path),rq_->storage_bytes()); }
    tq_->save(path);
    // Serving footprint: a reloaded index holds the model's serving state; encoder state is built only on append.
    const auto served=dv::TieredQIndex::load(path);
    return std::max<size_t>(std::filesystem::file_size(path), served.storage_bytes()+served.codec().memory_bytes());
  }
};
// Flat nested TieredQ (#13): a prefix_bits prefix scanned for every row, the top `candidates` rescored with a
// TieredQRefinement tier to code_bits, both trained on the same seeded sample as Index's TieredQ path.
class NestedIndex {
  std::unique_ptr<dv::TieredQNestedIndex> nested_;
  size_t width_, candidates_;
  double encode_s_ = 0;
 public:
  NestedIndex(Matrix base, uint64_t seed, size_t prefix_bits, size_t code_bits, size_t candidates)
      : width_(base.shape(1)), candidates_(candidates) {
    const size_t count = base.shape(0);
    if (count < top_k || !width_ || candidates < top_k || prefix_bits >= code_bits)
      throw std::invalid_argument("invalid nested index shape/budgets/candidates");
    const size_t train_count = std::min(count, training_limit);
    std::vector<uint32_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<float> sample(train_count*width_);
    for (size_t i=0; i<train_count; ++i) std::copy_n(base.data()+size_t(order[i])*width_, width_, sample.data()+i*width_);
    auto config = dv::TieredQConfig::target_code_bits(width_, prefix_bits);
    config.num_centroids = std::min<size_t>(256, count / 2);
    auto prefix = dv::TieredQ::train(config, sample.data(), train_count, seed);
    auto refinement = dv::TieredQRefinement::train(prefix, code_bits, sample.data(), train_count, seed);
    const auto start = std::chrono::steady_clock::now();
    nested_ = std::make_unique<dv::TieredQNestedIndex>(std::move(prefix), std::move(refinement), base.data(), count);
    encode_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  }
  double encode_seconds() const { return encode_s_; }
  void search(Matrix queries, Ids ids, int threads) const {
    const size_t rows=queries.shape(0);
    if (queries.shape(1)!=width_ || ids.shape(0)!=rows || ids.shape(1)!=top_k || threads<1)
      throw std::invalid_argument("invalid search shapes/threads");
    omp_set_num_threads(threads);
    if (rows==1) {
      const auto hits = nested_->search(queries.data(), top_k, candidates_);
      for (size_t i=0; i<top_k; ++i) ids.data()[i] = i<hits.size() ? hits[i].id : UINT32_MAX;
      return;
    }
    std::vector<float> scores(rows*top_k);
    nested_->search_batch(queries.data(), rows, top_k, candidates_, scores.data(), ids.data());
  }
  // Serving footprint: the larger of the serialized index plus tier rows and the in-RAM accounting.
  size_t storage(const std::string&) const {
    const size_t tier = nested_->size()*nested_->nested().tier_bytes_per_vector();
    return std::max(nested_->nested().serialize().size()+tier, nested_->storage_bytes());
  }
};
NB_MODULE(codec_native, m) {
  m.attr("rabitq_high_accuracy") = rabitq_high_accuracy;
  m.attr("rabitq_implicit_ids") = true;
  m.attr("rabitq_container") = "direct-flat";
#if defined(__AVX512BW__)
  m.attr("compiled_avx512bw") = true;
#else
  m.attr("compiled_avx512bw") = false;
#endif
#if defined(__AVX2__)
  m.attr("compiled_avx2") = true;
#else
  m.attr("compiled_avx2") = false;
#endif
  m.def("set_threads", [](int n){ omp_set_dynamic(0); omp_set_num_threads(n); });
  nb::class_<Index>(m,"Index")
    .def(nb::init<Matrix,size_t,uint64_t,bool,bool,bool,size_t,bool,bool,bool>(), nb::arg("base"), nb::arg("bits"),
         nb::arg("seed"), nb::arg("tieredq"), nb::arg("l2"), nb::arg("fast") = true, nb::arg("code_bits") = 0,
         nb::arg("native_metric") = false, nb::arg("exact") = false, nb::arg("fine_tiers") = true,
         nb::call_guard<nb::gil_scoped_release>())
    .def("diagnostics", &Index::diagnostics)
    .def_prop_ro("encode_seconds", &Index::encode_seconds)
    .def("search", &Index::search, nb::call_guard<nb::gil_scoped_release>())
    .def("storage", &Index::storage)
    .def("save", &Index::save, nb::call_guard<nb::gil_scoped_release>())
    .def("resident_bytes", &Index::resident_bytes)
    .def_static("load", &Index::load, nb::call_guard<nb::gil_scoped_release>());
  nb::class_<NestedIndex>(m,"NestedIndex")
    .def(nb::init<Matrix,uint64_t,size_t,size_t,size_t>(), nb::call_guard<nb::gil_scoped_release>())
    .def_prop_ro("encode_seconds", &NestedIndex::encode_seconds)
    .def("search", &NestedIndex::search, nb::call_guard<nb::gil_scoped_release>())
    .def("storage", &NestedIndex::storage);
}
