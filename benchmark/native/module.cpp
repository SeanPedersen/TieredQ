// Benchmark adapters: own compressed indexes, expose full native batch calls and storage.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <numeric>
#include <random>
#include <tieredq/tieredq.h>
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
  Index(Matrix base, size_t bits, uint64_t seed, bool tieredq, bool l2, bool fast = true)
      : width_(base.shape(1)), count_(base.shape(0)) {
    if (count_ < top_k || !width_ || bits < 1 || bits > 8) throw std::invalid_argument("invalid index shape/bits");
    if (tieredq) {
      auto config = dv::TieredQConfig::target_code_bits(width_, bits * (width_ - (l2 ? 1 : 0)));
      config.num_centroids = std::min<size_t>(256, count_ / 2);
      const size_t train_count = std::min(count_, training_limit);
      std::vector<size_t> order(count_);
      std::iota(order.begin(), order.end(), 0);
      std::mt19937_64 rng(seed);
      std::shuffle(order.begin(), order.end(), rng);
      std::vector<float> sample(train_count * width_);
      for (size_t i=0; i<train_count; ++i)
        std::copy_n(base.data()+order[i]*width_, width_, sample.data()+i*width_);
      tq_ = std::make_unique<dv::TieredQIndex>(dv::TieredQ::train(config, sample.data(), train_count, seed), fast ? dv::TieredQIndexMode::fast : dv::TieredQIndexMode::full);
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
  void search(Matrix queries, Ids ids, int threads) const {
    const size_t rows=queries.shape(0);
    if (queries.shape(1)!=width_ || ids.shape(0)!=rows || ids.shape(1)!=top_k || threads<1)
      throw std::invalid_argument("invalid search shapes/threads");
    omp_set_num_threads(threads);
    if (tq_) {
      std::vector<float> scores(rows*top_k);
      tq_->search_batch(queries.data(), rows, top_k, scores.data(), ids.data());
      return;
    }
    #pragma omp parallel for schedule(static) if(rows > 1)
    for (size_t i=0; i<rows; ++i) rq_->search(queries.data()+i*width_, top_k, ids.data()+i*top_k);
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
    .def(nb::init<Matrix,size_t,uint64_t,bool,bool>(), nb::call_guard<nb::gil_scoped_release>())
    .def(nb::init<Matrix,size_t,uint64_t,bool,bool,bool>(), nb::call_guard<nb::gil_scoped_release>())
    .def("diagnostics", &Index::diagnostics)
    .def_prop_ro("encode_seconds", &Index::encode_seconds)
    .def("search", &Index::search, nb::call_guard<nb::gil_scoped_release>())
    .def("storage", &Index::storage)
    .def("save", &Index::save, nb::call_guard<nb::gil_scoped_release>())
    .def("resident_bytes", &Index::resident_bytes)
    .def_static("load", &Index::load, nb::call_guard<nb::gil_scoped_release>());
}
