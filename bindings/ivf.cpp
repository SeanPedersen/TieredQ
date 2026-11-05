// Fast-mode inverted-file index: construct with a budget and a list count, then build/search/save/load. The codec
// trains on a seeded sample of the first training rows (train_lists' sample, else build()'s rows), as the flat index
// does. build() replaces the content; searches may run concurrently with it. The GIL is released for every native call.
#include "arrays.h"
#include "training.h"
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>

namespace tieredq_python {
using namespace nb::literals;
using Pruning = std::optional<dv::TieredQPruning>;

class PythonIVFIndex {
 public:
  PythonIVFIndex(uint32_t dimensions, uint32_t code_bits, uint32_t lists, const std::string& metric)
      : config_(dv::TieredQConfig::target_code_bits(dimensions, code_bits)), lists_(lists) {
    config_.metric = parse_metric(metric);
    config_.fine_tiers = false;  // only the flat index serves fine tiers
    config_.validate();
  }
  size_t size() const {
    std::shared_lock lock(mutex_);
    return index_ ? index_->size() : 0;
  }
  uint32_t dimensions() const { return config_.dimensions; }
  uint32_t code_bits() const {
    std::shared_lock lock(mutex_);
    return index_ ? index_->codec().config().code_bits : config_.code_bits;
  }
  const char* metric() const { return metric_name(config_.metric); }
  // The requested count until training (0: one list per codec centroid); then the index's lists.
  size_t lists() const {
    std::shared_lock lock(mutex_);
    return index_ ? index_->lists() : lists_;
  }
  size_t storage_bytes() const {
    std::shared_lock lock(mutex_);
    return index_ ? index_->storage_bytes() : 0;
  }
  void train_lists(Matrix sample) {
    require_width(sample.shape(1), config_.dimensions);
    std::unique_lock lock(mutex_);
    ensure_codec(sample.data(), sample.shape(0));
    index_->train_lists(sample.data(), sample.shape(0));
  }
  void build(Matrix rows) {
    require_width(rows.shape(1), config_.dimensions);
    std::unique_lock lock(mutex_);
    ensure_codec(rows.data(), rows.shape(0));
    index_->build(rows.data(), rows.shape(0));
  }
  SearchResult search(FloatArray queries, size_t k, size_t nprobe, Pruning pruning) const {
    const auto shape = query_rows(queries, config_.dimensions);
    std::shared_lock lock(mutex_);
    const auto& index = trained();
    auto result = empty_result(shape, std::min(k, index.size()));
    index.search_batch(queries.data(), shape.rows, k, nprobe, result.scores.data(), result.ids.data(),
                       pruning.value_or(dv::TieredQPruning{}));
    return result;
  }
  std::vector<uint32_t> list_assignment() const {
    std::shared_lock lock(mutex_);
    return trained().list_assignment();
  }
  void save(const std::filesystem::path& path) const {
    std::shared_lock lock(mutex_);
    trained().save(path.string());
  }
  static std::unique_ptr<PythonIVFIndex> load(const std::filesystem::path& path) {
    return std::unique_ptr<PythonIVFIndex>(new PythonIVFIndex(dv::TieredQIVFIndex::load(path.string())));
  }

 private:
  explicit PythonIVFIndex(dv::TieredQIVFIndex index)
      : config_(index.codec().config()), lists_(uint32_t(index.lists())), index_(std::move(index)) {}
  void ensure_codec(const float* rows, size_t count) {
    if (index_) return;
    index_.emplace(train_codec(config_, rows, count), lists_, training_seed);
  }
  const dv::TieredQIVFIndex& trained() const {
    if (!index_) throw std::runtime_error("build() the index before using it");
    return *index_;
  }

  dv::TieredQConfig config_;
  uint32_t lists_;
  std::optional<dv::TieredQIVFIndex> index_;
  mutable std::shared_mutex mutex_;
};

void bind_ivf(nb::module_& module) {
  nb::class_<dv::TieredQPruning>(module, "TieredQPruning",
      "Range-checkpoint pruning of IVF list scans (nibble scans: AVX-512, AVX2, NEON; FP32 code scans: AVX-512).\n"
      "confidence 0 is exact; a positive confidence bounds the suffix score at that many standard deviations:\n"
      "approximate and faster. prefix_tiers 0 (default) sizes the prefix automatically.")
      .def("__init__", [](dv::TieredQPruning* pruning, double confidence, uint32_t prefix_tiers) {
        new (pruning) dv::TieredQPruning{confidence, prefix_tiers};
      }, "confidence"_a = dv::TieredQPruning{}.confidence, "prefix_tiers"_a = dv::TieredQPruning{}.prefix_tiers)
      .def_rw("confidence", &dv::TieredQPruning::confidence)
      .def_rw("prefix_tiers", &dv::TieredQPruning::prefix_tiers);

  nb::class_<PythonIVFIndex>(module, "TieredQIVFIndex", "Fast-mode inverted-file vector index.")
      .def(nb::init<uint32_t, uint32_t, uint32_t, const std::string&>(),
           "dimensions"_a, "code_bits"_a, nb::kw_only(), "lists"_a = 0, "metric"_a = "ip",
           "lists 0 makes one list per codec centroid at no extra storage; otherwise a coarse quantizer with that\n"
           "many lists is trained. metric is 'ip', 'cosine' or 'l2' (scores are -squared distances).")
      .def("__len__", &PythonIVFIndex::size, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("dimensions", &PythonIVFIndex::dimensions)
      .def_prop_ro("code_bits", &PythonIVFIndex::code_bits, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("metric", &PythonIVFIndex::metric)
      .def_prop_ro("lists", &PythonIVFIndex::lists, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("storage_bytes", &PythonIVFIndex::storage_bytes,
                   "Fast representation (including list padding), list offsets and the id map.",
                   nb::call_guard<nb::gil_scoped_release>())
      .def("train_lists", &PythonIVFIndex::train_lists, "sample"_a.noconvert(),
           "Trains the codec (if untrained) and the coarse lists on sample, so later builds keep this partition.",
           nb::call_guard<nb::gil_scoped_release>())
      .def("build", &PythonIVFIndex::build, "rows"_a.noconvert(),
           "Replaces the content with rows; the first training (here or in train_lists) also trains the codec.",
           nb::call_guard<nb::gil_scoped_release>())
      .def("search", [](const PythonIVFIndex& self, FloatArray queries, size_t k, size_t nprobe, Pruning pruning) {
        SearchResult result;
        { nb::gil_scoped_release release; result = self.search(queries, k, nprobe, pruning); }
        return result_tuple(std::move(result));
      }, "queries"_a.noconvert(), "k"_a = 10, nb::kw_only(), "nprobe"_a, "pruning"_a = nb::none(),
         "Scans the nearest nprobe lists; slots past the probed rows hold id 2**32 - 1 and score -inf.")
      .def("list_assignment", [](const PythonIVFIndex& self) {
        std::vector<uint32_t> lists;
        { nb::gil_scoped_release release; lists = self.list_assignment(); }
        const size_t rows = lists.size();
        return owned_array(std::move(lists), {rows});
      }, "The list of every input row, in row order.")
      .def("save", &PythonIVFIndex::save, "path"_a, nb::call_guard<nb::gil_scoped_release>())
      .def_static("load", &PythonIVFIndex::load, "path"_a, nb::call_guard<nb::gil_scoped_release>());
}
}  // namespace tieredq_python
