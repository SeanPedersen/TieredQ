// Trained exhaustive index with turbovec-style ergonomics: construct with a
// budget, then add/search/save/load. The first non-empty add() trains the
// codec on that batch. Mutations and searches may run concurrently; the GIL
// is released for every native call.
#include "arrays.h"
#include "search.h"
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <algorithm>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>

namespace tieredq_python {
using namespace nb::literals;
namespace {
// A training sample smaller than twice the cap cannot seed that many
// centroids, so the default shrinks to fit instead of failing at train time.
constexpr uint32_t maximum_auto_centroids = 256;
constexpr uint64_t training_seed = 0;
uint32_t auto_centroids(size_t sample_rows) {
  return std::min(maximum_auto_centroids, static_cast<uint32_t>(sample_rows / 2));
}
struct SearchResult {
  std::vector<float> scores;
  std::vector<uint32_t> ids;
  std::vector<size_t> shape;
};
}  // namespace

class PythonIndex {
 public:
  PythonIndex(uint32_t dimensions, uint32_t code_bits)
      : config_(dv::TieredQConfig::target_code_bits(dimensions, code_bits)) {}

  size_t size() const {
    std::shared_lock lock(mutex_);
    return index_ ? index_->size() : 0;
  }
  uint32_t dimensions() const { return config_.dimensions; }
  uint32_t code_bits() const { return config_.code_bits; }
  size_t bytes_per_vector() const {
    std::shared_lock lock(mutex_);
    return trained().codec().bytes_per_vector();
  }
  void add(Matrix rows) {
    require_width(rows.shape(1), config_.dimensions);
    std::unique_lock lock(mutex_);
    if (!index_) {
      if (!rows.shape(0)) return;
      if (rows.shape(0) < 2)
        throw std::runtime_error("the first add() needs at least 2 rows to train");
      config_.num_centroids = auto_centroids(rows.shape(0));
      index_.emplace(dv::TieredQ::train(config_, rows.data(), rows.shape(0), training_seed));
    }
    index_->add(rows.data(), rows.shape(0));
  }
  SearchResult search(FloatArray queries, size_t k) const {
    const size_t rank = queries.ndim();
    if (rank != 1 && rank != 2)
      throw std::invalid_argument("queries must be a 1-D or 2-D float32 array");
    const size_t rows = rank == 1 ? 1 : queries.shape(0);
    const size_t width = rank == 1 ? queries.shape(0) : queries.shape(1);
    require_width(width, config_.dimensions);
    std::shared_lock lock(mutex_);
    const dv::TieredQIndex& index = trained();
    const size_t found = std::min(k, index.size());
    SearchResult result{std::vector<float>(array_elements(rows, found), 0.0f),
                        std::vector<uint32_t>(array_elements(rows, found), 0),
                        rank == 1 ? std::vector<size_t>{found} : std::vector<size_t>{rows, found}};
    search_rows(index, queries.data(), rows, width, k,
                result.scores.data(), result.ids.data());
    return result;
  }
  void save(const std::string& path) const {
    std::shared_lock lock(mutex_);
    trained().save(path);
  }
  static std::unique_ptr<PythonIndex> load(const std::string& path) {
    return std::unique_ptr<PythonIndex>(new PythonIndex(dv::TieredQIndex::load(path)));
  }

 private:
  explicit PythonIndex(dv::TieredQIndex index)
      : config_(index.codec().config()), index_(std::move(index)) {}
  const dv::TieredQIndex& trained() const {
    if (!index_) throw std::runtime_error("add() at least 2 rows before searching");
    return *index_;
  }
  dv::TieredQIndex& trained() {
    if (!index_) throw std::runtime_error("add() at least 2 rows before searching");
    return *index_;
  }

  dv::TieredQConfig config_;
  std::optional<dv::TieredQIndex> index_;
  mutable std::shared_mutex mutex_;
};

void bind_index(nb::module_& module) {
  nb::class_<PythonIndex>(module, "TieredQIndex", "Trained exhaustive in-memory vector index.")
      .def(nb::init<uint32_t, uint32_t>(), "dimensions"_a, "code_bits"_a)
      .def("__len__", &PythonIndex::size, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("dimensions", &PythonIndex::dimensions)
      .def_prop_ro("code_bits", &PythonIndex::code_bits)
      .def_prop_ro("bytes_per_vector", &PythonIndex::bytes_per_vector,
                   nb::call_guard<nb::gil_scoped_release>())
      .def("add", &PythonIndex::add, "rows"_a.noconvert(),
           nb::call_guard<nb::gil_scoped_release>())
      .def("search", [](const PythonIndex& self, FloatArray queries, size_t k) {
        SearchResult result;
        { nb::gil_scoped_release release; result = self.search(queries, k); }
        return nb::make_tuple(owned_array(std::move(result.scores), result.shape),
                              owned_array(std::move(result.ids), result.shape));
      }, "queries"_a.noconvert(), "k"_a = 10)
      .def("save", &PythonIndex::save, "path"_a, nb::call_guard<nb::gil_scoped_release>())
      .def_static("load", &PythonIndex::load, "path"_a, nb::call_guard<nb::gil_scoped_release>());
}
}  // namespace tieredq_python
