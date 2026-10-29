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
dv::TieredQIndexMode parse_index_mode(const std::string& mode) {
  if (mode=="fast") return dv::TieredQIndexMode::fast;
  if (mode=="full") return dv::TieredQIndexMode::full;
  throw std::invalid_argument("mode must be 'fast' or 'full'");
}
struct SearchResult {
  std::vector<float> scores;
  std::vector<uint32_t> ids;
  std::vector<size_t> shape;
};
}  // namespace

class PythonIndex {
 public:
  PythonIndex(uint32_t dimensions, uint32_t code_bits, const std::string& mode)
      : config_(dv::TieredQConfig::target_code_bits(dimensions, code_bits)), mode_(parse_index_mode(mode)) {
    config_.target_mode = mode_;
  }

  size_t size() const {
    std::shared_lock lock(mutex_);
    return index_ ? index_->size() : 0;
  }
  uint32_t dimensions() const { return config_.dimensions; }
  // The requested budget until add() trains; then the model's realised bits, which fast nibble targets may round
  // to their padded storage (#9).
  uint32_t code_bits() const {
    std::shared_lock lock(mutex_);
    return index_ ? index_->codec().config().code_bits : config_.code_bits;
  }
  const char* mode() const { return mode_==dv::TieredQIndexMode::fast ? "fast" : "full"; }
  size_t storage_bytes() const {
    std::shared_lock lock(mutex_);
    return index_ ? index_->storage_bytes() : 0;
  }
  size_t bytes_per_vector() const {
    std::shared_lock lock(mutex_);
    return trained().bytes_per_vector();
  }
  void add(Matrix rows) {
    require_width(rows.shape(1), config_.dimensions);
    std::unique_lock lock(mutex_);
    if (!index_) {
      if (!rows.shape(0)) return;
      if (rows.shape(0) < 2)
        throw std::runtime_error("the first add() needs at least 2 rows to train");
      config_.num_centroids = auto_centroids(rows.shape(0));
      index_.emplace(dv::TieredQ::train(config_, rows.data(), rows.shape(0), training_seed), mode_);
    }
    index_->add(rows.data(), rows.shape(0));
  }
  void build_fast_scan() {
    std::unique_lock lock(mutex_);
    trained().build_fast_scan();
  }
  bool has_fast_scan() const {
    std::shared_lock lock(mutex_);
    return index_ && index_->has_fast_scan();
  }
  bool uses_scalar_fast_scan() const {
    std::shared_lock lock(mutex_);
    return index_ && index_->uses_scalar_fast_scan();
  }
  void warn_if_scalar() {
    if (scalar_warning_emitted_ || !uses_scalar_fast_scan()) return;
    const char* message=dv::TieredQIndex::native_fast_scan_available()
        ? "TieredQ fast search is using the portable scalar scorer because this index's representation has no compatible SIMD kernel on this CPU. Rebuild the index from the original vectors on this machine to use its native fast representation."
        : "TieredQ fast search is using the portable scalar scorer because this CPU has no supported native fast kernel. Rebuilding on this CPU will not enable SIMD acceleration.";
    scalar_warning_emitted_=true;
    if (PyErr_WarnEx(PyExc_RuntimeWarning,message,1)<0) {
      scalar_warning_emitted_=false;
      throw nb::python_error();
    }
  }
  SearchResult search(FloatArray queries, size_t k, SearchMode mode) const {
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
                result.scores.data(), result.ids.data(), mode);
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
      : config_(index.codec().config()), mode_(index.mode()), index_(std::move(index)) {}
  const dv::TieredQIndex& trained() const {
    if (!index_) throw std::runtime_error("add() at least 2 rows before searching");
    return *index_;
  }
  dv::TieredQIndex& trained() {
    if (!index_) throw std::runtime_error("add() at least 2 rows before searching");
    return *index_;
  }

  dv::TieredQConfig config_;
  dv::TieredQIndexMode mode_;
  bool scalar_warning_emitted_ = false;
  std::optional<dv::TieredQIndex> index_;
  mutable std::shared_mutex mutex_;
};

namespace {
nb::tuple search_result(const PythonIndex& index, FloatArray queries, size_t k, SearchMode mode) {
  SearchResult result;
  { nb::gil_scoped_release release; result = index.search(queries, k, mode); }
  return nb::make_tuple(owned_array(std::move(result.scores), result.shape),
                        owned_array(std::move(result.ids), result.shape));
}
}  // namespace

void bind_index(nb::module_& module) {
  nb::class_<PythonIndex>(module, "TieredQIndex", "Trained exhaustive in-memory vector index.")
      .def(nb::init<uint32_t, uint32_t, const std::string&>(), "dimensions"_a, "code_bits"_a, nb::kw_only(), "mode"_a = "fast")
      .def("__len__", &PythonIndex::size, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("dimensions", &PythonIndex::dimensions)
      .def_prop_ro("code_bits", &PythonIndex::code_bits,
                   "Requested code bits per vector before training; the trained model's realised bits after.",
                   nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("mode", &PythonIndex::mode)
      .def_prop_ro("storage_bytes", &PythonIndex::storage_bytes,
                   "Allocated vector-data bytes, excluding shared models and query scratch.",
                   nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("bytes_per_vector", &PythonIndex::bytes_per_vector,
                   nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("uses_scalar_fast_scan", &PythonIndex::uses_scalar_fast_scan,
                   nb::call_guard<nb::gil_scoped_release>())
      .def("add", [](PythonIndex& self,Matrix rows) {
        { nb::gil_scoped_release release; self.add(rows); }
        self.warn_if_scalar();
      }, "rows"_a.noconvert())
      .def("search", [](const PythonIndex& self, FloatArray queries, size_t k) {
        return search_result(self, queries, k, SearchMode::configured);
      }, "queries"_a.noconvert(), "k"_a = 10)
      .def("build_fast_scan", &PythonIndex::build_fast_scan,
           "Build the optional approximate full-scan cache; unsupported CPUs retain exact fallback.",
           nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("has_fast_scan", &PythonIndex::has_fast_scan,
                   nb::call_guard<nb::gil_scoped_release>())
      .def("search_fast", [](const PythonIndex& self, FloatArray queries, size_t k) {
        return search_result(self, queries, k, SearchMode::fast);
      }, "queries"_a.noconvert(), "k"_a = 10,
         "Approximate full scan; falls back to exact search when the cache is absent.")
      .def("save", &PythonIndex::save, "path"_a, nb::call_guard<nb::gil_scoped_release>())
      .def_static("load", [](const std::string& path) {
        std::unique_ptr<PythonIndex> index;
        { nb::gil_scoped_release release; index=PythonIndex::load(path); }
        index->warn_if_scalar();
        return index;
      }, "path"_a);
}
}  // namespace tieredq_python
