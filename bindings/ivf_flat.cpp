// Unquantized IVF (#56): build on float32 rows, batch search, and knn_graph (every row's k nearest other rows, for
// dimensionality reduction). Native calls release the GIL; a shared mutex lets searches run concurrently and makes
// build exclusive.
#include "arrays.h"
#include "training.h"
#include <tieredq/tieredq_ivf_flat.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <mutex>
#include <optional>
#include <shared_mutex>

namespace tieredq_python {
using namespace nb::literals;

class PythonIVFFlatIndex {
 public:
  PythonIVFFlatIndex(uint32_t dimensions, std::optional<uint32_t> lists, const std::string& metric,
                     uint32_t iterations, uint32_t sample_rows_per_list, uint64_t seed)
      : index_(dimensions, config(lists, metric, iterations, sample_rows_per_list, seed)) {}

  void build(Matrix rows) {
    require_width(rows.shape(1), index_.dimensions());
    std::unique_lock lock(mutex_);
    index_.build(rows.data(), rows.shape(0));
  }
  SearchResult search(const FloatArray& queries, size_t k, size_t nprobe) const {
    const auto shape = query_rows(queries, index_.dimensions());
    std::shared_lock lock(mutex_);
    auto result = empty_result(shape, k);
    index_.search(queries.data(), shape.rows, k, nprobe, result.scores.data(), result.ids.data());
    return result;
  }
  SearchResult knn_graph(size_t k, size_t nprobe) const {
    std::shared_lock lock(mutex_);
    auto result = empty_result({index_.size(), false}, k);
    index_.knn_graph(k, nprobe, result.scores.data(), result.ids.data());
    return result;
  }
  size_t size() const { std::shared_lock lock(mutex_); return index_.size(); }
  uint32_t dimensions() const { return index_.dimensions(); }
  std::optional<size_t> lists() const {
    std::shared_lock lock(mutex_);
    if (index_.lists()) return index_.lists();
    return index_.config().lists ? std::optional<size_t>(index_.config().lists) : std::nullopt;
  }
  const char* metric() const { return metric_name(index_.config().metric); }
  size_t storage_bytes() const { std::shared_lock lock(mutex_); return index_.storage_bytes(); }
  std::vector<uint32_t> list_assignment() const { std::shared_lock lock(mutex_); return index_.list_assignment(); }
  // Needs the GIL (returns a dict); the copy under the lock is cheap.
  nb::dict build_stats() const {
    tq::TieredQIVFFlatBuildStats stats;
    { std::shared_lock lock(mutex_); stats = index_.build_stats(); }
    nb::dict out;
    out["train_s"] = stats.train_s; out["assign_s"] = stats.assign_s; out["store_s"] = stats.store_s;
    return out;
  }

 private:
  static tq::TieredQIVFFlatConfig config(std::optional<uint32_t> lists, const std::string& metric, uint32_t iterations,
                                         uint32_t sample_rows_per_list, uint64_t seed) {
    tq::TieredQIVFFlatConfig config;
    if (lists && *lists == 0) throw std::invalid_argument("lists must be positive (None: automatic)");
    config.lists = lists.value_or(0);
    config.metric = parse_metric(metric);
    config.iterations = iterations;
    config.sample_rows_per_list = sample_rows_per_list;
    config.seed = seed;
    return config;
  }
  tq::TieredQIVFFlatIndex index_;
  mutable std::shared_mutex mutex_;
};

void bind_ivf_flat(nb::module_& module) {
  const tq::TieredQIVFFlatConfig defaults;
  nb::class_<PythonIVFFlatIndex>(module, "TieredQIVFFlatIndex",
      "Unquantized (float32) inverted-file index with a list-major batch engine: queries grouped by their nearest\n"
      "list are scored against each probed list in one GEMM. knn_graph() is the all-rows self-join that\n"
      "dimensionality reduction needs.")
      .def(nb::init<uint32_t, std::optional<uint32_t>, const std::string&, uint32_t, uint32_t, uint64_t>(),
           "dimensions"_a, nb::kw_only(), "lists"_a = nb::none(), "metric"_a = "l2",
           "iterations"_a = defaults.iterations, "sample_rows_per_list"_a = defaults.sample_rows_per_list,
           "seed"_a = defaults.seed,
           "lists None (default) picks about sqrt(rows), a power of two, at build. metric is 'l2' (scores are\n"
           "-squared distances), 'ip' or 'cosine'. Lists train by k-means (iterations) on a seeded sample of\n"
           "sample_rows_per_list rows per list.")
      .def("__len__", &PythonIVFFlatIndex::size, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("dimensions", &PythonIVFFlatIndex::dimensions)
      .def_prop_ro("lists", &PythonIVFFlatIndex::lists, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("metric", &PythonIVFFlatIndex::metric)
      .def_prop_ro("storage_bytes", &PythonIVFFlatIndex::storage_bytes, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("build_stats", &PythonIVFFlatIndex::build_stats,
                   "Wall seconds of the last build: train_s (k-means), assign_s, store_s.")
      .def("build", &PythonIVFFlatIndex::build, "rows"_a.noconvert(), nb::call_guard<nb::gil_scoped_release>(),
           "Trains the lists on rows (float32, rows x dimensions) and stores them; replaces earlier content.")
      .def("search", [](const PythonIVFFlatIndex& self, FloatArray queries, size_t k, size_t nprobe) {
        SearchResult result;
        { nb::gil_scoped_release release; result = self.search(queries, k, nprobe); }
        return result_tuple(std::move(result));
      }, "queries"_a.noconvert(), "k"_a = 10, nb::kw_only(), "nprobe"_a,
         "(scores, ids) of the k best rows over each query's nprobe nearest lists; slots past the probed rows\n"
         "hold id 2**32 - 1 and score -inf.")
      .def("knn_graph", [](const PythonIVFFlatIndex& self, size_t k, size_t nprobe) {
        SearchResult result;
        { nb::gil_scoped_release release; result = self.knn_graph(k, nprobe); }
        return result_tuple(std::move(result));
      }, "k"_a = 15, nb::kw_only(), "nprobe"_a,
         "(scores, ids), each len(index) x k: every row's k nearest other rows over its nprobe nearest lists.\n"
         "Recall equals search() of the rows at the same nprobe.")
      .def("list_assignment", [](const PythonIVFFlatIndex& self) {
        std::vector<uint32_t> lists;
        { nb::gil_scoped_release release; lists = self.list_assignment(); }
        const size_t rows = lists.size();
        return owned_array(std::move(lists), {rows});
      }, "The list of every row, in row order.");
}
}  // namespace tieredq_python
