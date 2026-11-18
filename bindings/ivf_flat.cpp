// Flat IVF (#56): build on float32 rows or an SQ8 store, FP32 or SQ8 row storage, batch search, and knn_graph (every
// row's k nearest other rows, for dimensionality reduction). Native calls release the GIL; a shared mutex lets searches run concurrently and makes
// build exclusive.
#include "arrays.h"
#include "rows.h"
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
                     const std::string& storage, const std::string& trainer, uint32_t iterations, double epochs,
                     uint32_t sample_rows_per_list, uint64_t seed)
      : index_(dimensions, config(lists, metric, storage, trainer, iterations, epochs, sample_rows_per_list, seed)) {}

  void build(Matrix rows) {
    require_width(rows.shape(1), index_.dimensions());
    std::unique_lock lock(mutex_);
    index_.build(rows.data(), rows.shape(0));
  }
  void build_sq8(CodeMatrix codes, FloatVector scales, FloatVector offsets) {
    const size_t width = index_.dimensions();
    require_width(codes.shape(1), width);
    if (scales.shape(0) != width || offsets.shape(0) != width)
      throw std::invalid_argument("scales and offsets need one value per dimension");
    std::unique_lock lock(mutex_);
    index_.build(codes.data(), codes.shape(0), scales.data(), offsets.data());
  }
  void set_centroids(Matrix centroids) {
    require_width(centroids.shape(1), index_.dimensions());
    std::unique_lock lock(mutex_);
    index_.set_centroids(centroids.data(), centroids.shape(0));
  }
  void clear_centroids() { std::unique_lock lock(mutex_); index_.clear_centroids(); }
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
  const char* storage() const { return index_.config().storage == tq::TieredQIVFFlatStorage::sq8 ? "sq8" : "fp32"; }
  size_t storage_bytes() const { std::shared_lock lock(mutex_); return index_.storage_bytes(); }
  std::vector<uint32_t> list_assignment() const { std::shared_lock lock(mutex_); return index_.list_assignment(); }
  std::vector<float> centroids() const { std::shared_lock lock(mutex_); return index_.centroids(); }
  // Needs the GIL (returns a dict); the copy under the lock is cheap.
  nb::dict build_stats() const {
    tq::TieredQIVFFlatBuildStats stats;
    { std::shared_lock lock(mutex_); stats = index_.build_stats(); }
    nb::dict out;
    out["train_s"] = stats.train_s; out["assign_s"] = stats.assign_s; out["store_s"] = stats.store_s;
    return out;
  }

 private:
  static tq::TieredQIVFFlatConfig config(std::optional<uint32_t> lists, const std::string& metric,
                                         const std::string& storage, const std::string& trainer, uint32_t iterations,
                                         double epochs, uint32_t sample_rows_per_list, uint64_t seed) {
    tq::TieredQIVFFlatConfig config;
    if (lists && *lists == 0) throw std::invalid_argument("lists must be positive (None: automatic)");
    if (storage != "fp32" && storage != "sq8") throw std::invalid_argument("storage must be 'fp32' or 'sq8'");
    if (trainer != "lloyd" && trainer != "emeans") throw std::invalid_argument("trainer must be 'lloyd' or 'emeans'");
    config.lists = lists.value_or(0);
    config.metric = parse_metric(metric);
    config.storage = storage == "sq8" ? tq::TieredQIVFFlatStorage::sq8 : tq::TieredQIVFFlatStorage::fp32;
    config.trainer = trainer == "emeans" ? tq::TieredQIVFFlatTrainer::emeans : tq::TieredQIVFFlatTrainer::lloyd;
    config.iterations = iterations;
    config.epochs = epochs;
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
      "Flat inverted-file index with a list-major batch engine: queries grouped by their nearest list are scored\n"
      "against each probed list in one float32 GEMM. Rows are stored as float32 or SQ8 codes (a quarter of the\n"
      "RAM, decoded per probed list). knn_graph() is the all-rows self-join that dimensionality reduction needs.")
      .def(nb::init<uint32_t, std::optional<uint32_t>, const std::string&, const std::string&, const std::string&,
                    uint32_t, double, uint32_t, uint64_t>(),
           "dimensions"_a, nb::kw_only(), "lists"_a = nb::none(), "metric"_a = "l2", "storage"_a = "fp32",
           "trainer"_a = "emeans", "iterations"_a = defaults.iterations, "epochs"_a = defaults.epochs,
           "sample_rows_per_list"_a = defaults.sample_rows_per_list,
           "seed"_a = defaults.seed,
           "lists None (default) picks about sqrt(rows), a power of two, at build. metric is 'l2' (scores are\n"
           "-squared distances), 'ip' or 'cosine'. storage is 'fp32' or 'sq8' (int8 codes spanning each\n"
           "dimension's [min, max], or the codes given to build). Lists train on a seeded sample of\n"
           "sample_rows_per_list rows per list by trainer 'lloyd' (k-means, iterations) or 'emeans' (mini-batch EMA\n"
           "k-means, epochs: about Lloyd's quality in a quarter of the passes).")
      .def("__len__", &PythonIVFFlatIndex::size, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("dimensions", &PythonIVFFlatIndex::dimensions)
      .def_prop_ro("lists", &PythonIVFFlatIndex::lists, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("metric", &PythonIVFFlatIndex::metric)
      .def_prop_ro("storage", &PythonIVFFlatIndex::storage)
      .def_prop_ro("storage_bytes", &PythonIVFFlatIndex::storage_bytes, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("build_stats", &PythonIVFFlatIndex::build_stats,
                   "Wall seconds of the last build: train_s (k-means), assign_s, store_s.")
      .def("build", [](PythonIVFFlatIndex& self, OptionalMatrix rows, OptionalCodes codes, OptionalVector scales,
                       OptionalVector offsets) {
        if (rows.has_value() == codes.has_value()) throw std::invalid_argument("pass rows or codes, not both");
        if (codes && (!scales || !offsets)) throw std::invalid_argument("codes need scales and offsets");
        nb::gil_scoped_release release;
        if (rows) self.build(*rows); else self.build_sq8(*codes, *scales, *offsets);
      }, "rows"_a.noconvert() = nb::none(), nb::kw_only(), "codes"_a.noconvert() = nb::none(),
         "scales"_a.noconvert() = nb::none(), "offsets"_a.noconvert() = nb::none(),
         "Trains the lists on rows (float32, rows x dimensions) or an SQ8 store (codes= int8 rows x dimensions,\n"
         "e.g. a numpy memmap, with float32 scales= and offsets=: row = codes * scales + offsets, decoded a chunk\n"
         "at a time) and stores them; replaces earlier content. 'sq8' storage keeps given codes as they are\n"
         "(cosine re-encodes the normalized rows).")
      .def("set_centroids", &PythonIVFFlatIndex::set_centroids, "centroids"_a.noconvert(),
           nb::call_guard<nb::gil_scoped_release>(),
           "Lists for the next builds from these centroids (float32, lists x dimensions; normalized ones for\n"
           "cosine) instead of k-means, e.g. from another clustering method. build_stats train_s is then 0.")
      .def("clear_centroids", &PythonIVFFlatIndex::clear_centroids, nb::call_guard<nb::gil_scoped_release>(),
           "Back to k-means lists for the next builds.")
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
      }, "The list of every row, in row order.")
      .def("centroids", [](const PythonIVFFlatIndex& self) {
        std::vector<float> values;
        { nb::gil_scoped_release release; values = self.centroids(); }
        const size_t width = self.dimensions(), rows = width ? values.size() / width : 0;
        return owned_array(std::move(values), {rows, width});
      }, "The lists' centroids as routing ranks them (binary16-rounded), lists x dimensions float32.");
}
}  // namespace tieredq_python
