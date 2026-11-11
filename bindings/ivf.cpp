// Fast-mode inverted-file index: construct with a budget and a list count, then build/search/save/load. The codec
// trains on a seeded sample of the first training rows (train_lists' sample, else build()'s rows), as the flat index
// does. build() takes float rows, an SQ8 store or a row callable and replaces the content; searches may run
// concurrently with it. Many lists get the library's fast list training and router unless told otherwise. The GIL is
// released for every native call.
#include "arrays.h"
#include "lists.h"
#include "rows.h"
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

dv::TieredQRouting::Mode parse_routing(const std::string& name) {
  if (name == "auto") return dv::TieredQRouting::Mode::automatic;
  if (name == "router") return dv::TieredQRouting::Mode::router;
  if (name == "exact") return dv::TieredQRouting::Mode::exact;
  throw std::invalid_argument("routing must be 'auto', 'router' or 'exact'");
}

class PythonIVFIndex {
 public:
  // lists unset: TieredQIVFIndex::automatic_lists() of the first build's rows.
  PythonIVFIndex(uint32_t dimensions, uint32_t code_bits, std::optional<uint32_t> lists, const std::string& metric,
                 bool fine_tiers, bool exact_fast_scores, uint32_t max_scan_bits,
                 std::optional<dv::TieredQCoarseTraining> training, RouterChoice router)
      : config_(dv::TieredQConfig::target_code_bits(dimensions, code_bits)), lists_(lists), router_(router) {
    config_.metric = parse_metric(metric);
    config_.fine_tiers = fine_tiers;
    if (exact_fast_scores) config_.target_mode = dv::TieredQIndexMode::full;
    config_.max_scan_bits = max_scan_bits;
    config_.max_scan_bits = dv::TieredQIVFIndex::automatic_scan_bits(config_);
    config_.validate();
    if (training) set_coarse_training(*training);
    if (router_.use.value_or(false) && lists_ == 0u) throw std::invalid_argument("a router needs lists > 0");
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
  // The requested count until training (0: one list per codec centroid; unset: automatic); then the index's lists.
  std::optional<size_t> lists() const {
    std::shared_lock lock(mutex_);
    if (index_) return index_->lists();
    return lists_ ? std::optional<size_t>(*lists_) : std::nullopt;
  }
  size_t storage_bytes() const {
    std::shared_lock lock(mutex_);
    return index_ ? index_->storage_bytes() : 0;
  }
  void train_lists(Matrix sample) {
    require_width(sample.shape(1), config_.dimensions);
    require_lists("train_lists");
    std::unique_lock lock(mutex_);
    ensure_codec(float_rows(sample, config_.dimensions));
    index_->train_lists(sample.data(), sample.shape(0));
  }
  // Fixes the lists to given centroids (lists x dimensions); applied when the codec is trained, if it is not yet.
  void set_lists(Matrix centroids) {
    if (require_lists("set_lists") == 0) throw std::invalid_argument("set_lists needs lists > 0");
    require_width(centroids.shape(1), config_.dimensions);
    if (centroids.shape(0) != *lists_) throw std::invalid_argument("set_lists needs one centroid per list");
    std::unique_lock lock(mutex_);
    given_lists_.assign(centroids.data(), centroids.data() + centroids.shape(0) * centroids.shape(1));
    if (index_) index_->set_lists(given_lists_.data());
  }
  void build(const RowInput& rows) {
    std::unique_lock lock(mutex_);
    ensure_codec(rows);
    index_->build(rows.loader, rows.count);
  }
  SearchResult search(FloatArray queries, size_t k, size_t nprobe, Pruning pruning, dv::TieredQRouting routing) const {
    const auto shape = query_rows(queries, config_.dimensions);
    std::shared_lock lock(mutex_);
    const auto& index = trained();
    auto result = empty_result(shape, std::min(k, index.size()));
    index.search_batch(queries.data(), shape.rows, k, nprobe, result.scores.data(), result.ids.data(),
                       pruning.value_or(dv::TieredQPruning{}), routing);
    return result;
  }
  // Applies to the next list training and build; kept for the index trained later.
  void set_coarse_training(dv::TieredQCoarseTraining training) {
    if (require_lists("set_coarse_training") == 0) throw std::invalid_argument("set_coarse_training needs lists > 0");
    training.validate();
    std::unique_lock lock(mutex_);
    training_ = training;
    if (index_) index_->set_coarse_training(training);
  }
  void build_router(dv::TieredQRouterConfig config) {
    std::unique_lock lock(mutex_);
    if (!index_) throw std::runtime_error("build() the index before build_router()");
    index_->build_router(config);
  }
  void remove_router() {
    std::unique_lock lock(mutex_);
    router_ = {false, {}};
    if (index_) index_->remove_router();
  }
  // The training the next list training uses: the one set, else the library's automatic choice for the lists.
  dv::TieredQCoarseTraining coarse_training() const {
    std::shared_lock lock(mutex_);
    if (index_) return index_->coarse_training();
    if (require_lists("coarse_training") == 0) throw std::invalid_argument("coarse training needs lists > 0");
    return training_.value_or(dv::TieredQCoarseTraining::automatic(*lists_));
  }
  std::vector<uint32_t> probe_lists(FloatArray query, size_t nprobe, dv::TieredQRouting routing) const {
    const auto shape = query_rows(query, config_.dimensions);
    if (shape.rows != 1) throw std::invalid_argument("probe_lists takes one query");
    std::shared_lock lock(mutex_);
    return trained().probe_lists(query.data(), nprobe, routing);
  }
  bool has_router() const {
    std::shared_lock lock(mutex_);
    return index_ && index_->has_router();
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
  // The requested lists, which list-specific calls before the first build need to be explicit.
  uint32_t require_lists(const char* call) const {
    if (!lists_) throw std::invalid_argument(std::string(call) + " before build() needs an explicit lists=");
    return *lists_;
  }
  // The first training fixes automatic lists from its row count (build()'s rows, or train_lists()'s sample when the
  // lists are explicit).
  void ensure_codec(const RowInput& rows) {
    if (!index_) emplace(train_codec(config_, rows.loader, rows.count), rows.count);
  }
  void emplace(dv::TieredQ codec, size_t rows) {
    if (!lists_) lists_ = dv::TieredQIVFIndex::automatic_lists(rows);
    if (router_.use.value_or(false) && *lists_ == 0) throw std::invalid_argument("a router needs lists > 0");
    index_.emplace(std::move(codec), *lists_, training_seed);
    if (training_) index_->set_coarse_training(*training_);
    if (!given_lists_.empty()) index_->set_lists(given_lists_.data());
    if (router_.use) {
      if (*router_.use) index_->build_router(router_.config);
      else index_->remove_router();
    }
  }
  const dv::TieredQIVFIndex& trained() const {
    if (!index_) throw std::runtime_error("build() the index before using it");
    return *index_;
  }

  dv::TieredQConfig config_;
  std::optional<uint32_t> lists_;
  std::optional<dv::TieredQCoarseTraining> training_;
  RouterChoice router_;
  std::vector<float> given_lists_;
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
      .def("__init__", [](PythonIVFIndex* self, uint32_t dimensions, uint32_t code_bits, std::optional<uint32_t> lists,
                          const std::string& metric, bool fine_tiers, bool exact_fast_scores, uint32_t max_scan_bits,
                          std::optional<dv::TieredQCoarseTraining> coarse_training, nb::handle router) {
        new (self) PythonIVFIndex(dimensions, code_bits, lists, metric, fine_tiers, exact_fast_scores, max_scan_bits,
                                  coarse_training, parse_router(router));
      }, "dimensions"_a, "code_bits"_a, nb::kw_only(), "lists"_a = nb::none(), "metric"_a = "ip", "fine_tiers"_a = false,
           "exact_fast_scores"_a = false, "max_scan_bits"_a = 0, "coarse_training"_a = nb::none(),
           "router"_a = nb::none(),
           "lists None (default) picks the count from the first build's rows (2K at 1M rows, 64K at 8.84M; one\n"
           "list per codec centroid below about 143K rows); lists 0 makes one list per codec centroid at no extra\n"
           "storage; otherwise a coarse quantizer with that many lists is trained, by default as TieredQCoarseTraining.automatic(lists) (fast from 8,192 lists),\n"
           "and from TieredQRouterConfig.automatic_lists lists on with the two-level router: coarse_training and\n"
           "router (True, False or a TieredQRouterConfig) override them. metric is 'ip', 'cosine' or 'l2' (scores\n"
           "are -squared distances). fine_tiers\n"
           "lets steep spectra scan a short prefix of each code and rescore the best candidates of the probed\n"
           "lists exactly, at the same stored bytes (see TieredQIndex); off by default, as it gained at most 0.006\n"
           "recall@10 for 0-45% lower QPS on Gemini-768 99k (#32). exact_fast_scores trains Lloyd-Max codebooks\n"
           "scanned in exact FP32 (prunable on AVX-512); max_scan_bits caps the bits per coordinate the scan reads\n"
           "(0: 4 at up to 2 bits per dimension without fine tiers, for the nibble kernels; else as on TieredQIndex).\n"
           "train_lists, set_lists and set_coarse_training before the first build need an explicit lists.")
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
      .def("set_lists", &PythonIVFIndex::set_lists, "centroids"_a.noconvert(),
           "Fixes the coarse lists to given centroids (lists x dimensions float32; input space, normalized for cosine),\n"
           "e.g. a partition trained elsewhere; build() then only assigns rows.",
           nb::call_guard<nb::gil_scoped_release>())
      .def("build", [](PythonIVFIndex& self, OptionalMatrix rows, OptionalCodes codes, OptionalVector scales,
                       OptionalVector offsets, OptionalLoader loader, OptionalCount count) {
        const auto input = row_input(rows, codes, scales, offsets, loader, count, self.dimensions());
        nb::gil_scoped_release release;
        self.build(input);
      }, "rows"_a.noconvert() = nb::none(), nb::kw_only(), "codes"_a.noconvert() = nb::none(),
         "scales"_a.noconvert() = nb::none(), "offsets"_a.noconvert() = nb::none(), "loader"_a = nb::none(),
         "count"_a = nb::none(),
         "Replaces the content with rows (float32, rows x dimensions), an SQ8 store (codes= int8 rows x dimensions,\n"
         "e.g. a numpy memmap, with float32 scales= and offsets= per dimension: vector = codes * scales + offsets,\n"
         "decoded a chunk at a time) or loader=(ids -> float32 rows) with count=; all three build the same index.\n"
         "The first training (here or in train_lists) also trains the codec.")
      .def("search", [](const PythonIVFIndex& self, FloatArray queries, size_t k, size_t nprobe, Pruning pruning,
                        const std::string& routing, uint32_t super_probes) {
        const dv::TieredQRouting route{parse_routing(routing), super_probes};
        SearchResult result;
        { nb::gil_scoped_release release; result = self.search(queries, k, nprobe, pruning, route); }
        return result_tuple(std::move(result));
      }, "queries"_a.noconvert(), "k"_a = 10, nb::kw_only(), "nprobe"_a, "pruning"_a = nb::none(),
         "routing"_a = "auto", "super_probes"_a = 0,
         "Scans the nearest nprobe lists; slots past the probed rows hold id 2**32 - 1 and score -inf.\n"
         "routing 'auto' ranks lists through the router when one is built and exactly otherwise; 'router' and\n"
         "'exact' force either; super_probes > 0 overrides the router's probed supers.")
      .def("set_coarse_training", [](PythonIVFIndex& self, uint32_t iterations, uint32_t assignment_supers,
                                     uint32_t assignment_probes, uint32_t sample_rows_per_list) {
        self.set_coarse_training({iterations, assignment_supers, assignment_probes, sample_rows_per_list});
      }, nb::kw_only(), "iterations"_a = dv::TieredQCoarseTraining{}.iterations,
         "assignment_supers"_a = dv::TieredQCoarseTraining{}.assignment_supers,
         "assignment_probes"_a = dv::TieredQCoarseTraining{}.assignment_probes,
         "sample_rows_per_list"_a = dv::TieredQCoarseTraining{}.sample_rows_per_list,
         "Lloyd iterations of list training, restricted assignment (assignment_supers > 0: each row's nearest\n"
         "list among the members of its assignment_probes nearest supers) and build()'s list-training sample for\n"
         "later train_lists() and build(); overrides the automatic training (see coarse_training).",
         nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("coarse_training", &PythonIVFIndex::coarse_training,
                   "The list training later builds use: the one set, else TieredQCoarseTraining.automatic(lists).",
                   nb::call_guard<nb::gil_scoped_release>())
      .def("build_router", [](PythonIVFIndex& self, uint32_t lists_per_super, double bits_per_dimension, uint32_t rerank,
                              double expansion, uint32_t min_super_probes, uint64_t seed) {
        self.build_router({lists_per_super, bits_per_dimension, rerank, expansion, min_super_probes, seed});
      }, nb::kw_only(), "lists_per_super"_a = dv::TieredQRouterConfig{}.lists_per_super,
         "bits_per_dimension"_a = dv::TieredQRouterConfig{}.bits_per_dimension, "rerank"_a = dv::TieredQRouterConfig{}.rerank,
         "expansion"_a = dv::TieredQRouterConfig{}.expansion, "min_super_probes"_a = dv::TieredQRouterConfig{}.min_super_probes,
         "seed"_a = dv::TieredQRouterConfig{}.seed,
         "Builds the two-level compressed coarse router (coarse lists only) for searches over many\n"
         "lists; rebuilt by every later build() and saved with the index.",
         nb::call_guard<nb::gil_scoped_release>())
      .def("remove_router", &PythonIVFIndex::remove_router,
           "Drops the router (also the automatic one): searches rank every list exactly, later builds add none.",
           nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("has_router", &PythonIVFIndex::has_router, nb::call_guard<nb::gil_scoped_release>())
      .def("probe_lists", [](const PythonIVFIndex& self, FloatArray query, size_t nprobe, const std::string& routing,
                             uint32_t super_probes) {
        const dv::TieredQRouting route{parse_routing(routing), super_probes};
        std::vector<uint32_t> lists;
        { nb::gil_scoped_release release; lists = self.probe_lists(query, nprobe, route); }
        const size_t count = lists.size();
        return owned_array(std::move(lists), {count});
      }, "query"_a.noconvert(), nb::kw_only(), "nprobe"_a, "routing"_a = "auto", "super_probes"_a = 0,
         "The nprobe lists a search of query scans, nearest first (empty lists skipped).")
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
