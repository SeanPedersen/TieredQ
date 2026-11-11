// Nested IVF serving (#23, #25): build() trains the prefix codec, the lists and the refinement from one seeded sample
// (of the vectors, or of an explicit sample=), so they cannot be mismatched. The tier rows are attached separately
// from RAM or a row file; candidates() and rescore() split a search for tiers kept in an external store. Searches may
// run concurrently with attach_tier(); the GIL is released for every native call.
#include "arrays.h"
#include "training.h"
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <tieredq/tieredq_nested.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>

namespace tieredq_python {
using namespace nb::literals;
namespace {
using Pruning = std::optional<dv::TieredQPruning>;
using OptionalMatrix = std::optional<Matrix>;
using IdArray = nb::ndarray<nb::numpy, const uint32_t, nb::c_contig, nb::device::cpu>;
using ByteArray = nb::ndarray<nb::numpy, const uint8_t, nb::c_contig, nb::device::cpu>;
using ByteMatrix = nb::ndarray<nb::numpy, const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using CodeMatrix = nb::ndarray<nb::numpy, const int8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using FloatVector = nb::ndarray<nb::numpy, const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
// Tier rows encoded per write_tier() step, bounding its buffer.
constexpr size_t tier_write_chunk_rows = 16384;

dv::TieredQTierRows parse_tier_rows(const std::string& layout) {
  if (layout=="prefix_and_suffix") return dv::TieredQTierRows::prefix_and_suffix;
  if (layout=="suffix") return dv::TieredQTierRows::suffix;
  throw std::invalid_argument("tier_rows must be 'prefix_and_suffix' or 'suffix'");
}
const char* tier_rows_name(dv::TieredQTierRows layout) {
  return layout==dv::TieredQTierRows::suffix ? "suffix" : "prefix_and_suffix";
}
dv::TieredQFileReads parse_reads(const std::string& reads) {
  if (reads=="cached") return dv::TieredQFileReads::cached;
  if (reads=="direct") return dv::TieredQFileReads::direct;
  throw std::invalid_argument("reads must be 'cached' or 'direct'");
}
// Trains the prefix, the lists (on sample, when given) and the refinement on one seeded sample, then builds the lists.
dv::TieredQNestedIVFIndex train_nested(const dv::TieredQConfig& prefix, uint32_t total_bits, uint32_t lists,
                                       dv::TieredQTierRows tier_rows, Matrix vectors, OptionalMatrix sample, bool fine_tiers, uint32_t max_tier_bits) {
  const size_t width = prefix.dimensions;
  const float* training = sample ? sample->data() : vectors.data();
  const size_t training_rows = sample ? sample->shape(0) : vectors.shape(0);
  if (training_rows < minimum_training_rows) throw std::runtime_error("training needs at least 2 rows");
  const auto rows = training_sample(training, training_rows, width);
  const size_t sampled = rows.size() / width;
  auto config = prefix;
  config.num_centroids = auto_centroids(sampled);
  auto codec = dv::TieredQ::train(config, rows.data(), sampled, training_seed);
  auto refinement = dv::TieredQRefinement::train(codec, total_bits, rows.data(), sampled, training_seed,
                                               fine_tiers, max_tier_bits);
  dv::TieredQIVFIndex index(std::move(codec), lists, training_seed);
  if (sample) index.train_lists(sample->data(), sample->shape(0));
  index.build(vectors.data(), vectors.shape(0));
  return dv::TieredQNestedIVFIndex(std::move(index), std::move(refinement), tier_rows);
}
}  // namespace

class PythonNestedIndex {
 public:
  explicit PythonNestedIndex(dv::TieredQNestedIVFIndex index) : index_(std::move(index)) {}
  size_t size() const { return index_.size(); }
  const dv::TieredQ& prefix() const { return index_.index().codec(); }
  uint32_t dimensions() const { return prefix().config().dimensions; }
  const char* metric() const { return metric_name(prefix().config().metric); }
  uint32_t prefix_bits() const { return prefix().config().code_bits; }
  size_t total_bits() const { return prefix_bits() + index_.refinement().suffix_bits(); }
  uint32_t max_refinement_bits() const { return index_.refinement().max_code_bits(); }
  size_t lists() const { return index_.index().lists(); }
  size_t storage_bytes() const { return index_.index().storage_bytes(); }
  const char* tier_rows() const { return tier_rows_name(index_.tier_rows()); }
  size_t tier_bytes_per_vector() const { return index_.tier_bytes_per_vector(); }
  bool has_tier() const {
    std::shared_lock lock(mutex_);
    return index_.has_tier();
  }
  std::vector<uint8_t> encode_tier(Matrix rows, size_t first) const {
    require_width(rows.shape(1), dimensions());
    std::vector<uint8_t> tier(array_elements(rows.shape(0), tier_bytes_per_vector()));
    index_.encode_tier(rows.data(), first, rows.shape(0), tier.data());
    return tier;
  }
  void write_tier(const std::filesystem::path& path, Matrix rows) const {
    require_width(rows.shape(1), dimensions());
    if (rows.shape(0) != size()) throw std::invalid_argument("write_tier needs one row per indexed vector");
    const size_t stride = tier_bytes_per_vector();
    dv::TieredQRowFileWriter writer(path.string(), stride);
    std::vector<uint8_t> chunk(array_elements(std::min(size(), tier_write_chunk_rows), stride));
    for (size_t first = 0; first < size(); first += tier_write_chunk_rows) {
      const size_t count = std::min(tier_write_chunk_rows, size() - first);
      index_.encode_tier(rows.data() + first * dimensions(), first, count, chunk.data());
      writer.append(chunk.data(), count);
    }
    writer.finish();
  }
  void attach_tier(std::shared_ptr<const dv::TieredQRowSource> tier) {
    std::unique_lock lock(mutex_);
    index_.attach_tier(std::move(tier));
  }
  void attach_exact(std::shared_ptr<const dv::TieredQRowSource> codes, FloatVector scales, FloatVector offsets,
                    size_t finals) {
    auto exact = std::make_shared<dv::TieredQSQ8Rows>(std::move(codes),
        std::vector<float>(scales.data(), scales.data() + scales.shape(0)),
        std::vector<float>(offsets.data(), offsets.data() + offsets.shape(0)));
    std::unique_lock lock(mutex_);
    index_.attach_exact(std::move(exact), finals);
  }
  void detach_exact() {
    std::unique_lock lock(mutex_);
    index_.attach_exact(nullptr, 0);
  }
  bool has_exact() const {
    std::shared_lock lock(mutex_);
    return index_.has_exact();
  }
  size_t finals() const {
    std::shared_lock lock(mutex_);
    return index_.finals();
  }
  SearchResult search(FloatArray queries, size_t k, size_t nprobe, size_t candidates, Pruning pruning) const {
    const auto shape = query_rows(queries, dimensions());
    auto result = empty_result(shape, std::min(k, size()));
    std::shared_lock lock(mutex_);
    index_.search_batch(queries.data(), shape.rows, k, nprobe, candidates, result.scores.data(), result.ids.data(),
                        pruning.value_or(dv::TieredQPruning{}));
    return result;
  }
  SearchResult candidates(FloatArray queries, size_t k, size_t nprobe, Pruning pruning) const {
    const auto shape = query_rows(queries, dimensions());
    auto result = empty_result(shape, std::min(k, size()));
    index_.index().search_batch(queries.data(), shape.rows, k, nprobe, result.scores.data(), result.ids.data(),
                                pruning.value_or(dv::TieredQPruning{}));
    return result;
  }
  SearchResult rescore(FloatArray queries, IdArray ids, ByteArray tier_rows, size_t k) const {
    const auto shape = query_rows(queries, dimensions());
    const size_t depth = check_candidates(shape, ids, tier_rows);
    auto result = empty_result(shape, std::min(k, size()));
    index_.rescore_batch(queries.data(), shape.rows, ids.data(), depth, tier_rows.data(), k, result.scores.data(),
                         result.ids.data());
    return result;
  }
  std::vector<uint8_t> serialize() const { return index_.serialize(); }
  void save(const std::filesystem::path& path) const {
    const auto bytes = serialize();
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    if (!out) throw std::runtime_error("cannot write nested index to " + path.string());
  }
  static std::unique_ptr<PythonNestedIndex> load(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read nested index from " + path.string());
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    return std::make_unique<PythonNestedIndex>(dv::TieredQNestedIVFIndex::deserialize(bytes));
  }

 private:
  // Candidate ids are queries x depth (depth for a 1-D query); their tier rows add a trailing row-bytes axis.
  size_t check_candidates(QueryRows queries, const IdArray& ids, const ByteArray& tier_rows) const {
    const size_t id_rank = queries.single ? 1 : 2;
    if (ids.ndim() != id_rank || tier_rows.ndim() != id_rank + 1)
      throw std::invalid_argument("ids and tier_rows must add one and two axes to each query");
    if (!queries.single && (ids.shape(0) != queries.rows || tier_rows.shape(0) != queries.rows))
      throw std::invalid_argument("ids and tier_rows need one row per query");
    const size_t depth = ids.shape(id_rank - 1);
    if (tier_rows.shape(id_rank - 1) != depth) throw std::invalid_argument("tier_rows need one row per candidate id");
    if (tier_rows.shape(id_rank) != tier_bytes_per_vector())
      throw std::invalid_argument("tier row width does not match the index");
    return depth;
  }
  dv::TieredQNestedIVFIndex index_;
  mutable std::shared_mutex mutex_;  // guards the attached tier
};

namespace {
nb::ndarray<nb::numpy, uint8_t> tier_array(std::vector<uint8_t> tier, size_t stride) {
  const size_t rows = tier.size() / stride;
  return owned_array(std::move(tier), {rows, stride});
}
template<class Search>
nb::tuple released(Search search) {
  SearchResult result;
  { nb::gil_scoped_release release; result = search(); }
  return result_tuple(std::move(result));
}
}  // namespace

void bind_nested(nb::module_& module) {
  nb::class_<PythonNestedIndex>(module, "TieredQNestedIVFIndex",
      "In-RAM IVF over prefix codes whose candidates are rescored with refinement tier rows from RAM or SSD.")
      .def_static("build", [](Matrix vectors, uint32_t prefix_bits, uint32_t total_bits, uint32_t lists,
                              const std::string& metric, const std::string& tier_rows, OptionalMatrix sample, bool fine_tiers, uint32_t max_tier_bits) {
        auto config = dv::TieredQConfig::target_code_bits(uint32_t(vectors.shape(1)), prefix_bits);
        config.metric = parse_metric(metric);
        config.fine_tiers = false;  // The IVF prefix stays independently scannable.
        config.validate();
        if (sample) require_width(sample->shape(1), config.dimensions);
        const auto layout = parse_tier_rows(tier_rows);
        nb::gil_scoped_release release;
        return std::make_unique<PythonNestedIndex>(train_nested(config, total_bits, lists, layout, vectors, sample, fine_tiers, max_tier_bits));
      }, "vectors"_a.noconvert(), nb::kw_only(), "prefix_bits"_a, "total_bits"_a, "lists"_a = 0, "metric"_a = "ip",
         "tier_rows"_a = "prefix_and_suffix", "sample"_a.noconvert() = nb::none(),
         "fine_tiers"_a = false, "max_tier_bits"_a = 12,
         "Trains the prefix codec (prefix_bits), the refinement (total_bits, prefix included) and the lists on a\n"
         "seeded sample of at most 32,768 rows of sample, else of vectors, then indexes vectors. A given sample\n"
         "also trains the lists on all its rows, so a partition learned on earlier data stays fixed. tier_rows\n"
         "'prefix_and_suffix' reads each candidate in one row; 'suffix' stores a smaller tier and decodes the\n"
         "prefix from the lists. fine_tiers permits wider suffix refinement (up to max_tier_bits, at most 12)\n"
         "for inner product/cosine. No tier is attached yet.")
      .def("__len__", &PythonNestedIndex::size)
      .def_prop_ro("dimensions", &PythonNestedIndex::dimensions)
      .def_prop_ro("metric", &PythonNestedIndex::metric)
      .def_prop_ro("prefix_bits", &PythonNestedIndex::prefix_bits, "Code bits the IVF scans.")
      .def_prop_ro("total_bits", &PythonNestedIndex::total_bits, "Prefix plus suffix code bits candidates rescore with.")
      .def_prop_ro("max_refinement_bits", &PythonNestedIndex::max_refinement_bits, "Widest combined prefix/suffix code.")
      .def_prop_ro("lists", &PythonNestedIndex::lists)
      .def_prop_ro("storage_bytes", &PythonNestedIndex::storage_bytes, "In-RAM IVF bytes, excluding the tier.")
      .def_prop_ro("tier_rows", &PythonNestedIndex::tier_rows)
      .def_prop_ro("tier_bytes_per_vector", &PythonNestedIndex::tier_bytes_per_vector)
      .def_prop_ro("has_tier", &PythonNestedIndex::has_tier, nb::call_guard<nb::gil_scoped_release>())
      .def("encode_tier", [](const PythonNestedIndex& self, Matrix rows, size_t first) {
        std::vector<uint8_t> tier;
        { nb::gil_scoped_release release; tier = self.encode_tier(rows, first); }
        return tier_array(std::move(tier), self.tier_bytes_per_vector());
      }, "rows"_a.noconvert(), "first"_a = 0,
         "Tier rows (uint8, rows x tier_bytes_per_vector) of the vectors with ids first.., as given to build();\n"
         "encode a large set in chunks.")
      .def("write_tier", &PythonNestedIndex::write_tier, "path"_a, "rows"_a.noconvert(),
           "Encodes every vector given to build() into a row file for attach_tier(path).",
           nb::call_guard<nb::gil_scoped_release>())
      .def("attach_tier", [](PythonNestedIndex& self, ByteMatrix rows) {
        nb::gil_scoped_release release;
        std::vector<uint8_t> data(rows.data(), rows.data() + array_elements(rows.shape(0), rows.shape(1)));
        self.attach_tier(std::make_shared<dv::TieredQMemoryRows>(std::move(data), rows.shape(1)));
      }, "rows"_a.noconvert(), "Serves the tier from RAM: a copy of encode_tier() rows for every vector, in id order.")
      .def("attach_tier", [](PythonNestedIndex& self, const std::filesystem::path& path, const std::string& reads,
                             size_t queue_depth) {
        const auto mode = parse_reads(reads);
        nb::gil_scoped_release release;
        self.attach_tier(std::make_shared<dv::TieredQFileRows>(path.string(), mode, queue_depth));
      }, "path"_a, nb::kw_only(), "reads"_a = "cached", "queue_depth"_a = 1,
         "Serves the tier from a write_tier() file. reads 'direct' bypasses the page cache; queue_depth > 1 keeps\n"
         "that many reads in flight per query (io_uring builds), which SSDs need for throughput.")
      .def("attach_exact", [](PythonNestedIndex& self, CodeMatrix codes, FloatVector scales, FloatVector offsets,
                              size_t finals) {
        nb::gil_scoped_release release;
        const auto* begin = reinterpret_cast<const uint8_t*>(codes.data());
        std::vector<uint8_t> data(begin, begin + array_elements(codes.shape(0), codes.shape(1)));
        self.attach_exact(std::make_shared<dv::TieredQMemoryRows>(std::move(data), codes.shape(1)), scales, offsets, finals);
      }, "codes"_a.noconvert(), "scales"_a.noconvert(), "offsets"_a.noconvert(), nb::kw_only(), "finals"_a,
         "Final exact stage from an SQ8 store in RAM (int8 codes, rows x dimensions, in id order; vector =\n"
         "codes * scales + offsets): each search rescores the tier's best max(k, finals) candidates exactly.")
      .def("attach_exact", [](PythonNestedIndex& self, const std::filesystem::path& path, FloatVector scales,
                              FloatVector offsets, size_t finals, const std::string& reads, size_t queue_depth) {
        const auto mode = parse_reads(reads);
        nb::gil_scoped_release release;
        self.attach_exact(std::make_shared<dv::TieredQFileRows>(path.string(), mode, queue_depth), scales, offsets, finals);
      }, "path"_a, "scales"_a.noconvert(), "offsets"_a.noconvert(), nb::kw_only(), "finals"_a, "reads"_a = "cached",
         "queue_depth"_a = 1,
         "Final exact stage from a write_sq8_rows() file, e.g. on SSD with reads='direct' and queue_depth=32, so\n"
         "only the IVF and the tier stay in RAM.")
      .def("detach_exact", &PythonNestedIndex::detach_exact, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("has_exact", &PythonNestedIndex::has_exact, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("finals", &PythonNestedIndex::finals, nb::call_guard<nb::gil_scoped_release>())
      .def_static("write_sq8_rows", [](const std::filesystem::path& path, CodeMatrix codes) {
        nb::gil_scoped_release release;
        dv::TieredQRowFileWriter writer(path.string(), codes.shape(1));
        const auto* rows = reinterpret_cast<const uint8_t*>(codes.data());
        for (size_t first = 0; first < codes.shape(0); first += tier_write_chunk_rows) {
          const size_t count = std::min(tier_write_chunk_rows, size_t(codes.shape(0)) - first);
          writer.append(rows + first * codes.shape(1), count);
        }
        writer.finish();
      }, "path"_a, "codes"_a.noconvert(), "Writes int8 SQ8 codes (rows x dimensions, in id order) as a row file for attach_exact(path).")
      .def("search", [](const PythonNestedIndex& self, FloatArray queries, size_t k, size_t nprobe, size_t candidates,
                        Pruning pruning) {
        return released([&] { return self.search(queries, k, nprobe, candidates, pruning); });
      }, "queries"_a.noconvert(), "k"_a = 10, nb::kw_only(), "nprobe"_a, "candidates"_a, "pruning"_a = nb::none(),
         "The IVF's top candidates (>= k) over nprobe lists, rescored with their tier rows; slots past the probed\n"
         "rows hold id 2**32 - 1 and score -inf.")
      .def("candidates", [](const PythonNestedIndex& self, FloatArray queries, size_t k, size_t nprobe,
                            Pruning pruning) {
        SearchResult result;
        { nb::gil_scoped_release release; result = self.candidates(queries, k, nprobe, pruning); }
        return owned_array(std::move(result.ids), result.shape);
      }, "queries"_a.noconvert(), "k"_a, nb::kw_only(), "nprobe"_a, "pruning"_a = nb::none(),
         "First step of a search over an external tier: the ids of the IVF's top k by prefix score (padded with\n"
         "2**32 - 1), whose tier rows rescore() takes.")
      .def("rescore", [](const PythonNestedIndex& self, FloatArray queries, IdArray ids, ByteArray tier_rows,
                         size_t k) {
        return released([&] { return self.rescore(queries, ids, tier_rows, k); });
      }, "queries"_a.noconvert(), "ids"_a.noconvert(), "tier_rows"_a.noconvert(), "k"_a = 10,
         "Second step: rescores candidates() ids with their tier rows (one encode_tier() row per id; rows of\n"
         "padded ids are ignored). Needs no attached tier.")
      .def("serialize", [](const PythonNestedIndex& self) {
        std::vector<uint8_t> bytes;
        { nb::gil_scoped_release release; bytes = self.serialize(); }
        return nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      }, "IVF, refinement and tier layout; the tier is attached separately after deserializing.")
      .def_static("deserialize", [](nb::bytes data) {
        const auto* begin = reinterpret_cast<const uint8_t*>(data.c_str());
        const std::vector<uint8_t> bytes(begin, begin + data.size());
        nb::gil_scoped_release release;
        return std::make_unique<PythonNestedIndex>(dv::TieredQNestedIVFIndex::deserialize(bytes));
      }, "data"_a)
      .def("save", &PythonNestedIndex::save, "path"_a, nb::call_guard<nb::gil_scoped_release>())
      .def_static("load", &PythonNestedIndex::load, "path"_a, nb::call_guard<nb::gil_scoped_release>());
}
}  // namespace tieredq_python
