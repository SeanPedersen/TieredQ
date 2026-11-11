// Nested IVF serving (#23, #25, #52): build() trains the prefix codec, the lists and the refinement from one seeded
// sample (of the vectors, or of an explicit sample=), so they cannot be mismatched; its defaults are the library's
// low-RAM configuration (TieredQNestedConfig). Vectors come as float rows, an SQ8 store or a row callable, for build()
// and write_tier() alike. The tier rows are attached separately from RAM or a row file; candidates() and rescore()
// split a search for tiers kept in an external store. Searches may run concurrently with attach_tier(); the GIL is
// released for every native call.
#include "arrays.h"
#include "lists.h"
#include "rows.h"
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
using Pruning = std::optional<tq::TieredQPruning>;
using OptionalBits = std::optional<uint32_t>;
using IdArray = nb::ndarray<nb::numpy, const uint32_t, nb::c_contig, nb::device::cpu>;
using ByteArray = nb::ndarray<nb::numpy, const uint8_t, nb::c_contig, nb::device::cpu>;
using ByteMatrix = nb::ndarray<nb::numpy, const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
// SQ8 rows copied per write_sq8_rows() step, bounding its buffer.
constexpr size_t sq8_write_chunk_rows = 16384;
// Reads in flight for direct-I/O tiers unless given: the NVMe tier's measured setting (#49).
constexpr size_t direct_queue_depth = 32;

tq::TieredQTierRows parse_tier_rows(const std::string& layout) {
  if (layout=="prefix_and_suffix") return tq::TieredQTierRows::prefix_and_suffix;
  if (layout=="suffix") return tq::TieredQTierRows::suffix;
  throw std::invalid_argument("tier_rows must be 'prefix_and_suffix' or 'suffix'");
}
const char* tier_rows_name(tq::TieredQTierRows layout) {
  return layout==tq::TieredQTierRows::suffix ? "suffix" : "prefix_and_suffix";
}
tq::TieredQFileReads parse_reads(const std::string& reads) {
  if (reads=="cached") return tq::TieredQFileReads::cached;
  if (reads=="direct") return tq::TieredQFileReads::direct;
  throw std::invalid_argument("reads must be 'cached' or 'direct'");
}
}  // namespace

class PythonNestedIndex {
 public:
  explicit PythonNestedIndex(tq::TieredQNestedIVFIndex index) : index_(std::move(index)) {}
  size_t size() const { return index_.size(); }
  const tq::TieredQ& prefix() const { return index_.index().codec(); }
  uint32_t dimensions() const { return prefix().config().dimensions; }
  const char* metric() const { return metric_name(prefix().config().metric); }
  uint32_t prefix_bits() const { return prefix().config().code_bits; }
  size_t total_bits() const { return prefix_bits() + index_.refinement().suffix_bits(); }
  uint32_t max_refinement_bits() const { return index_.refinement().max_code_bits(); }
  size_t lists() const { return index_.index().lists(); }
  size_t storage_bytes() const { return index_.storage_bytes(); }
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
  void write_tier(const std::filesystem::path& path, const RowInput& rows) const {
    if (rows.count != size()) throw std::invalid_argument("write_tier needs one row per indexed vector");
    index_.write_tier(path.string(), rows.loader);
  }
  bool has_router() const { return index_.index().has_router(); }
  void attach_tier(std::shared_ptr<const tq::TieredQRowSource> tier) {
    std::unique_lock lock(mutex_);
    index_.attach_tier(std::move(tier));
  }
  void attach_exact(std::shared_ptr<const tq::TieredQRowSource> codes, FloatVector scales, FloatVector offsets,
                    size_t finals) {
    auto exact = std::make_shared<tq::TieredQSQ8Rows>(std::move(codes),
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
  SearchResult search(FloatArray queries, size_t k, size_t nprobe, std::optional<size_t> candidates,
                      Pruning pruning) const {
    const auto shape = query_rows(queries, dimensions());
    auto result = empty_result(shape, std::min(k, size()));
    std::shared_lock lock(mutex_);
    index_.search_batch(queries.data(), shape.rows, k, nprobe,
                        candidates.value_or(tq::TieredQNestedIVFIndex::candidates_for(k)), result.scores.data(),
                        result.ids.data(), pruning.value_or(tq::TieredQPruning{}));
    return result;
  }
  SearchResult candidates(FloatArray queries, size_t k, size_t nprobe, Pruning pruning) const {
    const auto shape = query_rows(queries, dimensions());
    auto result = empty_result(shape, std::min(k, size()));
    index_.index().search_batch(queries.data(), shape.rows, k, nprobe, result.scores.data(), result.ids.data(),
                                pruning.value_or(tq::TieredQPruning{}));
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
    return std::make_unique<PythonNestedIndex>(tq::TieredQNestedIVFIndex::deserialize(bytes));
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
  tq::TieredQNestedIVFIndex index_;
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
      .def_static("build", [](OptionalMatrix vectors, OptionalCodes codes, OptionalVector scales, OptionalVector offsets,
                              OptionalLoader loader, OptionalCount count, OptionalCount dimensions,
                              OptionalBits prefix_bits, OptionalBits total_bits, OptionalBits lists,
                              const std::string& metric, const std::string& tier_rows, OptionalMatrix sample,
                              bool fine_tiers, uint32_t max_tier_bits, uint32_t max_scan_bits,
                              std::optional<tq::TieredQCoarseTraining> coarse_training, nb::handle router) {
        const size_t width = vectors ? vectors->shape(1) : codes ? codes->shape(1) : dimensions.value_or(0);
        if (!width) throw std::invalid_argument("loader= needs dimensions=");
        const auto input = row_input(vectors, codes, scales, offsets, loader, count, width);
        if (sample) require_width(sample->shape(1), width);
        tq::TieredQNestedConfig config;
        config.dimensions = uint32_t(width);
        config.prefix_bits = prefix_bits.value_or(0);
        config.total_bits = total_bits.value_or(0);
        config.lists = lists.value_or(tq::TieredQNestedConfig::automatic_lists);
        config.metric = parse_metric(metric);
        config.tier_rows = parse_tier_rows(tier_rows);
        config.fine_tiers = fine_tiers;
        config.max_tier_bits = max_tier_bits;
        config.max_scan_bits = max_scan_bits;
        config.coarse_training = coarse_training;
        const auto choice = parse_router(router);
        config.router = choice.use;
        config.router_config = choice.config;
        config.seed = training_seed;
        nb::gil_scoped_release release;
        return std::make_unique<PythonNestedIndex>(tq::TieredQNestedIVFIndex::build(
            config, input.loader, input.count, sample ? sample->data() : nullptr, sample ? sample->shape(0) : 0));
      }, "vectors"_a.noconvert() = nb::none(), nb::kw_only(), "codes"_a.noconvert() = nb::none(),
         "scales"_a.noconvert() = nb::none(), "offsets"_a.noconvert() = nb::none(), "loader"_a = nb::none(),
         "count"_a = nb::none(), "dimensions"_a = nb::none(), "prefix_bits"_a = nb::none(),
         "total_bits"_a = nb::none(), "lists"_a = nb::none(), "metric"_a = "ip", "tier_rows"_a = "suffix",
         "sample"_a.noconvert() = nb::none(), "fine_tiers"_a = true, "max_tier_bits"_a = 12, "max_scan_bits"_a = 0,
         "coarse_training"_a = nb::none(), "router"_a = nb::none(),
         "Trains the prefix codec (prefix_bits per vector; default 1 per dimension), the refinement (total_bits,\n"
         "prefix included; default 8 per dimension) and the lists on a seeded sample of at most 32,768 rows of\n"
         "sample, else of the vectors, then indexes the vectors. Vectors are float32 rows, an SQ8 store (codes=\n"
         "int8 rows x dimensions, e.g. a numpy memmap, with float32 scales= and offsets=: vector = codes * scales\n"
         "+ offsets, decoded a chunk at a time) or loader=(ids -> float32 rows) with count= and dimensions=.\n"
         "The defaults are the low-RAM configuration: lists None picks 2K at 1M rows, 64K at 8.84M (see automatic_lists; 0 keeps one\n"
         "list per codec centroid), many lists train fast and get the router (coarse_training and router: True,\n"
         "False or a TieredQRouterConfig override them), and the 8-bit fine-tier suffix ('suffix' tier_rows) is\n"
         "meant for an SSD. max_scan_bits 0 caps prefix scans at 4 bits up to 2 bits per dimension. A given sample\n"
         "also trains the lists on all its rows, so a partition learned on earlier data stays fixed. tier_rows\n"
         "'prefix_and_suffix' reads each candidate in one row. No tier is attached yet.")
      .def("__len__", &PythonNestedIndex::size)
      .def_prop_ro("dimensions", &PythonNestedIndex::dimensions)
      .def_prop_ro("metric", &PythonNestedIndex::metric)
      .def_prop_ro("prefix_bits", &PythonNestedIndex::prefix_bits, "Code bits the IVF scans.")
      .def_prop_ro("total_bits", &PythonNestedIndex::total_bits, "Prefix plus suffix code bits candidates rescore with.")
      .def_prop_ro("max_refinement_bits", &PythonNestedIndex::max_refinement_bits, "Widest combined prefix/suffix code.")
      .def_prop_ro("lists", &PythonNestedIndex::lists)
      .def_prop_ro("has_router", &PythonNestedIndex::has_router)
      .def_prop_ro("storage_bytes", &PythonNestedIndex::storage_bytes, "RAM the index holds: IVF, codec, refinement and id map; the tier\n"
                   "is excluded.")
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
      .def("write_tier", [](const PythonNestedIndex& self, const std::filesystem::path& path, OptionalMatrix rows,
                            OptionalCodes codes, OptionalVector scales, OptionalVector offsets, OptionalLoader loader) {
        const auto input = row_input(rows, codes, scales, offsets, loader,
                                     loader ? OptionalCount(self.size()) : std::nullopt, self.dimensions());
        nb::gil_scoped_release release;
        self.write_tier(path, input);
      }, "path"_a, "rows"_a.noconvert() = nb::none(), nb::kw_only(), "codes"_a.noconvert() = nb::none(),
         "scales"_a.noconvert() = nb::none(), "offsets"_a.noconvert() = nb::none(), "loader"_a = nb::none(),
         "Encodes every vector given to build() (the same rows, SQ8 store or loader) into a row file for\n"
         "attach_tier(path), a chunk at a time.")
      .def("attach_tier", [](PythonNestedIndex& self, ByteMatrix rows) {
        nb::gil_scoped_release release;
        std::vector<uint8_t> data(rows.data(), rows.data() + array_elements(rows.shape(0), rows.shape(1)));
        self.attach_tier(std::make_shared<tq::TieredQMemoryRows>(std::move(data), rows.shape(1)));
      }, "rows"_a.noconvert(), "Serves the tier from RAM: a copy of encode_tier() rows for every vector, in id order.")
      .def("attach_tier", [](PythonNestedIndex& self, const std::filesystem::path& path, const std::string& reads,
                             OptionalCount queue_depth) {
        const auto mode = parse_reads(reads);
        const size_t depth = queue_depth.value_or(mode == tq::TieredQFileReads::direct ? direct_queue_depth : 1);
        nb::gil_scoped_release release;
        self.attach_tier(std::make_shared<tq::TieredQFileRows>(path.string(), mode, depth));
      }, "path"_a, nb::kw_only(), "reads"_a = "cached", "queue_depth"_a = nb::none(),
         "Serves the tier from a write_tier() file. reads 'direct' bypasses the page cache; queue_depth > 1 keeps\n"
         "that many reads in flight per query (io_uring builds), which SSDs need for throughput (default: 32 for\n"
         "direct reads, 1 for cached ones).")
      .def("attach_exact", [](PythonNestedIndex& self, CodeMatrix codes, FloatVector scales, FloatVector offsets,
                              size_t finals) {
        nb::gil_scoped_release release;
        const auto* begin = reinterpret_cast<const uint8_t*>(codes.data());
        std::vector<uint8_t> data(begin, begin + array_elements(codes.shape(0), codes.shape(1)));
        self.attach_exact(std::make_shared<tq::TieredQMemoryRows>(std::move(data), codes.shape(1)), scales, offsets, finals);
      }, "codes"_a.noconvert(), "scales"_a.noconvert(), "offsets"_a.noconvert(), nb::kw_only(), "finals"_a,
         "Final exact stage from an SQ8 store in RAM (int8 codes, rows x dimensions, in id order; vector =\n"
         "codes * scales + offsets): each search rescores the tier's best max(k, finals) candidates exactly.")
      .def("attach_exact", [](PythonNestedIndex& self, const std::filesystem::path& path, FloatVector scales,
                              FloatVector offsets, size_t finals, const std::string& reads, size_t queue_depth) {
        const auto mode = parse_reads(reads);
        nb::gil_scoped_release release;
        self.attach_exact(std::make_shared<tq::TieredQFileRows>(path.string(), mode, queue_depth), scales, offsets, finals);
      }, "path"_a, "scales"_a.noconvert(), "offsets"_a.noconvert(), nb::kw_only(), "finals"_a, "reads"_a = "cached",
         "queue_depth"_a = 1,
         "Final exact stage from a write_sq8_rows() file, e.g. on SSD with reads='direct' and queue_depth=32, so\n"
         "only the IVF and the tier stay in RAM.")
      .def("detach_exact", &PythonNestedIndex::detach_exact, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("has_exact", &PythonNestedIndex::has_exact, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("finals", &PythonNestedIndex::finals, nb::call_guard<nb::gil_scoped_release>())
      .def_static("write_sq8_rows", [](const std::filesystem::path& path, CodeMatrix codes) {
        nb::gil_scoped_release release;
        tq::TieredQRowFileWriter writer(path.string(), codes.shape(1));
        const auto* rows = reinterpret_cast<const uint8_t*>(codes.data());
        for (size_t first = 0; first < codes.shape(0); first += sq8_write_chunk_rows) {
          const size_t count = std::min(sq8_write_chunk_rows, size_t(codes.shape(0)) - first);
          writer.append(rows + first * codes.shape(1), count);
        }
        writer.finish();
      }, "path"_a, "codes"_a.noconvert(), "Writes int8 SQ8 codes (rows x dimensions, in id order) as a row file for attach_exact(path).")
      .def("search", [](const PythonNestedIndex& self, FloatArray queries, size_t k, size_t nprobe,
                        std::optional<size_t> candidates, Pruning pruning) {
        return released([&] { return self.search(queries, k, nprobe, candidates, pruning); });
      }, "queries"_a.noconvert(), "k"_a = 10, nb::kw_only(), "nprobe"_a, "candidates"_a = nb::none(),
         "pruning"_a = nb::none(),
         "The IVF's top candidates (>= k; default 40, or 2k beyond k = 20) over nprobe lists, rescored with their\n"
         "tier rows; slots past the probed rows hold id 2**32 - 1 and score -inf.")
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
        return std::make_unique<PythonNestedIndex>(tq::TieredQNestedIVFIndex::deserialize(bytes));
      }, "data"_a)
      .def("save", &PythonNestedIndex::save, "path"_a, nb::call_guard<nb::gil_scoped_release>())
      .def_static("load", &PythonNestedIndex::load, "path"_a, nb::call_guard<nb::gil_scoped_release>());
}
}  // namespace tieredq_python
