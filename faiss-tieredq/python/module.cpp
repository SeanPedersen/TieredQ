// Python module of faiss-tieredq: IndexIVFTieredQ with its own coarse quantizer (IndexFlat or IndexHNSWFlat), since an
// out-of-tree index cannot take objects of Faiss's SWIG module. NumPy in and out; searches release the GIL.
#include "faiss_tieredq/IndexIVFTieredQ.h"
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/IDSelector.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using faiss::idx_t;
using Matrix = nb::ndarray<nb::numpy, const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Ids = nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

namespace {
template<class T>
nb::ndarray<nb::numpy, T> owned(std::vector<T> data, size_t rows, size_t columns) {
  auto* storage = new std::vector<T>(std::move(data));
  nb::capsule owner(storage, [](void* pointer) noexcept { delete static_cast<std::vector<T>*>(pointer); });
  const size_t shape[2] = {rows, columns};
  return nb::ndarray<nb::numpy, T>(storage->data(), 2, shape, owner);
}

faiss::MetricType parse_metric(const std::string& metric) {
  if (metric == "ip" || metric == "cosine") return faiss::METRIC_INNER_PRODUCT;
  if (metric == "l2") return faiss::METRIC_L2;
  throw std::invalid_argument("metric must be 'ip', 'cosine' or 'l2'");
}

class PythonIndex {
 public:
  PythonIndex(size_t dimensions, size_t lists, uint32_t code_bits, const std::string& metric, const std::string& quantizer,
              int hnsw_m) {
    const auto type = parse_metric(metric);
    faiss::Index* coarse = nullptr;
    if (quantizer == "flat") coarse = new faiss::IndexFlat(idx_t(dimensions), type);
    else if (quantizer == "hnsw") coarse = new faiss::IndexHNSWFlat(int(dimensions), hnsw_m, type);
    else throw std::invalid_argument("quantizer must be 'flat' or 'hnsw'");
    index_ = std::make_unique<faiss_tieredq::IndexIVFTieredQ>(coarse, dimensions, lists, code_bits, type, metric == "cosine");
    index_->own_fields = true;
  }
  explicit PythonIndex(faiss_tieredq::IndexIVFTieredQ* index) : index_(index) {}

  void check_width(size_t width) const { if (width != size_t(index_->d)) throw std::invalid_argument("array width does not match index"); }
  void train(Matrix rows) {
    check_width(rows.shape(1));
    nb::gil_scoped_release release;
    index_->train(idx_t(rows.shape(0)), rows.data());
  }
  // Fixes the partition: the quantizer holds exactly these centroids (lists x dimensions).
  void set_centroids(Matrix centroids) {
    check_width(centroids.shape(1));
    if (centroids.shape(0) != index_->nlist) throw std::invalid_argument("set_centroids needs one centroid per list");
    if (index_->ntotal) throw std::invalid_argument("set_centroids needs an empty index");
    index_->quantizer->reset();
    index_->quantizer->add(idx_t(centroids.shape(0)), centroids.data());
    index_->quantizer->is_trained = true;
  }
  void add(Matrix rows, std::optional<Ids> ids) {
    check_width(rows.shape(1));
    if (ids && ids->shape(0) != rows.shape(0)) throw std::invalid_argument("ids must match rows");
    nb::gil_scoped_release release;
    index_->add_with_ids(idx_t(rows.shape(0)), rows.data(), ids ? ids->data() : nullptr);
  }
  nb::tuple search(Matrix queries, size_t k, size_t nprobe, std::optional<int> ef_search) const {
    check_width(queries.shape(1));
    const size_t n = queries.shape(0);
    std::vector<float> distances(n * k);
    std::vector<idx_t> labels(n * k);
    {
      nb::gil_scoped_release release;
      faiss::SearchParametersHNSW coarse;
      faiss::SearchParametersIVF params;
      params.nprobe = nprobe;
      if (ef_search) { coarse.efSearch = *ef_search; params.quantizer_params = &coarse; }
      index_->search(idx_t(n), queries.data(), idx_t(k), distances.data(), labels.data(), &params);
    }
    return nb::make_tuple(owned(std::move(distances), n, k), owned(std::move(labels), n, k));
  }
  size_t remove_ids(Ids ids) {
    faiss::IDSelectorBatch selector(ids.shape(0), ids.data());
    return index_->remove_ids(selector);
  }
  void save(const std::string& path) const { faiss_tieredq::write_index_tieredq(*index_, path); }
  static PythonIndex load(const std::string& path) { return PythonIndex(faiss_tieredq::read_index_tieredq(path)); }
  const faiss_tieredq::IndexIVFTieredQ& index() const { return *index_; }

 private:
  std::unique_ptr<faiss_tieredq::IndexIVFTieredQ> index_;
};
}  // namespace

NB_MODULE(_core, module) {
  module.doc() = "IndexIVFTieredQ: a Faiss IVF index on TieredQ fast-scan tiles";
  nb::class_<PythonIndex>(module, "IndexIVFTieredQ")
      .def(nb::init<size_t, size_t, uint32_t, const std::string&, const std::string&, int>(), "dimensions"_a, "lists"_a,
           "code_bits"_a, nb::kw_only(), "metric"_a = "ip", "quantizer"_a = "flat", "hnsw_m"_a = 32,
           "metric 'ip', 'cosine' (the codec normalizes; normalize the rows for the quantizer too) or 'l2'; quantizer\n"
           "'flat' or 'hnsw' (IndexHNSWFlat with hnsw_m links, for 64K+ lists).")
      .def("train", &PythonIndex::train, "rows"_a.noconvert(),
           "Trains the quantizer (unless set_centroids fixed it) and the TieredQ codec on a seeded sample of rows.")
      .def("set_centroids", &PythonIndex::set_centroids, "centroids"_a.noconvert())
      .def("add", &PythonIndex::add, "rows"_a.noconvert(), "ids"_a.noconvert() = nb::none())
      .def("search", &PythonIndex::search, "queries"_a.noconvert(), "k"_a = 10, nb::kw_only(), "nprobe"_a = 1,
           "ef_search"_a = nb::none(),
           "Returns (distances, labels): similarities for 'ip' / 'cosine' (descending), squared distances for 'l2'\n"
           "(ascending); missing results have label -1. ef_search sets an HNSW quantizer's search breadth.")
      .def("remove_ids", &PythonIndex::remove_ids, "ids"_a.noconvert())
      .def("save", &PythonIndex::save, "path"_a)
      .def_static("load", &PythonIndex::load, "path"_a)
      .def_prop_ro("ntotal", [](const PythonIndex& self) { return size_t(self.index().ntotal); })
      .def_prop_ro("nlist", [](const PythonIndex& self) { return self.index().nlist; })
      .def_prop_ro("dimensions", [](const PythonIndex& self) { return size_t(self.index().d); })
      .def_prop_ro("is_trained", [](const PythonIndex& self) { return self.index().is_trained; })
      .def_prop_ro("code_size", [](const PythonIndex& self) { return self.index().code_size; })
      .def_prop_ro("storage_bytes", [](const PythonIndex& self) { return self.index().storage_bytes(); },
                   "Tiles, ids and codec as allocated; excludes the quantizer.");
}
