// nanobind bindings: HullRouter keeps references to the numpy arrays it views (no copies).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <omp.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <tuple>

#include "kernels.hpp"
#include "partition.hpp"
#include "router.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <typename T, typename Dim>
using Array = nb::ndarray<const T, Dim, nb::c_contig, nb::device::cpu>;
using F1 = Array<float, nb::ndim<1>>;
using F2 = Array<float, nb::ndim<2>>;
using U2 = Array<uint8_t, nb::ndim<2>>;
using I1 = Array<int64_t, nb::ndim<1>>;
using U32_2 = Array<uint32_t, nb::ndim<2>>;

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> owned_1d(const std::vector<T>& values) {
    auto* data = new T[values.size()];
    std::copy(values.begin(), values.end(), data);
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(data, {values.size()}, owner);
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> owned_2d(T* data, size_t rows, size_t cols) {
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(data, {rows, cols}, owner);
}

void expect(bool ok, const std::string& what) {
    if (!ok) throw std::invalid_argument("HullRouter: " + what);
}

class PyRouter {
   public:
    PyRouter(F1 mean, F2 proj, F1 proj_offset, F2 coarse_z, U2 fine_codes, F1 fine_lo, F1 fine_step, F1 fine_bias,
             U2 sketch_codes, F1 sketch_lo, F1 sketch_step, F1 sketch_bias, F1 sketch_resid, F2 vectors, I1 ids,
             I1 coarse_offsets, I1 fine_offsets, size_t fine_per_coarse, float gamma, size_t prefix_dims,
             size_t fine_used_dims)
        : arrays_(mean, proj, proj_offset, coarse_z, fine_codes, fine_lo, fine_step, fine_bias, sketch_codes,
                  sketch_lo, sketch_step, sketch_bias, sketch_resid, vectors, ids, coarse_offsets, fine_offsets),
          router_(make_view(fine_per_coarse, gamma, prefix_dims, fine_used_dims)) {}

    std::tuple<nb::ndarray<nb::numpy, float, nb::ndim<2>>, nb::ndarray<nb::numpy, int64_t, nb::ndim<2>>,
               nb::ndarray<nb::numpy, uint32_t, nb::ndim<2>>>
    search(F2 queries, size_t k, size_t shortlist, size_t fine_cells, size_t rerank, size_t refine, size_t gap, size_t hops,
           size_t expand_from, size_t expand_k, size_t expand_exact, size_t n_threads) const {
        expect(queries.shape(1) == dim_, "query dim mismatch");
        const size_t nq = queries.shape(0);
        auto* dist = new float[nq * k];
        auto* ids = new int64_t[nq * k];
        constexpr size_t stat_cols = sizeof(hh::QueryStats) / sizeof(uint32_t);
        auto* stats = new uint32_t[nq * stat_cols];
        const hh::SearchParams params{k, shortlist, fine_cells, rerank, refine, gap, hops, expand_from, expand_k, expand_exact};
        {
            nb::gil_scoped_release release;
            router_.search_batch(queries.data(), nq, params, n_threads, dist, ids, reinterpret_cast<hh::QueryStats*>(stats));
        }
        return {owned_2d(dist, nq, k), owned_2d(ids, nq, k), owned_2d(stats, nq, stat_cols)};
    }

    void build_graph(size_t k, size_t shortlist, size_t fine_cells, size_t rerank, size_t refine, bool reverse_edges,
                     size_t refine_fanout, size_t n_threads) {
        nb::gil_scoped_release release;
        router_.build_graph(k, {k, shortlist, fine_cells, rerank, refine}, n_threads, reverse_edges);
        if (refine_fanout > 0) router_.refine_graph(refine_fanout, n_threads);
    }

    void set_graph(U32_2 graph) {
        expect(graph.shape(0) == n_vectors_, "graph rows must equal the vector count");
        nb::gil_scoped_release release;
        router_.set_graph(graph.data(), graph.shape(1));
    }

    nb::ndarray<nb::numpy, uint32_t, nb::ndim<2>> graph() const {
        const size_t k = router_.graph_k(), n = k ? router_.graph().size() / k : 0;
        auto* copy = new uint32_t[n * k];
        std::copy(router_.graph().begin(), router_.graph().end(), copy);
        return owned_2d(copy, n, k);
    }

   private:
    hh::IndexView make_view(size_t m, float gamma, size_t prefix_dims, size_t fine_used_dims) {
        const auto& [mean, proj, proj_offset, coarse_z, fine_codes, fine_lo, fine_step, fine_bias, sketch_codes,
                     sketch_lo, sketch_step, sketch_bias, sketch_resid, vectors, ids, coarse_offsets,
                     fine_offsets] = arrays_;
        const size_t n = vectors.shape(0), pca = proj.shape(0), n_coarse = coarse_z.shape(0);
        const size_t fine_dims = 2 * fine_codes.shape(1);
        dim_ = vectors.shape(1);
        n_vectors_ = n;
        expect(proj.shape(1) == dim_ && mean.shape(0) == dim_ && proj_offset.shape(0) == pca, "projection shape");
        expect(fine_codes.shape(0) == n_coarse * m && fine_lo.shape(0) == fine_dims && fine_step.shape(0) == fine_dims,
               "fine_codes shape");
        const size_t sketch_bits = 8 * sketch_codes.shape(1) / pca;
        expect(sketch_codes.shape(0) == n && sketch_codes.shape(1) * 8 == pca * sketch_bits, "sketch_codes shape");
        expect(fine_bias.shape(0) == n_coarse * m && sketch_bias.shape(0) == n && sketch_resid.shape(0) == n,
               "bias shape");
        expect(ids.shape(0) == n && coarse_offsets.shape(0) == n_coarse + 1 &&
                   fine_offsets.shape(0) == n_coarse * m + 1,
               "offsets shape");
        return {n,
                dim_,
                pca,
                coarse_z.shape(1),
                fine_dims,
                sketch_bits,
                n_coarse,
                m,
                mean.data(),
                proj.data(),
                proj_offset.data(),
                coarse_z.data(),
                fine_codes.data(),
                fine_lo.data(),
                fine_step.data(),
                fine_bias.data(),
                sketch_codes.data(),
                sketch_lo.data(),
                sketch_step.data(),
                sketch_bias.data(),
                sketch_resid.data(),
                vectors.data(),
                ids.data(),
                coarse_offsets.data(),
                fine_offsets.data(),
                gamma,
                prefix_dims,
                fine_used_dims};
    }

    std::tuple<F1, F2, F1, F2, U2, F1, F1, F1, U2, F1, F1, F1, F1, F2, I1, I1, I1> arrays_;
    size_t dim_ = 0, n_vectors_ = 0;
    hh::Router router_;
};

// OpenMP thread budget of TieredQ's training for one call (restored afterwards)
class OmpThreads {
   public:
    explicit OmpThreads(size_t threads) : previous_(omp_get_max_threads()) {
        if (threads) omp_set_num_threads(static_cast<int>(threads));
    }
    ~OmpThreads() { omp_set_num_threads(previous_); }
    OmpThreads(const OmpThreads&) = delete;
    OmpThreads& operator=(const OmpThreads&) = delete;

   private:
    int previous_;
};

class PyPartition {
   public:
    PyPartition(F2 rows, size_t coarse_lists, size_t fine_per_coarse, double coarse_epochs, uint32_t sample_rows_per_list,
                double fine_epochs, uint64_t seed, size_t n_threads)
        : partition_(build(rows, {coarse_lists, fine_per_coarse, coarse_epochs, sample_rows_per_list, fine_epochs, seed},
                           n_threads)) {}

    auto order() const { return owned_1d(partition_.order); }
    auto coarse_offsets() const { return owned_1d(partition_.coarse_offsets); }
    auto fine_offsets() const { return owned_1d(partition_.fine_offsets); }
    auto coarse_centroids() const { return centroid_matrix(partition_.coarse_centroids); }
    auto fine_centroids() const { return centroid_matrix(partition_.fine_centroids); }
    nb::dict build_stats() const {
        nb::dict out;
        out["coarse_s"] = partition_.stats.coarse_s;
        out["fine_s"] = partition_.stats.fine_s;
        return out;
    }

   private:
    static hh::Partition build(F2 rows, const hh::PartitionConfig& config, size_t n_threads) {
        nb::gil_scoped_release release;
        const OmpThreads threads(n_threads);
        return hh::Partition(rows.data(), rows.shape(0), rows.shape(1), config);
    }

    nb::ndarray<nb::numpy, float, nb::ndim<2>> centroid_matrix(const std::vector<float>& values) const {
        auto* data = new float[values.size()];
        std::copy(values.begin(), values.end(), data);
        return owned_2d(data, values.size() / partition_.dim(), partition_.dim());
    }

    hh::Partition partition_;
};

}  // namespace

NB_MODULE(_hyperhull, m) {
    m.attr("SIMD_LEVEL") = hh::kSimdLevel;
    nb::class_<PyRouter>(m, "HullRouter")
        .def(nb::init<F1, F2, F1, F2, U2, F1, F1, F1, U2, F1, F1, F1, F1, F2, I1, I1, I1, size_t, float, size_t, size_t>(),
             "mean"_a, "proj"_a, "proj_offset"_a, "coarse_z"_a, "fine_codes"_a, "fine_lo"_a, "fine_step"_a,
             "fine_bias"_a, "sketch_codes"_a, "sketch_lo"_a, "sketch_step"_a, "sketch_bias"_a, "sketch_resid"_a,
             "vectors"_a, "ids"_a, "coarse_offsets"_a, "fine_offsets"_a, "fine_per_coarse"_a, "gamma"_a,
             "prefix_dims"_a = 0, "fine_used_dims"_a = 0)
        .def("search", &PyRouter::search, "queries"_a, "k"_a = 10, "shortlist"_a, "fine_cells"_a, "rerank"_a,
             "refine"_a = 0, "gap"_a = 0, "hops"_a = 0, "expand_from"_a = 0, "expand_k"_a = 0, "expand_exact"_a = 0,
             "n_threads"_a = 1,
             "Returns (distances, ids, stats) per query; stats columns: sketched, reranked, expanded, ns per stage "
             "(project, coarse, fine, sketch, refine, rerank, expand). One thread = one query stream. "
             "hops > 0 expands the results along the k-NN graph (see build_graph).")
        .def("build_graph", &PyRouter::build_graph, "k"_a = 16, "shortlist"_a = 64, "fine_cells"_a = 128, "rerank"_a = 60,
             "refine"_a = 400, "reverse_edges"_a = true, "refine_fanout"_a = 0, "n_threads"_a = 32,
             "k-NN graph by self-search with the given params (+ reverse edges, + optional NN-descent round).")
        .def("set_graph", &PyRouter::set_graph, "graph"_a,
             "Adopt an external k-NN graph: n_vectors × k neighbour rows (stored-row space, uint32).")
        .def("graph", &PyRouter::graph, "Copy of the k-NN graph (rows × k neighbour rows in index order).");

    nb::class_<PyPartition>(m, "Partition",
                            "Two-level partition trained with TieredQ e-means: coarse lists as TieredQIVFFlatIndex trains them, "
                            "fine cells inside each list (one list per thread); rows laid out coarse-major, fine-minor.")
        .def(nb::init<F2, size_t, size_t, double, uint32_t, double, uint64_t, size_t>(), "rows"_a, "coarse_lists"_a = 1024,
             "fine_per_coarse"_a = 32, "coarse_epochs"_a = 2.5, "sample_rows_per_list"_a = 256, "fine_epochs"_a = 2.5,
             "seed"_a = 42, "n_threads"_a = 0, "n_threads 0 keeps the OpenMP default.")
        .def_prop_ro("order", &PyPartition::order, nb::rv_policy::take_ownership, "Stored position → original row (int64).")
        .def_prop_ro("coarse_offsets", &PyPartition::coarse_offsets, nb::rv_policy::take_ownership)
        .def_prop_ro("fine_offsets", &PyPartition::fine_offsets, nb::rv_policy::take_ownership)
        .def_prop_ro("coarse_centroids", &PyPartition::coarse_centroids, nb::rv_policy::take_ownership)
        .def_prop_ro("fine_centroids", &PyPartition::fine_centroids, nb::rv_policy::take_ownership,
                     "coarse_lists · fine_per_coarse rows; empty slots repeat a sibling (or the coarse) centroid.")
        .def_prop_ro("build_stats", &PyPartition::build_stats, nb::rv_policy::take_ownership, "Wall seconds: coarse_s (training + assignment), fine_s.");
}
