// Index assembly, bounded best-first traversal, and logical persistent memory accounting.
#include "internal.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
namespace geometry_ann {
Index::Index(std::span<const float> input, std::size_t d, const BuildOptions& o, QuantizedInput quantized)
    : dimension_(d), variant_(o.variant), geometry_metadata_(o.variant != Variant::fixed || o.geometry_metadata) {
    Matrix data(input,quantized);
    if (!d || !data.size() || data.size() % d || data.size() / d > std::numeric_limits<Id>::max())
        throw std::invalid_argument("invalid vector shape or ID capacity");
    if (!o.coarse || !o.children || !o.iterations || !o.training_sample || !o.target_leaf_size ||
        !o.pca_sample || o.directions > 4 || o.leaf_budget < std::min(o.coarse, data.size() / d))
        throw std::invalid_argument("invalid build options (PCA directions must be 0..4)");
    if (quantized.codes.empty()) for (float value : input) if (!std::isfinite(value)) throw std::invalid_argument("nonfinite vector");
    if (o.graph_degree && (o.graph_degree < 4 || o.graph_degree > 32 || !o.graph_parent_probes || o.graph_refinements > 8))
        throw std::invalid_argument("graph degree must be 4..32, parent probes positive, and refinements <= 8");
    if (!quantized.codes.empty()) {
        if ((!input.empty() && quantized.codes.size() != input.size()) || quantized.scales.size() != d || (!quantized.offsets.empty() && quantized.offsets.size() != d))
            throw std::invalid_argument("invalid SQ8 shape");
        for (float scale : quantized.scales) if (!std::isfinite(scale) || scale <= 0 || scale > std::numeric_limits<float>::max() / 128)
            throw std::invalid_argument("invalid SQ8 scale");
        code_scales_.assign(quantized.scales.begin(),quantized.scales.end());
        code_offsets_.assign(d,0);
        if (!quantized.offsets.empty()) code_offsets_.assign(quantized.offsets.begin(),quantized.offsets.end());
        for (std::size_t j=0;j<d;++j)
            if (!std::isfinite(code_offsets_[j]) || std::abs(double(code_offsets_[j]))+128*double(code_scales_[j]) > std::numeric_limits<float>::max())
                throw std::invalid_argument("invalid SQ8 offset/range");
    } else if (!quantized.scales.empty() || !quantized.offsets.empty()) throw std::invalid_argument("SQ8 scales without codes");
    using Clock = std::chrono::steady_clock;
    auto elapsed = [](Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); };
    if (o.neighbor_support && (!o.graph_degree || !geometry_metadata_))
        throw std::invalid_argument("neighbor supports require graph and radius metadata");
    if (o.shared_directions > 32 || (o.shared_directions && !o.shared_sample))
        throw std::invalid_argument("invalid shared projection configuration");
    auto start = Clock::now();
    Group ids(data.size() / d); std::iota(ids.begin(), ids.end(), Id(0));
    std::mt19937 rng(o.seed);
    auto coarse = cluster(data, d, ids, o.coarse, o, rng);
    statistics_.clustering_s = elapsed(start);
    auto measure_geometry = [&](const Group& group) {
        auto stage = Clock::now(); auto result = centroid_envelope(data, d, group);
        statistics_.centroids_s += elapsed(stage);
        if (geometry_metadata_) {
            stage = Clock::now(); add_radius(result, data, d, group); statistics_.radius_s += elapsed(stage);
        }
        return result;
    };
    for (const auto& group : coarse) parents_.push_back({measure_geometry(group), {}});
    std::vector<Group> groups;
    std::vector<std::size_t> owners;
    start = Clock::now();
    if (o.variant == Variant::fixed) {
        for (std::size_t parent = 0; parent < coarse.size(); ++parent) {
            auto children = cluster(data, d, coarse[parent], o.children, o, rng);
            for (auto& child : children) { owners.push_back(parent); groups.push_back(std::move(child)); }
        }
    } else groups = adaptive(data, d, std::move(coarse), o, owners, rng);
    statistics_.clustering_s += elapsed(start);
    if (o.shared_directions) {
        auto stage = Clock::now();
        constexpr std::uint32_t shared_seed_salt = 0x53484152;
        std::mt19937 shared_rng(o.seed ^ shared_seed_salt);
        Group sample = ids; std::shuffle(sample.begin(), sample.end(), shared_rng);
        sample.resize(std::min(sample.size(), o.shared_sample));
        shared_geometry_ = envelope(data, d, sample, o.shared_directions, o.shared_sample, shared_rng);
        statistics_.shared_basis_s = elapsed(stage);
    }
    if (quantized.codes.empty()) vectors_.reserve(data.size());
    ids_.reserve(ids.size()); leaves_.reserve(groups.size());
    for (std::size_t i = 0; i < groups.size(); ++i) {
        auto& group = groups[i];
        parents_[owners[i]].leaves.push_back(i);
        auto geometry = measure_geometry(group);
        if (o.variant == Variant::pca) {
            auto stage = Clock::now(); add_projections(geometry, data, d, group, o.directions, o.pca_sample, rng);
            statistics_.projections_s += elapsed(stage);
        }
        if (o.shared_directions) {
            auto stage = Clock::now(); add_shared_ranges(geometry, shared_geometry_, data, d, group);
            statistics_.shared_ranges_s += elapsed(stage);
        }
        auto packing_start = Clock::now();
        leaves_.push_back({std::move(geometry), ids_.size(), group.size()});
        ids_.insert(ids_.end(), group.begin(), group.end());
        statistics_.packing_s += elapsed(packing_start);
    }
    // Finish geometry before allocating the packed corpus. Stream the input in
    // original row order so file-backed inputs do not cause random disk reads
    // once the owned index displaces source pages from RAM.
    start = Clock::now();
    std::vector<Id> destination(ids_.size());
    for (std::size_t packed = 0; packed < ids_.size(); ++packed) destination[ids_[packed]] = Id(packed);
    if (quantized.codes.empty()) {
        vectors_.resize(data.size());
        for (std::size_t id = 0; id < destination.size(); ++id)
            std::copy_n(data.data() + id * d, d, vectors_.data() + std::size_t(destination[id]) * d);
    } else {
        codes_.resize(quantized.codes.size());
        for (std::size_t id = 0; id < destination.size(); ++id)
            std::copy_n(quantized.codes.data() + id * d, d, codes_.data() + std::size_t(destination[id]) * d);
    }
    statistics_.packing_s += elapsed(start);
    if (o.graph_degree) {
        start = Clock::now(); auto graph = build_graph(leaves_, parents_, o);
        graph_edges_ = std::move(graph.edges); graph_entries_ = std::move(graph.entries); graph_degree_ = graph.degree;
        statistics_.graph_distance_evaluations = graph.distance_evaluations; statistics_.graph_s = elapsed(start);
        if (o.neighbor_support) {
            start=Clock::now();
            Matrix packed(std::span<const float>(vectors_), {codes_,code_scales_,code_offsets_});
            edge_supports_=build_edge_supports(packed,d,leaves_,graph_edges_,graph_degree_);
            statistics_.support_s=elapsed(start);
        }
    }
}
SearchResult Index::search(std::span<const float> q, const SearchOptions& o) const {

    if (o.bounds==BoundsMode::support && (edge_supports_.empty() || o.routing!=Routing::flat || o.metadata_evaluations<leaves_.size()))
        throw std::invalid_argument("support search requires built edge supports and full flat centroid evaluation");
    if (q.size() != dimension_ || !o.k) throw std::invalid_argument("invalid query shape or k");
    for (float value : q) if (!std::isfinite(value)) throw std::invalid_argument("nonfinite query");
    if ((o.bounds == BoundsMode::pca || o.bounds == BoundsMode::pca_max) && variant_ != Variant::pca)
        throw std::invalid_argument("PCA search requires a PCA index");
    if (o.bounds == BoundsMode::radius && !geometry_metadata_)
        throw std::invalid_argument("radius search requires radius metadata");
    if (!std::isfinite(o.bound_weight) || o.bound_weight < 0 || o.bound_weight > 1)
        throw std::invalid_argument("bound weight must be within [0,1]");
    if (!o.leaf_scans || !o.vector_scans) return {};
    if (!o.metadata_evaluations) { SearchResult result; result.metadata_exhausted = true; return result; }
    if ((o.bounds == BoundsMode::shared || o.bounds == BoundsMode::shared_residual) && (shared_geometry_.centroid.empty() || !geometry_metadata_))
        throw std::invalid_argument("shared envelope search requires built shared bounds and radii");
    if (o.routing == Routing::graph) {
        if (graph_entries_.empty() || !o.graph_seeds) throw std::invalid_argument("graph routing requires a graph and seed count");
        return search_graph(q, o);
    }
    return search_hierarchy(q, o);
}

namespace {
std::size_t envelope_bytes(const Envelope& g) {
    return (g.centroid.capacity() + g.directions.capacity()) * sizeof(float) +
           (g.low.capacity() + g.high.capacity() + g.norm.capacity() + g.shared_low.capacity() + g.shared_high.capacity() + g.shared_center_low.capacity() + g.shared_center_high.capacity()) * sizeof(double);
}
}
std::size_t Index::raw_bytes() const { return vectors_.capacity() * sizeof(float) + codes_.capacity(); }
std::size_t Index::overhead_bytes() const {
    std::size_t bytes = edge_supports_.capacity()*sizeof(EdgeSupport) + sizeof(*this) + (code_scales_.capacity()+code_offsets_.capacity()) * sizeof(float) + envelope_bytes(shared_geometry_) + graph_bytes() + ids_.capacity() * sizeof(Id) + leaves_.capacity() * sizeof(Leaf) +
                        parents_.capacity() * sizeof(Parent);
    for (const auto& parent : parents_) bytes += envelope_bytes(parent.geometry) + parent.leaves.capacity() * sizeof(std::size_t);
    for (const auto& leaf : leaves_) bytes += envelope_bytes(leaf.geometry);
    return bytes;
}
std::size_t Index::bounds_bytes(BoundsMode mode) const {
    if (mode == BoundsMode::none) return 0;
    std::size_t bytes = (parents_.size() + leaves_.size()) * sizeof(double) * 2;
    if (mode == BoundsMode::support) return bytes + edge_supports_.capacity()*sizeof(EdgeSupport);
    if (mode == BoundsMode::shared || mode == BoundsMode::shared_residual) {
        bytes += envelope_bytes(shared_geometry_);
        for (const auto& leaf : leaves_)
            bytes += (leaf.geometry.shared_low.capacity() + leaf.geometry.shared_high.capacity() + leaf.geometry.shared_center_low.capacity() + leaf.geometry.shared_center_high.capacity()) * sizeof(double);
        return bytes;
    }
    if (mode != BoundsMode::pca && mode != BoundsMode::pca_max) return bytes;
    for (const auto& leaf : leaves_) {
        const auto& g = leaf.geometry;
        bytes += g.directions.capacity() * sizeof(float) +
                 (g.low.capacity() + g.high.capacity() + g.norm.capacity() + g.shared_low.capacity() + g.shared_high.capacity() + g.shared_center_low.capacity() + g.shared_center_high.capacity()) * sizeof(double);
    }
    return bytes;
}

}
