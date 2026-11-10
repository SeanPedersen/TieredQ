// Budget enforcement and geometry evaluation shared by hierarchy, flat, and graph routing.
#include "search_state.h"
#include <algorithm>
#include <limits>
namespace geometry_ann {
bool VisitGreater::operator()(const Visit& a, const Visit& b) const {
    if (a.priority != b.priority) return a.priority > b.priority;
    if (a.parent != b.parent) return a.parent < b.parent;
    if (a.centroid_distance != b.centroid_distance) return a.centroid_distance > b.centroid_distance;
    return a.index > b.index;
}
SearchState::SearchState(std::span<const float> query, const SearchOptions& options, Variant variant,
                         std::span<const float> vectors, std::span<const Id> ids, std::span<const Leaf> leaves, const Envelope* shared_geometry, QuantizedInput quantized)
    : query_(query), vectors_(vectors), quantized_(quantized), ids_(ids), leaves_(leaves), options_(options), mode_(options.bounds), shared_geometry_(shared_geometry) {
    if (mode_ == BoundsMode::automatic)
        mode_ = variant == Variant::fixed ? BoundsMode::none : variant == Variant::radius ? BoundsMode::radius : BoundsMode::pca;
    if ((mode_ == BoundsMode::shared || mode_ == BoundsMode::shared_residual) && shared_geometry_) {
        shared_query_ = shared_query(query, *shared_geometry_);
        result.shared_query_projections = shared_query_.low.size();
    }
}
std::optional<Visit> SearchState::evaluate(const Envelope& g, std::size_t index, bool parent, BoundsMode override_mode) {
    if (result.metadata_evaluations == options_.metadata_evaluations) {
        result.metadata_exhausted = true; return std::nullopt;
    }
    ++result.metadata_evaluations;
    if (parent) ++result.parents_evaluated;
    double distance = centroid_l2(query_, g.centroid);
    auto mode = override_mode == BoundsMode::automatic ? mode_ : override_mode;
    double bound = 0;
    if (mode != BoundsMode::none) {
        ++result.radius_evaluations;
        bool projections = mode == BoundsMode::pca || mode == BoundsMode::pca_max;
        if (projections) result.projection_evaluations += g.norm.size();
        auto bounds = distance_bounds(query_, g, projections, distance);
        bound = mode == BoundsMode::pca_max ? bounds.max_slab : bounds.joint;
        if ((mode == BoundsMode::shared || mode == BoundsMode::shared_residual) && shared_geometry_) {
            double shape = mode == BoundsMode::shared_residual
                ? shared_residual_lower_bound(shared_query_, *shared_geometry_, g, distance)
                : shared_lower_bound(shared_query_, *shared_geometry_, g);
            bound = std::max(bound, shape);
            result.shared_interval_evaluations += g.shared_low.size();
        }
        result.projection_improvements += bound > bounds.radius;
        result.positive_bounds += bound > 0;
    }
    double priority = distance;
    if (mode != BoundsMode::none) {
        if (options_.ordering == Ordering::bound) priority = bound;
        if (options_.ordering == Ordering::variance) priority = distance / (1 + g.variance);
        if (options_.ordering == Ordering::hybrid)
            priority = (1 - options_.bound_weight) * distance + options_.bound_weight * bound;
        if (parent) priority = bound;
    }
    return Visit{priority, bound, distance, index, parent};
}
bool SearchState::can_scan() const {
    return result.leaves_scanned < options_.leaf_scans && result.vectors_scanned < options_.vector_scans;
}
bool SearchState::discard_if_bounded(const Visit& visit) {
    double threshold = best_.size() < options_.k ? std::numeric_limits<double>::infinity() : best_.top().distance;
    if (visit.bound > threshold) { ++result.pruned; return true; }
    return false;
}
void SearchState::scan(const Visit& visit) {
    if (discard_if_bounded(visit)) return;
    const auto& leaf = leaves_[visit.index];
    if (leaf.size > options_.vector_scans - result.vectors_scanned) {
        ++result.leaves_skipped_for_vector_budget; return;
    }
    ++result.leaves_scanned; result.vectors_scanned += leaf.size;
    for (std::size_t i = leaf.offset; i < leaf.offset + leaf.size; ++i) {
        if (!quantized_.codes.empty()) {
            auto code = quantized_.codes.subspan(i * query_.size(), query_.size());
#ifdef GEOMETRY_FAST_SCAN
            if (best_.size() == options_.k && sq8_lower_bound(query_, code, quantized_.scales, quantized_.offsets) > best_.top().distance) continue;
#endif
            Neighbor candidate{sq8_l2(query_, code, quantized_.scales, quantized_.offsets),ids_[i]};
            if (best_.size() < options_.k) best_.push(candidate);
            else if (candidate < best_.top()) { best_.pop(); best_.push(candidate); }
            continue;
        }
        auto vector = vectors_.subspan(i * query_.size(), query_.size());
#ifdef GEOMETRY_FAST_SCAN
        if (best_.size() == options_.k && scan_lower_bound(query_, vector) > best_.top().distance) continue;
#endif
        Neighbor candidate{l2(query_, vector), ids_[i]};
        if (best_.size() < options_.k) best_.push(candidate);
        else if (candidate < best_.top()) { best_.pop(); best_.push(candidate); }
    }
}
SearchResult SearchState::finish() {
    while (!best_.empty()) { result.neighbors.push_back(best_.top()); best_.pop(); }
    std::reverse(result.neighbors.begin(), result.neighbors.end()); return std::move(result);
}
}
