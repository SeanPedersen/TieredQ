// Existing hierarchical and exhaustive leaf-centroid policies using shared query mechanics.
#include "search_state.h"
#include <chrono>
#include <algorithm>
namespace geometry_ann {
SearchResult Index::search_hierarchy(std::span<const float> query, const SearchOptions& options) const {
    auto start = std::chrono::steady_clock::now();
    SearchState state(query, options, variant_, vectors_, ids_, leaves_, &shared_geometry_, {codes_,code_scales_,code_offsets_});
    std::vector<Visit> initial;
    initial.reserve(options.routing == Routing::flat ? leaves_.size() : parents_.size());
    if (options.routing == Routing::flat) {
        for (std::size_t i = 0; i < leaves_.size(); ++i) {
            auto visit = state.evaluate(leaves_[i].geometry, i, false);
            if (!visit) break;
            initial.push_back(*visit);
        }
    } else {
        for (std::size_t i = 0; i < parents_.size(); ++i) {
            auto visit = state.evaluate(parents_[i].geometry, i, true);
            if (!visit) break;
            initial.push_back(*visit);
        }
    }
    if (options.routing == Routing::flat) {
        if (options.bounds==BoundsMode::support) {
            std::vector<double> distances; distances.reserve(initial.size());
            for (const auto& v:initial) distances.push_back(v.centroid_distance);
            for (auto& v:initial) {
                auto shape=edge_support_lower_bound(std::span(edge_supports_).subspan(v.index*graph_degree_,graph_degree_),
                    std::span(graph_edges_).subspan(v.index*graph_degree_,graph_degree_),distances,v.centroid_distance,dimension_);
                auto previous=v.bound; v.bound=std::max(v.bound,shape);
                state.result.positive_bounds+=previous==0 && v.bound>0;
                state.result.projection_improvements+=v.bound>previous;
                state.result.projection_evaluations+=graph_degree_;
                if (options.ordering==Ordering::bound) v.priority=v.bound;
                else if (options.ordering==Ordering::hybrid)
                    v.priority=(1-options.bound_weight)*v.centroid_distance+options.bound_weight*v.bound;
            }
        }
        std::make_heap(initial.begin(), initial.end(), VisitGreater{});
        auto routed = std::chrono::steady_clock::now();
        while (!initial.empty() && state.can_scan()) {
            std::pop_heap(initial.begin(), initial.end(), VisitGreater{});
            auto visit = initial.back(); initial.pop_back();
            state.scan(visit);
        }
        auto result = state.finish();
        result.routing_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(routed - start).count();
        result.scanning_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - routed).count();
        return result;
    }
    VisitQueue queue(VisitGreater{}, std::move(initial));
    auto routed = std::chrono::steady_clock::now();
    while (!queue.empty() && state.can_scan()) {
        auto visit = queue.top(); queue.pop();
        if (!visit.parent) { state.scan(visit); continue; }
        if (state.discard_if_bounded(visit)) continue;
        // Descendant leaves carry valid individual bounds; no graph descendants are pruned here.
        for (auto child : parents_[visit.index].leaves) {
            auto next = state.evaluate(leaves_[child].geometry, child, false);
            if (!next) break;
            queue.push(*next);
        }
    }
    auto result = state.finish();
    // For hierarchy, routing and scanning are interleaved after initial parents.
    // The hierarchy split measures initial parent setup, not all routing work.
    result.routing_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(routed - start).count();
    result.scanning_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - routed).count();
    return result;
}
}
