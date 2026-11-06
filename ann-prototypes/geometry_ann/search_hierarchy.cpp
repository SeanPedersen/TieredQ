// Existing hierarchical and exhaustive leaf-centroid policies using shared query mechanics.
#include "search_state.h"
namespace geometry_ann {
SearchResult Index::search_hierarchy(std::span<const float> query, const SearchOptions& options) const {
    SearchState state(query, options, variant_, vectors_, ids_, leaves_, &shared_geometry_);
    VisitQueue queue;
    if (options.routing == Routing::flat) {
        for (std::size_t i = 0; i < leaves_.size(); ++i) {
            auto visit = state.evaluate(leaves_[i].geometry, i, false);
            if (!visit) break;
            queue.push(*visit);
        }
    } else {
        for (std::size_t i = 0; i < parents_.size(); ++i) {
            auto visit = state.evaluate(parents_[i].geometry, i, true);
            if (!visit) break;
            queue.push(*visit);
        }
    }
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
    return state.finish();
}
}
