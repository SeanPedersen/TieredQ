// Flat routing over a compressed centroid table: approximate leaf order, exact re-rank of its head, then scanning.
// Ranking may grow lazily: deeper top-k lists extend shallower ones, so scanning resumes where it stopped.
#include "compressed_routing.h"
#include "search_state.h"
#include <algorithm>
#include <chrono>
namespace geometry_ann {
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t growth_factor = 4;
std::size_t nanoseconds(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}
}
SearchResult Index::search_compressed(std::span<const float> query, const SearchOptions& options) const {
    auto start = Clock::now();
    SearchState state(query, options, variant_, vectors_, ids_, leaves_, &shared_geometry_, {codes_,code_scales_,code_offsets_});
    const auto limit = std::min(options.routing_candidates, leaves_.size());
    // The exact head is re-ranked once, so the first round must contain all of it.
    auto depth = options.routing_initial ? std::min(std::max(options.routing_initial, options.metadata_evaluations), limit) : limit;
    std::vector<Visit> order;
    std::size_t routing_ns = 0, scanned = 0;
    while (true) {
        auto ranking = Clock::now();
        auto ranked = router_->rank(query, depth);
        std::sort(ranked.begin(), ranked.end(), [](const RankedLeaf& a, const RankedLeaf& b) {
            return a.distance < b.distance || (a.distance == b.distance && a.leaf < b.leaf);
        });
        // Leaves already in `order` are the prefix of this deeper ranking; only the new tail is appended.
        auto exact = std::min(ranked.size(), options.metadata_evaluations);
        for (std::size_t i = order.size(); i < ranked.size(); ++i) {
            if (i < exact) order.push_back(*state.evaluate(leaves_[ranked[i].leaf].geometry, ranked[i].leaf, false, BoundsMode::none));
            else order.push_back({ranked[i].distance, 0, ranked[i].distance, ranked[i].leaf, false});
        }
        if (!state.result.routing_rounds)
            std::sort(order.begin(), order.begin() + std::ptrdiff_t(exact), [](const Visit& a, const Visit& b) { return VisitGreater{}(b, a); });
        ++state.result.routing_rounds;
        routing_ns += nanoseconds(ranking);
        for (; scanned < order.size() && state.can_scan(); ++scanned) state.scan(order[scanned]);
        if (!state.can_scan() || depth == limit) break;
        depth = std::min(limit, depth * growth_factor);
    }
    auto result = state.finish();
    result.routing_ns = routing_ns;
    result.scanning_ns = nanoseconds(start) - routing_ns;
    return result;
}
}
