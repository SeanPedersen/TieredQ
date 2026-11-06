// Sparse centroid graph: nearby-parent candidate pools, neighbor refinement, and connected ring.
#include "internal.h"
#include <algorithm>
#include <numeric>
#include <utility>
namespace geometry_ann {
namespace {
using Candidate = std::pair<double, Id>;
class Builder {
public:
    Builder(std::span<const Leaf> leaves, std::span<const Parent> parents, const BuildOptions& options)
        : leaves_(leaves), parents_(parents), options_(options), owners_(leaves.size()),
          pools_(leaves.size()), previous_(leaves.size()), following_(leaves.size()) {
        graph_.degree = std::min(options.graph_degree, leaves.size() - 1);
        for (std::size_t p = 0; p < parents.size(); ++p)
            for (auto leaf : parents[p].leaves) owners_[leaf] = p;
    }
    GraphTopology run() {
        make_ring(); make_entries(); make_pools();
        for (std::size_t round = 0; round <= options_.graph_refinements; ++round) {
            std::vector<Id> edges; edges.reserve(leaves_.size() * graph_.degree);
            for (Id node = 0; node < leaves_.size(); ++node) {
                auto candidates = pools_[node];
                if (round) {
                    for (auto neighbor : neighbors(node)) {
                        candidates.push_back(neighbor);
                        for (auto next : neighbors(neighbor)) candidates.push_back(next);
                    }
                }
                auto selected = select(node, std::move(candidates));
                edges.insert(edges.end(), selected.begin(), selected.end());
            }
            graph_.edges = std::move(edges);
        }
        return std::move(graph_);
    }
private:
    std::span<const Leaf> leaves_;
    std::span<const Parent> parents_;
    const BuildOptions& options_;
    GraphTopology graph_;
    std::vector<std::size_t> owners_;
    std::vector<Group> pools_;
    std::vector<Id> previous_, following_;
    std::span<const Id> neighbors(Id node) const {
        return std::span(graph_.edges).subspan(std::size_t(node) * graph_.degree, graph_.degree);
    }
    double distance(std::span<const float> a, std::span<const float> b) {
        ++graph_.distance_evaluations; return l2(a, b);
    }
    double distance(Id a, Id b) { return distance(leaves_[a].geometry.centroid, leaves_[b].geometry.centroid); }
    void make_ring() {
        std::size_t d = leaves_.front().geometry.centroid.size();
        std::vector<double> sums(d, 0), squares(d, 0);
        for (const auto& leaf : leaves_) for (std::size_t j = 0; j < d; ++j) {
            double value = leaf.geometry.centroid[j]; sums[j] += value; squares[j] += value * value;
        }
        std::size_t axis = 0;
        for (std::size_t j = 1; j < d; ++j)
            if (squares[j] - sums[j] * sums[j] / leaves_.size() >
                squares[axis] - sums[axis] * sums[axis] / leaves_.size()) axis = j;
        Group order(leaves_.size()); std::iota(order.begin(), order.end(), Id(0));
        std::sort(order.begin(), order.end(), [&](Id a, Id b) {
            float x = leaves_[a].geometry.centroid[axis], y = leaves_[b].geometry.centroid[axis];
            return x < y || (x == y && a < b);
        });
        for (std::size_t i = 0; i < order.size(); ++i) {
            previous_[order[i]] = order[(i + order.size() - 1) % order.size()];
            following_[order[i]] = order[(i + 1) % order.size()];
        }
    }
    void make_entries() {
        for (const auto& parent : parents_) {
            Id best = Id(parent.leaves.front()); double nearest = std::numeric_limits<double>::infinity();
            for (auto leaf : parent.leaves) {
                double candidate = distance(parent.geometry.centroid, leaves_[leaf].geometry.centroid);
                if (candidate < nearest) { nearest = candidate; best = Id(leaf); }
            }
            graph_.entries.push_back(best);
        }
    }
    void make_pools() {
        for (Id node = 0; node < leaves_.size(); ++node) {
            std::vector<Candidate> ranked;
            for (std::size_t p = 0; p < parents_.size(); ++p)
                ranked.emplace_back(distance(leaves_[node].geometry.centroid, parents_[p].geometry.centroid), Id(p));
            std::size_t count = std::min(options_.graph_parent_probes, ranked.size());
            std::partial_sort(ranked.begin(), ranked.begin() + count, ranked.end());
            auto add = [&](std::size_t parent) {
                for (auto leaf : parents_[parent].leaves) pools_[node].push_back(Id(leaf));
            };
            add(owners_[node]);
            for (std::size_t i = 0; i < count; ++i) if (ranked[i].second != owners_[node]) add(ranked[i].second);
        }
    }
    Group select(Id node, Group candidates) {
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        std::vector<Candidate> ranked;
        for (Id candidate : candidates) if (candidate != node) ranked.emplace_back(distance(node, candidate), candidate);
        std::sort(ranked.begin(), ranked.end());
        Group selected;
        auto add = [&](Id id) {
            if (id != node && selected.size() < graph_.degree && std::find(selected.begin(), selected.end(), id) == selected.end())
                selected.push_back(id);
        };
        add(previous_[node]); add(following_[node]);
        for (auto [ignored, candidate] : ranked) {
            (void)ignored;
            if (owners_[candidate] != owners_[node]) { add(candidate); break; }
        }
        for (auto [query_distance, candidate] : ranked) {
            if (selected.size() == graph_.degree) break;
            bool redundant = false;
            for (Id existing : selected)
                if (distance(existing, candidate) < query_distance) { redundant = true; break; }
            if (!redundant) add(candidate);
        }
        for (auto [ignored, candidate] : ranked) { (void)ignored; add(candidate); }
        // The connected ring remains valid even when a tiny parent pool lacks enough candidates.
        for (Id candidate = 0; selected.size() < graph_.degree; ++candidate) add(candidate);
        return selected;
    }
};
}
GraphTopology build_graph(std::span<const Leaf> leaves, std::span<const Parent> parents, const BuildOptions& options) {
    return Builder(leaves, parents, options).run();
}
}
