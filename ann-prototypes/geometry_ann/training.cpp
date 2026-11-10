// Sampled Lloyd training and best-score adaptive bisection without point-level graphs.
#include "internal.h"
#include "parallel.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
namespace geometry_ann {
std::vector<Group> cluster(Matrix data, std::size_t d, const Group& ids,
                           std::size_t count, const BuildOptions& o, std::mt19937& rng) {
    if (ids.empty()) return {};
    count = std::min(count, ids.size());
    Group sample = ids;
    std::shuffle(sample.begin(), sample.end(), rng);
    sample.resize(std::min(sample.size(), std::max(count, o.training_sample)));
    std::vector<float> centers(count * d);
    for (std::size_t c = 0; c < count; ++c)
        std::copy_n(decoded_row(data,d,sample[c]).begin(), d, centers.begin() + c * d);
    auto nearest = [&](Id id) {
        std::size_t best = 0;
        double distance = std::numeric_limits<double>::infinity();
        for (std::size_t c = 0; c < count; ++c) {
            double candidate = data.distance(std::span(centers).subspan(c*d,d),id,d);
            if (candidate < distance) { distance = candidate; best = c; }
        }
        return best;
    };
    // Assignments are independent and run in parallel; sums accumulate serially in sample
    // order, so partitions are bit-identical for every thread count.
    auto assign = [&](const Group& members) {
        std::vector<std::size_t> owner(members.size());
        parallel_for(members.size(), o.build_threads, [&](std::size_t i, std::size_t) { owner[i] = nearest(members[i]); });
        return owner;
    };
    for (std::size_t iteration = 0; iteration < o.iterations; ++iteration) {
        std::vector<double> sums(count * d, 0);
        std::vector<std::size_t> sizes(count, 0);
        auto owner = assign(sample);
        for (std::size_t i = 0; i < sample.size(); ++i) {
            auto c = owner[i]; Id id = sample[i]; ++sizes[c];
            for (std::size_t j = 0; j < d; ++j) sums[c * d + j] += data.value(id,j,d);
        }
        for (std::size_t c = 0; c < count; ++c) {
            if (!sizes[c]) continue;
            for (std::size_t j = 0; j < d; ++j) centers[c * d + j] = float(sums[c * d + j] / sizes[c]);
        }
    }
    std::vector<Group> groups(count);
    auto owner = assign(ids);
    for (std::size_t i = 0; i < ids.size(); ++i) groups[owner[i]].push_back(ids[i]);
    std::erase_if(groups, [](const Group& g) { return g.empty(); });
    return groups;
}
namespace {
struct Split { double score; std::size_t axis; };
Split split_score(Matrix data, std::size_t d, const Group& ids, std::size_t target) {
    if (ids.size() <= target) return {0, 0};
    std::vector<double> mean(d, 0), variance(d, 0);
    for (Id id : ids) for (std::size_t j = 0; j < d; ++j) mean[j] += data.value(id,j,d);
    for (auto& value : mean) value /= ids.size();
    for (Id id : ids) for (std::size_t j = 0; j < d; ++j) {
        double delta = data.value(id,j,d) - mean[j]; variance[j] += delta * delta;
    }
    auto axis = std::size_t(std::max_element(variance.begin(), variance.end()) - variance.begin());
    double total = std::accumulate(variance.begin(), variance.end(), 0.0);
    if (total == 0) return {0, axis};
    // Total scatter already combines population and dispersion; ratio is axis anisotropy.
    return {total * (1 + variance[axis] / total), axis};
}
}
std::vector<Group> adaptive(Matrix data, std::size_t d, std::vector<Group> groups,
                            const BuildOptions& o, std::vector<std::size_t>& owners, std::mt19937& rng) {
    using Candidate = std::pair<double, std::size_t>;
    std::priority_queue<Candidate> queue;
    for (std::size_t i = 0; i < groups.size(); ++i) {
        owners.push_back(i); queue.emplace(split_score(data, d, groups[i], o.target_leaf_size).score, i);
    }
    while (groups.size() < o.leaf_budget && !queue.empty()) {
        auto [score, index] = queue.top(); queue.pop();
        if (score <= 0) break;
        auto axis = split_score(data, d, groups[index], o.target_leaf_size).axis;
        std::vector<Group> children;
        if (o.local_kmeans_split) children = cluster(data, d, groups[index], 2, o, rng);
        if (children.size() != 2) {
            Group ids = std::move(groups[index]);
            auto middle = ids.begin() + ids.size() / 2;
            std::nth_element(ids.begin(), middle, ids.end(), [&](Id a, Id b) {
                return data.value(a,axis,d) < data.value(b,axis,d);
            });
            children = {Group(ids.begin(), middle), Group(middle, ids.end())};
        }
        groups[index] = std::move(children[0]);
        groups.push_back(std::move(children[1])); owners.push_back(owners[index]);
        for (auto child : {index, groups.size() - 1})
            queue.emplace(split_score(data, d, groups[child], o.target_leaf_size).score, child);
    }
    return groups;
}
std::size_t refine_leaves(Matrix data, std::size_t d, std::vector<Group>& leaves, std::vector<std::size_t>& owners,
                          std::span<const float> parent_centroids, const BuildOptions& o) {
    const std::size_t parent_count = parent_centroids.size() / d;
    const std::size_t probes = std::min(o.reassign_parents, parent_count);
    std::size_t rows = 0;
    for (const auto& leaf : leaves) rows += leaf.size();
    std::vector<std::uint32_t> current(rows);
    for (std::size_t l = 0; l < leaves.size(); ++l) for (Id id : leaves[l]) current[id] = std::uint32_t(l);
    std::size_t changed = 0;
    for (std::size_t iteration = 0; iteration < o.reassign_iterations; ++iteration) {
        std::vector<float> centers(leaves.size() * d);
        parallel_for(leaves.size(), o.build_threads, [&](std::size_t l, std::size_t) {
            std::vector<double> sum(d, 0);
            for (Id id : leaves[l]) for (std::size_t j = 0; j < d; ++j) sum[j] += data.value(id, j, d);
            for (std::size_t j = 0; j < d; ++j) centers[l * d + j] = float(sum[j] / double(leaves[l].size()));
        });
        std::vector<std::vector<std::uint32_t>> by_parent(parent_count);
        for (std::size_t l = 0; l < leaves.size(); ++l) by_parent[owners[l]].push_back(std::uint32_t(l));
        std::vector<std::vector<std::pair<double, std::uint32_t>>> scratch(std::max<std::size_t>(1, o.build_threads));
        std::vector<std::uint32_t> next(rows);
        parallel_for(rows, o.build_threads, [&](std::size_t row, std::size_t thread) {
            auto& ranked = scratch[thread]; ranked.clear();
            for (std::size_t p = 0; p < parent_count; ++p)
                ranked.emplace_back(data.distance(parent_centroids.subspan(p * d, d), Id(row), d), std::uint32_t(p));
            std::partial_sort(ranked.begin(), ranked.begin() + probes, ranked.end());
            // Start from the current leaf so ties never move a vector.
            std::uint32_t best = current[row];
            double nearest = data.distance(std::span<const float>(centers).subspan(std::size_t(best) * d, d), Id(row), d);
            for (std::size_t i = 0; i < probes; ++i) for (auto leaf : by_parent[ranked[i].second]) {
                double candidate = data.distance(std::span<const float>(centers).subspan(std::size_t(leaf) * d, d), Id(row), d);
                if (candidate < nearest || (candidate == nearest && leaf < best)) { nearest = candidate; best = leaf; }
            }
            next[row] = best;
        });
        std::vector<Group> regrouped(leaves.size());
        for (std::size_t row = 0; row < rows; ++row) {
            changed += next[row] != current[row];
            regrouped[next[row]].push_back(Id(row));
        }
        std::vector<Group> kept; std::vector<std::size_t> kept_owners;
        for (std::size_t l = 0; l < regrouped.size(); ++l) if (!regrouped[l].empty()) {
            for (Id id : regrouped[l]) current[id] = std::uint32_t(kept.size());
            kept.push_back(std::move(regrouped[l])); kept_owners.push_back(owners[l]);
        }
        leaves = std::move(kept); owners = std::move(kept_owners);
    }
    return changed;
}
std::vector<Group> external_leaves(Matrix data, std::size_t d, const BuildOptions& o, std::mt19937& rng,
                                   std::vector<Group>& leaves, std::vector<std::size_t>& owners) {
    std::uint32_t label_count = 0;
    for (auto label : o.assignments) label_count = std::max(label_count, label + 1);
    std::vector<Group> labelled(label_count);
    for (std::size_t row = 0; row < o.assignments.size(); ++row) labelled[o.assignments[row]].push_back(Id(row));
    leaves.clear();
    for (auto& group : labelled) if (!group.empty()) leaves.push_back(std::move(group));
    std::vector<float> centers(leaves.size() * d);
    parallel_for(leaves.size(), o.build_threads, [&](std::size_t l, std::size_t) {
        std::vector<double> sum(d, 0);
        for (Id id : leaves[l]) for (std::size_t j = 0; j < d; ++j) sum[j] += data.value(id, j, d);
        for (std::size_t j = 0; j < d; ++j) centers[l * d + j] = float(sum[j] / double(leaves[l].size()));
    });
    Group leaf_ids(leaves.size()); std::iota(leaf_ids.begin(), leaf_ids.end(), Id(0));
    auto parents = cluster(Matrix(std::span<const float>(centers)), d, leaf_ids, std::min(o.coarse, leaves.size()), o, rng);
    owners.assign(leaves.size(), 0);
    std::vector<Group> members(parents.size());
    for (std::size_t p = 0; p < parents.size(); ++p)
        for (Id leaf : parents[p]) {
            owners[leaf] = p;
            members[p].insert(members[p].end(), leaves[leaf].begin(), leaves[leaf].end());
        }
    return members;
}
}
