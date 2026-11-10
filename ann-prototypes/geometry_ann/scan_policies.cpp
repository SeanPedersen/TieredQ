// Spill and stopping simulations. Margins use the exact decoded-member metric of the index.
#include "scan_policies.h"
#include "parallel.h"
#include <algorithm>
#include <queue>
#include <stdexcept>
namespace geometry_ann {
std::vector<Id> nearest_leaves(std::span<const Leaf> leaves, std::size_t count, std::size_t threads) {
    const std::size_t n = leaves.size();
    if (!count || count >= n) throw std::invalid_argument("neighbour count must be within 1..leaves-1");
    std::vector<Id> result(n * count);
    parallel_for(n, threads, [&](std::size_t l, std::size_t) {
        std::vector<std::pair<double, Id>> ranked; ranked.reserve(n - 1);
        for (std::size_t other = 0; other < n; ++other)
            if (other != l) ranked.emplace_back(centroid_l2(leaves[l].geometry.centroid, leaves[other].geometry.centroid), Id(other));
        std::partial_sort(ranked.begin(), ranked.begin() + count, ranked.end());
        for (std::size_t e = 0; e < count; ++e) result[l * count + e] = ranked[e].second;
    });
    return result;
}

SpillPlan build_spill_plan(Matrix packed, std::size_t d, std::span<const Leaf> leaves,
                           std::span<const Id> candidates, std::size_t candidate_count, std::size_t threads) {
    std::size_t rows = 0;
    for (const auto& leaf : leaves) rows = std::max(rows, leaf.offset + leaf.size);
    SpillPlan plan;
    plan.leaf_of_row.resize(rows); plan.alternative.assign(rows, no_spill); plan.margin.assign(rows, std::numeric_limits<float>::infinity());
    parallel_for(leaves.size(), threads, [&](std::size_t l, std::size_t) {
        const auto& leaf = leaves[l];
        for (std::size_t i = leaf.offset; i < leaf.offset + leaf.size; ++i) {
            plan.leaf_of_row[i] = std::uint32_t(l);
            double own = packed.distance(leaf.geometry.centroid, Id(i), d), best = std::numeric_limits<double>::infinity();
            std::uint32_t alternative = no_spill;
            for (std::size_t e = 0; e < candidate_count; ++e) {
                auto other = candidates[l * candidate_count + e];
                double distance = packed.distance(leaves[other].geometry.centroid, Id(i), d);
                if (distance < best) { best = distance; alternative = other; }
            }
            plan.alternative[i] = alternative;
            plan.margin[i] = own > 0 ? float((best - own) / own) : std::numeric_limits<float>::infinity();
        }
    });
    return plan;
}

Spill apply_spill(const SpillPlan& plan, std::span<const Leaf> leaves, double tau) {
    Spill spill; spill.target.assign(plan.alternative.size(), no_spill);
    for (const auto& leaf : leaves) spill.leaf_sizes.push_back(leaf.size);
    std::size_t copies = 0;
    for (std::size_t row = 0; row < plan.alternative.size(); ++row)
        if (plan.alternative[row] != no_spill && plan.margin[row] < tau) {
            spill.target[row] = plan.alternative[row]; ++spill.leaf_sizes[plan.alternative[row]]; ++copies;
        }
    spill.fraction = plan.alternative.empty() ? 0 : double(copies) / double(plan.alternative.size());
    return spill;
}

double spilled_coverage(std::span<const std::size_t> order, const Spill& spill, std::span<const std::uint32_t> leaf_of_row,
                        std::span<const std::size_t> truth_rows, std::size_t budget) {
    std::vector<char> scanned(spill.leaf_sizes.size(), 0);
    std::size_t spent = 0;
    for (auto leaf : order) {
        if (spill.leaf_sizes[leaf] > budget - spent) continue;
        spent += spill.leaf_sizes[leaf]; scanned[leaf] = 1;
    }
    std::size_t found = 0;
    for (auto row : truth_rows)
        found += scanned[leaf_of_row[row]] || (spill.target[row] != no_spill && scanned[spill.target[row]]);
    return double(found) / double(truth_rows.size());
}

AdaptiveScan patience_scan(std::span<const std::size_t> order, std::span<const Leaf> leaves,
                           std::span<const double> member_distances, std::span<const std::size_t> truth_leaves,
                           std::size_t k, std::size_t patience, std::size_t max_vectors) {
    std::priority_queue<double> best;
    std::vector<char> scanned(leaves.size(), 0);
    AdaptiveScan result;
    std::size_t idle = 0;
    for (auto l : order) {
        const auto& leaf = leaves[l];
        if (leaf.size > max_vectors - result.vectors) continue;
        bool improved = false;
        for (std::size_t i = leaf.offset; i < leaf.offset + leaf.size; ++i) {
            double distance = member_distances[i];
            if (best.size() < k) { best.push(distance); improved = true; }
            else if (distance < best.top()) { best.pop(); best.push(distance); improved = true; }
        }
        scanned[l] = 1; result.vectors += leaf.size; ++result.leaves;
        idle = improved ? 0 : idle + 1;
        if (best.size() == k && idle >= patience) break;
    }
    std::size_t found = 0;
    for (auto leaf : truth_leaves) found += scanned[leaf];
    result.coverage = double(found) / double(truth_leaves.size());
    return result;
}
}
