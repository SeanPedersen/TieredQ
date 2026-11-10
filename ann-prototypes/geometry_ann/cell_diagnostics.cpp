// Offline cell-ceiling diagnostic: how many leaves can each convex lower bound rule out?
// Compares the enclosing ball, single and joint member-extreme slabs (nearest-centroid
// directions), the exact convex-hull distance, and the non-convex nearest-member oracle.
// Runs outside timed search; temporary per-thread buffers hold one distance per vector.
#include "cell_bounds.h"
#include "parallel.h"
#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>
namespace geometry_ann {
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t recall_k = 10;
constexpr double hybrid_weight = .5;
double nanoseconds(Clock::time_point start) {
    return double(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}
struct QueryContext {
    std::span<const Leaf> leaves;
    std::vector<std::size_t> profits, truth_leaves;
    std::vector<double> truth_distances, centroid_distances;
    double reference = 0;
};
std::size_t vectors_to_cover(const QueryContext& context, std::span<const std::size_t> order) {
    std::size_t spent = 0, covered = 0;
    for (auto leaf : order) {
        spent += context.leaves[leaf].size; covered += context.profits[leaf];
        if (covered >= recall_k) break;
    }
    return spent;
}
template<class Key>
std::vector<std::size_t> order_by(std::size_t count, Key key) {
    std::vector<std::size_t> order(count); std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return key(a) < key(b); });
    return order;
}
void add_safety(CellScore& score, const QueryContext& context, std::span<const double> bounds) {
    for (std::size_t t = 0; t < context.truth_leaves.size(); ++t) {
        double bound = bounds[context.truth_leaves[t]];
        score.unsafe += bound > context.truth_distances[t];
        if (context.reference > 0) score.mean_truth_bound_ratio += bound / context.reference;
    }
    score.mean_truth_bound_ratio /= double(context.truth_leaves.size());
}
CellScore ranked_score(std::string name, const QueryContext& context, std::span<const double> bounds, double bound_ns) {
    CellScore score; score.name = std::move(name); score.bound_ns_per_leaf = bound_ns;
    const auto n = context.leaves.size();
    const auto& centroid = context.centroid_distances;
    for (std::size_t l = 0; l < n; ++l) if (bounds[l] <= context.reference) {
        ++score.certified_clusters; score.certified_vectors += context.leaves[l].size;
    }
    add_safety(score, context, bounds);
    // Pure order ranks by the bound; centroid distance breaks the many zero-bound ties.
    auto pure = order_by(n, [&](std::size_t l) { return std::pair(bounds[l], centroid[l]); });
    auto hybrid = order_by(n, [&](std::size_t l) {
        return std::pair((1 - hybrid_weight) * centroid[l] + hybrid_weight * bounds[l], centroid[l]); });
    score.pure_vectors_10 = vectors_to_cover(context, pure);
    score.hybrid_vectors_10 = vectors_to_cover(context, hybrid);
    std::vector<std::size_t> rank(n);
    for (std::size_t i = 0; i < n; ++i) rank[pure[i]] = i + 1;
    for (auto leaf : context.truth_leaves) score.mean_truth_rank += double(rank[leaf]);
    score.mean_truth_rank /= double(context.truth_leaves.size());
    return score;
}
enum class HullClass { included, excluded, undecided };
}

CellStudy Index::diagnose_cells(std::span<const float> queries, std::span<const std::vector<Id>> truth,
                                const CellDiagnosticOptions& options) const {
    if (queries.size() != truth.size() * dimension_) throw std::invalid_argument("cell diagnostic query shape");
    if (!options.small_directions || options.small_directions > options.directions)
        throw std::invalid_argument("small cell directions must be within 1..directions");
    const std::size_t n = leaves_.size(), d = dimension_;
    Matrix packed(std::span<const float>(vectors_), {codes_, code_scales_, code_offsets_});
    CellStudy study;
    auto start = Clock::now();
    auto slabs = build_cell_slabs(packed, d, leaves_, options.directions, options.threads);
    study.slab_build_s = nanoseconds(start) * 1e-9; study.slab_bytes = slabs.bytes();
    std::vector<Id> position(ids_.size());
    for (std::size_t i = 0; i < ids_.size(); ++i) position[ids_[i]] = Id(i);
    study.queries.resize(truth.size());
    std::vector<std::vector<double>> member_buffers(std::max<std::size_t>(1, options.threads));
    parallel_for(truth.size(), options.threads, [&](std::size_t qi, std::size_t thread) {
        auto query = queries.subspan(qi * d, d);
        auto& members = member_buffers[thread]; members.resize(ids_.size());
        for (std::size_t i = 0; i < ids_.size(); ++i) members[i] = packed.distance(query, Id(i), d);
        QueryContext context{leaves_, std::vector<std::size_t>(n), {}, {}, std::vector<double>(n), 0};
        for (std::size_t l = 0; l < n; ++l) context.centroid_distances[l] = centroid_l2(query, leaves_[l].geometry.centroid);
        if (truth[qi].size() != recall_k) throw std::invalid_argument("cell diagnostic requires 10 truth IDs");
        for (Id id : truth[qi]) {
            if (id >= position.size()) throw std::invalid_argument("cell diagnostic truth ID");
            auto leaf = leaf_containing(leaves_, position[id]);
            ++context.profits[leaf]; context.truth_leaves.push_back(leaf);
            context.truth_distances.push_back(members[position[id]]);
            context.reference = std::max(context.reference, members[position[id]]);
        }
        auto& row = study.queries[qi];
        row.query = qi; row.reference = context.reference;
        row.oracle_vectors_10 = cover_costs(context.profits, leaves_, recall_k)[recall_k];
        row.truth_clusters = std::size_t(std::count_if(context.profits.begin(), context.profits.end(), [](std::size_t p) { return p > 0; }));

        std::vector<double> zero(n, 0), ball(n), nearest(n, std::numeric_limits<double>::infinity());
        auto ball_start = Clock::now();
        for (std::size_t l = 0; l < n; ++l)
            ball[l] = geometry_metadata_ ? distance_bounds(query, leaves_[l].geometry, false, context.centroid_distances[l]).radius : 0;
        double ball_ns = nanoseconds(ball_start) / double(n);
        for (std::size_t l = 0; l < n; ++l)
            for (std::size_t i = leaves_[l].offset; i < leaves_[l].offset + leaves_[l].size; ++i) nearest[l] = std::min(nearest[l], members[i]);
        row.scores.push_back(ranked_score("centroid", context, zero, 0));
        if (geometry_metadata_) row.scores.push_back(ranked_score("ball", context, ball, ball_ns));
        std::vector<double> widest_joint;
        std::vector<std::size_t> direction_counts{options.small_directions};
        if (options.directions != options.small_directions) direction_counts.push_back(options.directions);
        for (auto used : direction_counts) {
            auto prefix = "knn" + std::to_string(used);
            auto timed_bounds = [&](std::size_t sweeps, bool joint) {
                std::vector<double> values(n);
                auto slab_start = Clock::now();
                for (std::size_t l = 0; l < n; ++l) {
                    auto bounds = slab_bounds(slabs, l, used, context.centroid_distances, d, sweeps);
                    values[l] = std::max(ball[l], joint ? bounds.joint : bounds.single);
                }
                return std::pair(values, nanoseconds(slab_start) / double(n));
            };
            auto [single, single_ns] = timed_bounds(0, false);
            auto [cheap, cheap_ns] = timed_bounds(options.cheap_sweeps, true);
            auto [joint, joint_ns] = timed_bounds(options.converged_sweeps, true);
            row.scores.push_back(ranked_score(prefix + "-single", context, single, single_ns));
            row.scores.push_back(ranked_score(prefix + "-joint" + std::to_string(options.cheap_sweeps), context, cheap, cheap_ns));
            row.scores.push_back(ranked_score(prefix + "-joint", context, joint, joint_ns));
            widest_joint = std::move(joint);
        }
        if (options.hull) {
            CellScore hull; hull.name = "hull"; hull.ranked = false;
            std::vector<double> lower(n);
            auto hull_start = Clock::now();
            for (std::size_t l = 0; l < n; ++l) {
                const auto& leaf = leaves_[l];
                // Any valid outer bound above the reference already excludes the hull.
                if (widest_joint[l] > context.reference) { lower[l] = widest_joint[l]; continue; }
                auto decision = hull_distance(query, packed, d, leaf.offset, leaf.size,
                    std::span(members).subspan(leaf.offset, leaf.size), {context.reference, 0, options.hull_iterations});
                lower[l] = std::max(decision.lower, widest_joint[l]); hull.hull_iterations += decision.iterations;
                auto verdict = decision.upper <= context.reference ? HullClass::included
                             : lower[l] > context.reference ? HullClass::excluded : HullClass::undecided;
                if (verdict == HullClass::included) { ++hull.certified_clusters; hull.certified_vectors += leaf.size; }
                if (verdict == HullClass::undecided) { ++hull.undecided_clusters; hull.undecided_vectors += leaf.size; }
            }
            hull.bound_ns_per_leaf = nanoseconds(hull_start) / double(n);
            add_safety(hull, context, lower);
            row.scores.push_back(hull);
            if (options.hull_rank) {
                // Rank candidate hulls by an estimated distance; excluded hulls keep their (larger) lower bound.
                std::vector<double> estimate(lower);
                auto rank_start = Clock::now();
                for (std::size_t l = 0; l < n; ++l) {
                    if (lower[l] > context.reference) continue;
                    const auto& leaf = leaves_[l];
                    estimate[l] = hull_distance(query, packed, d, leaf.offset, leaf.size,
                        std::span(members).subspan(leaf.offset, leaf.size),
                        {std::nullopt, options.rank_gap, options.rank_iterations}).upper;
                }
                row.scores.push_back(ranked_score("hull-rank", context, estimate, nanoseconds(rank_start) / double(n)));
            }
        }
        row.scores.push_back(ranked_score("nearest-member", context, nearest, 0));
    });
    return study;
}
}
