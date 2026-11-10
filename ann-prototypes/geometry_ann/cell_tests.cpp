// Correctness gates for joint slab bounds, hull distances, and the cell-ceiling diagnostic.
#include "cell_bounds.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace geometry_ann;
namespace {
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }

// Box [-1,1]² around the origin, slab normals toward neighbor centroids (1,0) and (0,1).
void check_joint_reaches_polytope_corner() {
    CellSlabs slabs; slabs.directions = 2;
    slabs.neighbors = {1, 2, 0, 0, 0, 0}; slabs.low = {-1, -1, 0, 0, 0, 0}; slabs.high = {1, 1, 0, 0, 0, 0};
    slabs.gram = {1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    // Query (3,3): distances to centroids (0,0), (1,0), (0,1).
    std::vector<double> centroid_distances = {18, 13, 13};
    auto bounds = slab_bounds(slabs, 0, 2, centroid_distances, 2, 256);
    require(std::abs(bounds.single - 4) < 1e-9, "single slab bound should equal the one-axis gap");
    require(bounds.joint <= 8 && bounds.joint > 8 - 1e-6, "joint bound should reach the box corner distance");
    require(slab_bounds(slabs, 0, 1, centroid_distances, 2, 256).joint <= 4 + 1e-9, "one direction cannot exceed its slab");
}

void check_hull_distance_on_square() {
    std::vector<float> square = {0, 0, 1, 0, 0, 1, 1, 1};
    std::vector<float> query = {3, .5f};
    std::vector<double> members;
    for (std::size_t i = 0; i < 4; ++i) members.push_back(l2(query, std::span(square).subspan(i * 2, 2)));
    // Squared face distance is 4; thresholds just either side must be certified correctly.
    constexpr double margin = 1e-7;
    auto below = hull_distance(query, Matrix(square), 2, 0, 4, members, {4 - margin, 0, 1000});
    require(below.lower > 4 - margin && below.lower <= 4 + 1e-12, "hull must be excluded just below its distance");
    auto above = hull_distance(query, Matrix(square), 2, 0, 4, members, {4 + margin, 0, 1000});
    require(above.upper <= 4 + margin && above.lower <= 4 + 1e-12, "hull must be included just above its distance");
    auto inside = hull_distance(std::vector<float>{.5f, .5f}, Matrix(square), 2, 0, 4,
        std::vector<double>(4, .5), {.25, 0, 1000});
    require(inside.lower <= 1e-12 && inside.upper <= .25, "interior query must be certified within the threshold");
    auto converged = hull_distance(query, Matrix(square), 2, 0, 4, members, {std::nullopt, 1e-9, 1000});
    require(converged.lower <= 4 + 1e-12 && converged.upper >= 4 - 1e-12 && converged.upper - converged.lower <= 1e-6,
            "unthresholded hull estimate must converge to the face distance");
}

std::vector<std::vector<Id>> brute_truth(std::span<const float> base, std::span<const float> queries, std::size_t d) {
    std::vector<std::vector<Id>> truth;
    for (std::size_t q = 0; q < queries.size() / d; ++q) {
        std::vector<Neighbor> all;
        for (std::size_t i = 0; i < base.size() / d; ++i)
            all.push_back({l2(base.subspan(i * d, d), queries.subspan(q * d, d)), Id(i)});
        std::partial_sort(all.begin(), all.begin() + 10, all.end());
        std::vector<Id> ids; for (std::size_t i = 0; i < 10; ++i) ids.push_back(all[i].id);
        truth.push_back(ids);
    }
    return truth;
}

void check_diagnostic(std::span<const float> base, std::span<const float> queries, std::size_t d, QuantizedInput quantized) {
    BuildOptions build; build.coarse = 4; build.children = 8; build.leaf_budget = 32; build.geometry_metadata = true;
    Index index(base, d, build, quantized);
    std::vector<float> decoded;
    if (!quantized.codes.empty()) {
        Matrix codes(std::span<const float>{}, quantized);
        for (std::size_t i = 0; i < quantized.codes.size() / d; ++i) for (std::size_t j = 0; j < d; ++j) decoded.push_back(codes.value(Id(i), j, d));
    }
    auto truth = brute_truth(decoded.empty() ? base : std::span<const float>(decoded), queries, d);
    CellDiagnosticOptions options; options.directions = 12; options.small_directions = 4; options.threads = 3; options.hull_iterations = 200;
    auto study = index.diagnose_cells(queries, truth, options);
    require(study.queries.size() == truth.size(), "missing diagnostic queries");
    for (const auto& row : study.queries) {
        auto find = [&](const std::string& name) {
            auto it = std::find_if(row.scores.begin(), row.scores.end(), [&](const CellScore& s) { return s.name == name; });
            require(it != row.scores.end(), "missing cell score"); return *it;
        };
        for (const auto& score : row.scores) {
            require(score.unsafe == 0, "cell lower bound exceeds a true-neighbor distance");
            if (score.ranked) require(score.pure_vectors_10 >= row.oracle_vectors_10 && score.hybrid_vectors_10 >= row.oracle_vectors_10, "ranking beats the oracle");
        }
        auto centroid = find("centroid"), ball = find("ball"), single = find("knn12-single"), joint = find("knn12-joint");
        auto hull = find("hull"), nearest = find("nearest-member"), hull_rank = find("hull-rank");
        require(hull_rank.certified_clusters >= nearest.certified_clusters, "hull estimates cannot exceed nearest members");
        require(centroid.certified_clusters == index.leaf_count(), "centroid ordering certifies every leaf");
        require(ball.certified_clusters <= centroid.certified_clusters && single.certified_clusters <= ball.certified_clusters, "slabs must refine the ball");
        require(joint.certified_clusters <= single.certified_clusters, "joint bound must refine single slabs");
        require(find("knn12-joint4").certified_clusters <= single.certified_clusters, "cheap joint bound must refine single slabs");
        require(find("knn4-joint").certified_clusters >= joint.certified_clusters, "more directions must not loosen the joint bound");
        require(hull.certified_clusters + hull.undecided_clusters <= joint.certified_clusters, "hull must refine the slab polytope");
        require(nearest.certified_clusters <= hull.certified_clusters + hull.undecided_clusters, "nearest member must refine the hull");
        require(nearest.certified_clusters >= row.truth_clusters, "every truth cluster has a member within the reference");
    }
}
}
int main() {
    try {
        check_joint_reaches_polytope_corner();
        check_hull_distance_on_square();
        constexpr std::size_t d = 24, population = 1600, query_count = 12;
        std::mt19937 rng(91); std::normal_distribution<float> normal;
        std::vector<float> base(population * d), queries(query_count * d);
        for (std::size_t i = 0; i < population; ++i) {
            float center = float(i % 7) * 2.5f;
            for (std::size_t j = 0; j < d; ++j) base[i * d + j] = normal(rng) * (j % 3 ? .4f : 1.5f) + (j % 7 == i % 7 ? center : 0);
        }
        for (auto& value : queries) value = normal(rng) * 2;
        check_diagnostic(base, queries, d, {});
        std::vector<std::int8_t> codes(population * d); std::vector<float> scales(d), offsets(d);
        for (std::size_t j = 0; j < d; ++j) { scales[j] = .03f + float(j) * .001f; offsets[j] = normal(rng) * .1f; }
        for (std::size_t i = 0; i < population * d; ++i)
            codes[i] = std::int8_t(std::clamp(std::lround((base[i] - offsets[i % d]) / scales[i % d]), -128L, 127L));
        check_diagnostic({}, queries, d, {codes, scales, offsets});
        std::cout << "cell tests passed\n";
    } catch (const std::exception& error) { std::cerr << "cell test failed: " << error.what() << '\n'; return 1; }
}
