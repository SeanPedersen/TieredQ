// Correctness gates for complete search, conservative geometry, duplicates, and strict budgets.
#include "internal.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace geometry_ann;
namespace geometry_ann { void check_graph(); }
namespace {
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
std::vector<Neighbor> brute(std::span<const float> data, std::span<const float> query, std::size_t k) {
    std::vector<Neighbor> result;
    for (std::size_t i = 0; i < data.size() / query.size(); ++i)
        result.push_back({l2(data.subspan(i * query.size(), query.size()), query), Id(i)});
    std::sort(result.begin(), result.end()); result.resize(std::min(k, result.size())); return result;
}
void check_index(std::span<const float> data, std::size_t d, bool local_kmeans) {
    BuildOptions build; build.coarse = 4; build.children = 4; build.leaf_budget = 16;
    build.target_leaf_size = 8; build.local_kmeans_split = local_kmeans;
    for (auto variant : {Variant::fixed, Variant::radius, Variant::pca}) {
        build.variant = variant; Index index(data, d, build);
        require(index.raw_bytes() >= data.size_bytes(), "packed vectors missing");
        for (std::size_t q = 0; q < std::min<std::size_t>(30, data.size() / d); ++q) {
            std::vector<float> query(data.begin() + q * d, data.begin() + (q + 1) * d);
            if (q % 2) for (auto& value : query) value += 0.37f;
            auto expected = brute(data, query, 10);
            for (auto order : {Ordering::centroid, Ordering::bound, Ordering::variance}) {
                SearchOptions options; options.leaf_scans = index.leaf_count(); options.ordering = order;
                auto found = index.search(query, options);
                require(found.neighbors.size() == expected.size(), "wrong result count");
                for (std::size_t i = 0; i < expected.size(); ++i) {
                    require(found.neighbors[i].id == expected[i].id, "complete traversal differs from brute force");
                    require(found.neighbors[i].distance == expected[i].distance, "wrong exact distance");
                }
                options.routing = Routing::flat;
                auto flat = index.search(query, options);
                require(flat.neighbors.size() == expected.size(), "flat routing lost results");
                for (std::size_t i = 0; i < expected.size(); ++i)
                    require(flat.neighbors[i].id == expected[i].id, "complete flat routing differs from brute force");
                options.routing = Routing::hierarchy;
                options.leaf_scans = 2; options.metadata_evaluations = 7;
                auto limited = index.search(query, options);
                require(limited.leaves_scanned <= 2 && limited.metadata_evaluations <= 7, "budget exceeded");
                options.metadata_evaluations = 0;
                require(index.search(query, options).neighbors.empty(), "zero metadata cap ignored");
                options.vector_scans = 11; options.leaf_scans = index.leaf_count();
                options.metadata_evaluations = 4096; options.routing = Routing::flat;
                require(index.search(query, options).vectors_scanned <= 11, "strict vector cap exceeded");
                options.vector_scans = 0;
                require(index.search(query, options).vectors_scanned == 0, "zero vector cap ignored");
                options.metadata_evaluations = 100; options.leaf_scans = 0;
                require(index.search(query, options).vectors_scanned == 0, "zero leaf cap ignored");
            }
        }
    }
}
}
int main() {
    try {
        constexpr std::size_t dimension = 9, population = 193;
        std::mt19937 rng(77); std::normal_distribution<float> normal;
        std::vector<float> data(population * dimension);
        for (auto& value : data) value = normal(rng);
        check_index(data, dimension, false); check_index(data, dimension, true);
        std::vector<float> duplicates(31 * dimension, 2); check_index(duplicates, dimension, false);
        std::vector<float> singleton(dimension, 0); check_index(singleton, dimension, false);
        Group ids(population); std::iota(ids.begin(), ids.end(), Id(0));
        auto geometry = envelope(data, dimension, ids, 4, 31, rng);
        for (Id id : ids)
            require(lower_bound(row(data, dimension, id), geometry, true) == 0, "envelope excludes a member");
        std::vector<float> known = {1, 2, 3, 4, 5, 6, 7, 8, 9}, zero(9, 0);
        require(l2(known, zero) == 285, "L2 accumulation or tail is incorrect");
        std::vector<float> large = data;
        for (auto& value : large) value *= 1e25f;
        auto large_geometry = envelope(large, dimension, ids, 3, 31, rng);
        require(!large_geometry.norm.empty(), "large finite coordinates lost PCA directions");
        for (float value : large_geometry.directions)
            require(std::isfinite(value), "PCA overflow produced nonfinite metadata");
        for (int test = 0; test < 200; ++test) {
            std::vector<float> query(dimension);
            for (auto& value : query) value = normal(rng) * 20;
            double nearest = brute(data, query, 1).front().distance;
            require(lower_bound(query, geometry, true) <= nearest, "PCA bound exceeds true distance");
            require(lower_bound(query, geometry, false) <= nearest, "radius bound exceeds true distance");
        }
        // A deliberately non-unit stored direction checks the norm divisor.
        Envelope slab; slab.centroid = {0, 0}; slab.radius = 100;
        slab.directions = {2, 0}; slab.norm = {4}; slab.low = {-2}; slab.high = {2};
        std::vector<float> outside = {3, 0};
        double bound = lower_bound(outside, slab, true);
        require(bound > 3.99 && bound <= 4, "incorrect projection metric");
        bool rejected = false;
        try { Index invalid(data, 0, BuildOptions{}); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid dimension accepted");
        check_graph();
        std::cout << "geometry tests passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
