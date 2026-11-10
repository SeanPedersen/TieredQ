// Extreme-value ranking model. For n members with projection spread s along the query
// direction u, min_x ||q-x||² = D + ||x-c||² - 2√D (x-c)·u is approximated by
// D + r̄² - 2√D · s · √(2 ln n): the expected maximum of n Gaussian projections.
#include "rank_scores.h"
#include "parallel.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
namespace geometry_ann {
namespace {
constexpr double orthogonality_floor = 1e-12;
std::vector<double> orthonormal_basis(Matrix packed, std::size_t d, std::size_t rows, std::size_t axes,
                                      std::size_t sample, std::uint32_t seed) {
    std::mt19937 rng(seed);
    Group ids(rows); std::iota(ids.begin(), ids.end(), Id(0));
    std::shuffle(ids.begin(), ids.end(), rng); ids.resize(std::min(ids.size(), sample));
    auto pca = envelope(packed, d, ids, axes, sample, rng, true);
    std::vector<double> basis;
    for (std::size_t k = 0; k < pca.directions.size() / d; ++k) {
        std::vector<double> v(pca.directions.begin() + k * d, pca.directions.begin() + (k + 1) * d);
        for (std::size_t previous = 0; previous < basis.size() / d; ++previous) {
            double projection = 0;
            for (std::size_t j = 0; j < d; ++j) projection += v[j] * basis[previous * d + j];
            for (std::size_t j = 0; j < d; ++j) v[j] -= projection * basis[previous * d + j];
        }
        double norm = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
        if (norm < orthogonality_floor) continue;
        for (double value : v) basis.push_back(value / norm);
    }
    return basis;
}
double project(std::span<const double> axis, std::span<const float> x) {
    double sum = 0;
    for (std::size_t j = 0; j < x.size(); ++j) sum += axis[j] * double(x[j]);
    return sum;
}
}

std::size_t SpreadModel::bytes() const {
    return (basis.capacity() + center_projection.capacity() + axis_variance.capacity() +
            mean_radius_sq.capacity() + log_size.capacity()) * sizeof(double);
}

SpreadModel build_spread_model(Matrix packed, std::size_t d, std::span<const Leaf> leaves, std::size_t axes,
                               std::size_t sample, std::uint32_t seed, std::size_t threads) {
    if (axes >= d) throw std::invalid_argument("spread axes must be below the dimension");
    SpreadModel model; model.dimension = d;
    std::size_t rows = 0;
    for (const auto& leaf : leaves) rows = std::max(rows, leaf.offset + leaf.size);
    if (axes) model.basis = orthonormal_basis(packed, d, rows, axes, sample, seed);
    model.axes = model.basis.size() / d;
    const std::size_t n = leaves.size(), r = model.axes;
    model.center_projection.resize(n * r); model.axis_variance.resize(n * r);
    model.mean_radius_sq.resize(n); model.log_size.resize(n);
    parallel_for(n, threads, [&](std::size_t l, std::size_t) {
        const auto& leaf = leaves[l];
        std::span<const float> center = leaf.geometry.centroid;
        std::vector<double> centered_axis(r);
        for (std::size_t k = 0; k < r; ++k) {
            centered_axis[k] = project(std::span(model.basis).subspan(k * d, d), center);
            model.center_projection[l * r + k] = centered_axis[k];
        }
        std::vector<float> point(d);
        double radius_sum = 0;
        for (std::size_t i = leaf.offset; i < leaf.offset + leaf.size; ++i) {
            for (std::size_t j = 0; j < d; ++j) point[j] = packed.value(Id(i), j, d);
            radius_sum += centroid_l2(point, center);
            for (std::size_t k = 0; k < r; ++k) {
                double offset = project(std::span(model.basis).subspan(k * d, d), point) - centered_axis[k];
                model.axis_variance[l * r + k] += offset * offset;
            }
        }
        double size = double(leaf.size);
        for (std::size_t k = 0; k < r; ++k) model.axis_variance[l * r + k] /= size;
        model.mean_radius_sq[l] = radius_sum / size;
        model.log_size[l] = std::log(size);
    });
    return model;
}

std::vector<double> project_query(const SpreadModel& model, std::span<const float> query) {
    std::vector<double> result(model.axes);
    for (std::size_t k = 0; k < model.axes; ++k)
        result[k] = project(std::span(model.basis).subspan(k * model.dimension, model.dimension), query);
    return result;
}

double expected_nearest(const SpreadModel& model, std::size_t leaf, double centroid_distance,
                        std::span<const double> query_projection, double beta, bool directional) {
    const double radius_sq = model.mean_radius_sq[leaf];
    double spread_sq = radius_sq / double(model.dimension);
    if (directional && model.axes && centroid_distance > 0) {
        const std::size_t r = model.axes;
        double captured = 0, explained = 0, along = 0;
        for (std::size_t k = 0; k < r; ++k) {
            double cosine = (query_projection[k] - model.center_projection[leaf * r + k]) / std::sqrt(centroid_distance);
            double variance = model.axis_variance[leaf * r + k];
            captured += cosine * cosine; explained += variance; along += variance * cosine * cosine;
        }
        double residual = std::max(0.0, radius_sq - explained) / double(model.dimension - r);
        spread_sq = along + residual * std::max(0.0, 1 - captured);
    }
    double extreme = std::sqrt(2 * model.log_size[leaf]);
    return centroid_distance + radius_sq - 2 * beta * std::sqrt(centroid_distance * spread_sq) * extreme;
}
}
