// Full-population conservative envelopes with sampled covariance-vector PCA iterations.
#include "internal.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
namespace geometry_ann {
namespace {
constexpr std::size_t power_iterations = 8;
constexpr double normalization_floor = 1e-20;
constexpr double rounding_safety_factor = 16;
double rounding_margin(std::size_t dimension) {
    return rounding_safety_factor * std::numeric_limits<double>::epsilon() * double(dimension + 1);
}
double projection_error(std::span<const float> a, std::span<const float> b) {
    double magnitude = 0;
    for (std::size_t j = 0; j < a.size(); ++j) magnitude += std::abs(double(a[j]) * b[j]);
    return rounding_margin(a.size()) * magnitude;
}
double dot(std::span<const float> a, std::span<const float> b) {
    double sum = 0;
    for (std::size_t j = 0; j < a.size(); ++j) sum += double(a[j]) * b[j];
    return sum;
}
}
double l2(std::span<const float> a, std::span<const float> b) {
    constexpr std::size_t accumulation_lanes = 4;
    double lanes[accumulation_lanes] = {};
    std::size_t j = 0;
    for (; j + accumulation_lanes <= a.size(); j += accumulation_lanes)
        for (std::size_t lane = 0; lane < accumulation_lanes; ++lane) {
            double delta = double(a[j + lane]) - b[j + lane]; lanes[lane] += delta * delta;
        }
    double sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for (; j < a.size(); ++j) {
        double delta = double(a[j]) - b[j]; sum += delta * delta;
    }
    return sum;
}
std::span<const float> row(std::span<const float> data, std::size_t d, Id id) {
    return data.subspan(std::size_t(id) * d, d);
}
Envelope centroid_envelope(std::span<const float> data, std::size_t d, const Group& ids) {
    Envelope result;
    std::vector<double> sum(d, 0);
    for (Id id : ids) for (std::size_t j = 0; j < d; ++j) sum[j] += data[id * d + j];
    for (double value : sum) result.centroid.push_back(float(value / ids.size()));
    return result;
}
void add_radius(Envelope& result, std::span<const float> data, std::size_t d, const Group& ids) {
    for (Id id : ids) {
        double distance = l2(row(data, d, id), result.centroid);
        result.radius = std::max(result.radius, distance); result.variance += distance;
    }
    result.variance /= ids.size();
    result.radius = std::nextafter(std::sqrt(result.radius * (1 + rounding_margin(d))),
                                   std::numeric_limits<double>::infinity());
}
void add_projections(Envelope& result, std::span<const float> data, std::size_t d, const Group& ids,
                     std::size_t count, std::size_t sample_size, std::mt19937& rng) {
    if (!count) return;
    result.directions.reserve(std::min(count, d) * d);
    result.low.reserve(count); result.high.reserve(count); result.norm.reserve(count);
    Group sample = ids;
    std::shuffle(sample.begin(), sample.end(), rng);
    sample.resize(std::min(sample_size, sample.size()));
    std::normal_distribution<float> normal;
    for (std::size_t component = 0; component < std::min(count, d); ++component) {
        std::vector<float> direction(d);
        for (auto& value : direction) value = normal(rng);
        auto normalize = [&]() {
            for (std::size_t previous = 0; previous < result.norm.size(); ++previous) {
                auto basis = std::span(result.directions).subspan(previous * d, d);
                double projection = dot(direction, basis) / result.norm[previous];
                for (std::size_t j = 0; j < d; ++j) direction[j] -= float(projection * basis[j]);
            }
            double norm = std::sqrt(dot(direction, direction));
            if (norm < normalization_floor) return false;
            for (auto& value : direction) value = float(value / norm);
            return true;
        };
        if (!normalize()) break;
        bool valid = true;
        for (std::size_t iteration = 0; iteration < power_iterations; ++iteration) {
            std::vector<double> next(d, 0);
            for (Id id : sample) {
                double projection = 0;
                for (std::size_t j = 0; j < d; ++j)
                    projection += (double(data[id * d + j]) - result.centroid[j]) * direction[j];
                for (std::size_t j = 0; j < d; ++j)
                    next[j] += projection * (double(data[id * d + j]) - result.centroid[j]);
            }
            double scale = 0;
            for (double value : next) scale = std::max(scale, std::abs(value));
            if (scale == 0) { valid = false; break; }
            for (std::size_t j = 0; j < d; ++j) direction[j] = float(next[j] / scale);
            if (!normalize()) { valid = false; break; }
        }
        if (!valid) break;
        double low = std::numeric_limits<double>::infinity(), high = -low;
        // Extrema use the final stored direction over ALL members, not just the PCA sample.
        for (Id id : ids) {
            double projection = dot(row(data, d, id), direction);
            double error = projection_error(row(data, d, id), direction);
            low = std::min(low, projection - error); high = std::max(high, projection + error);
        }
        result.low.push_back(low); result.high.push_back(high);
        result.norm.push_back(dot(direction, direction));
        result.directions.insert(result.directions.end(), direction.begin(), direction.end());
    }
    for (std::size_t a = 0; a < result.norm.size(); ++a) {
        double row_sum = 0;
        auto u = std::span(result.directions).subspan(a * d, d);
        for (std::size_t b = 0; b < result.norm.size(); ++b) {
            auto v = std::span(result.directions).subspan(b * d, d);
            row_sum += std::abs(dot(u, v)) + projection_error(u, v);
        }
        result.projection_operator_norm_sq = std::max(result.projection_operator_norm_sq, row_sum);
    }
    result.projection_operator_norm_sq = std::nextafter(result.projection_operator_norm_sq * (1 + rounding_margin(d)),
                                                       std::numeric_limits<double>::infinity());

}
Envelope envelope(std::span<const float> data, std::size_t d, const Group& ids,
                  std::size_t count, std::size_t sample_size, std::mt19937& rng, bool statistics) {
    auto result = centroid_envelope(data, d, ids);
    if (!statistics) return result;
    add_radius(result, data, d, ids);
    add_projections(result, data, d, ids, count, sample_size, rng);
    return result;
}

BoundValues distance_bounds(std::span<const float> q, const Envelope& g, bool projections, double centroid_distance) {
    if (centroid_distance < 0) centroid_distance = l2(q, g.centroid);
    double distance = std::sqrt(centroid_distance / (1 + rounding_margin(q.size())));
    double slack = rounding_margin(q.size()) * (1 + distance + g.radius);
    double gap = std::max(0.0, distance - g.radius - slack);
    double bound = gap * gap;
    if (!projections) return {bound, bound, bound};
    double sum_squared_gaps = 0;
    double radius_bound = bound;
    for (std::size_t axis = 0; axis < g.norm.size(); ++axis) {
        auto direction = std::span(g.directions).subspan(axis * q.size(), q.size());
        double projection = dot(q, direction);
        double projection_slack = projection_error(q, direction) + rounding_margin(q.size()) *
            (1 + std::abs(projection) + std::abs(g.low[axis]) + std::abs(g.high[axis]));
        double separation = std::max({0.0, g.low[axis] - projection - projection_slack,
                                      projection - g.high[axis] - projection_slack});
        sum_squared_gaps += separation * separation;
        // Max of slab distances is valid even when stored directions are not exactly orthogonal.
        bound = std::max(bound, separation * separation / (g.norm[axis] * (1 + rounding_margin(q.size()))));
    }
    double joint = bound;
    // Gershgorin bounds ||U||²; summed slab gaps stay valid for imperfect FP32 orthogonality.
    if (g.projection_operator_norm_sq > 0)
        joint = std::max(joint, sum_squared_gaps / (g.projection_operator_norm_sq * (1 + rounding_margin(q.size()))));
    return {radius_bound, bound, joint};
}
double lower_bound(std::span<const float> q, const Envelope& g, bool projections, double centroid_distance) {
    return distance_bounds(q, g, projections, centroid_distance).joint;
}
void add_shared_ranges(Envelope& leaf, const Envelope& basis, std::span<const float> data, std::size_t d, const Group& ids) {
    for (std::size_t axis = 0; axis < basis.norm.size(); ++axis) {
        auto direction = std::span(basis.directions).subspan(axis * d, d);
        double low = std::numeric_limits<double>::infinity(), high = -low;
        for (Id id : ids) {
            auto point = row(data, d, id);
            double projection = dot(point, direction), error = projection_error(point, direction);
            low = std::min(low, projection - error); high = std::max(high, projection + error);
        }
        leaf.shared_low.push_back(low); leaf.shared_high.push_back(high);
    }
}
SharedQuery shared_query(std::span<const float> query, const Envelope& basis) {
    SharedQuery result;
    for (std::size_t axis = 0; axis < basis.norm.size(); ++axis) {
        auto direction = std::span(basis.directions).subspan(axis * query.size(), query.size());
        double projection = dot(query, direction), error = projection_error(query, direction);
        result.low.push_back(projection - error); result.high.push_back(projection + error);
    }
    return result;
}
double shared_lower_bound(const SharedQuery& query, const Envelope& basis, const Envelope& leaf) {
    double sum = 0, bound = 0;
    for (std::size_t axis = 0; axis < leaf.shared_low.size(); ++axis) {
        double slack = rounding_margin(basis.centroid.size()) *
            (1 + std::abs(query.low[axis]) + std::abs(query.high[axis]) +
             std::abs(leaf.shared_low[axis]) + std::abs(leaf.shared_high[axis]));
        double gap = std::max({0.0, leaf.shared_low[axis] - query.high[axis] - slack,
                                    query.low[axis] - leaf.shared_high[axis] - slack});
        sum += gap * gap;
        bound = std::max(bound, gap * gap / (basis.norm[axis] * (1 + rounding_margin(basis.centroid.size()))));
    }
    if (basis.projection_operator_norm_sq > 0)
        bound = std::max(bound, sum / (basis.projection_operator_norm_sq * (1 + rounding_margin(basis.centroid.size()))));
    return bound;
}

}
