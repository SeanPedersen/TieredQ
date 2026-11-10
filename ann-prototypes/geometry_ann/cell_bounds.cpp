// Polytope and convex-hull lower bounds for leaf clusters (offline cell-ceiling study).
//
// Joint slab bound: for any weights w, u = Σ w_e a_e satisfies
//   ||x-q||·||u|| >= u·(x-q) >= S(w) = Σ_e min(w_e·lo_e, w_e·hi_e),
// where [lo_e, hi_e] encloses (x-q)·a_e. Hence ||x-q||² >= S(w)²/(wᵀGw) for every w,
// so an unconverged optimizer still yields a valid bound. Coordinate ascent on the
// concave dual φ(w) = 2S(w) - wᵀGw reaches the exact distance from q to the slab polytope.
#include "cell_bounds.h"
#include "parallel.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define CELL_X86 1
#endif
namespace geometry_ann {
namespace {
constexpr std::size_t max_directions = 64;
constexpr std::size_t certify_interval = 4;
constexpr double sweep_tolerance = 1e-9;
constexpr double infinity = std::numeric_limits<double>::infinity();
double error_factor(std::size_t d) { return 32 * std::numeric_limits<double>::epsilon() * double(d + 8); }

void decode(Matrix data, std::size_t d, std::size_t row, std::span<float> out) {
    for (std::size_t j = 0; j < d; ++j) out[j] = data.value(Id(row), j, d);
}
double scalar_code_dot(const double* v, const std::int8_t* code, std::size_t n) {
    double sum = 0;
    for (std::size_t j = 0; j < n; ++j) sum += v[j] * double(code[j]);
    return sum;
}
#ifdef CELL_X86
__attribute__((target("avx2")))
double avx2_code_dot(const double* v, const std::int8_t* code, std::size_t n) {
    auto even = _mm256_setzero_pd(), odd = even; std::size_t j = 0;
    for (; j + 8 <= n; j += 8) {
        std::int32_t low, high; std::memcpy(&low, code + j, 4); std::memcpy(&high, code + j + 4, 4);
        auto a = _mm256_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_cvtsi32_si128(low)));
        auto b = _mm256_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_cvtsi32_si128(high)));
        even = _mm256_add_pd(even, _mm256_mul_pd(a, _mm256_loadu_pd(v + j)));
        odd = _mm256_add_pd(odd, _mm256_mul_pd(b, _mm256_loadu_pd(v + j + 4)));
    }
    double lanes[4]; _mm256_storeu_pd(lanes, _mm256_add_pd(even, odd));
    return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]) + scalar_code_dot(v + j, code + j, n - j);
}
#endif
double code_dot(const double* v, const std::int8_t* code, std::size_t n) {
#ifdef CELL_X86
    static const bool available = __builtin_cpu_supports("avx2");
    if (available) return avx2_code_dot(v, code, n);
#endif
    return scalar_code_dot(v, code, n);
}
// Dot products of one fixed direction v with many packed rows, decoding nothing.
class RowDot {
public:
    RowDot(Matrix data, std::size_t d, std::span<const double> v) : data_(data), d_(d), v_(v) {
        if (data.quantized.codes.empty()) return;
        scaled_.resize(d);
        for (std::size_t j = 0; j < d; ++j) {
            scaled_[j] = v[j] * double(data.quantized.scales[j]);
            if (!data.quantized.offsets.empty()) constant_ += v[j] * double(data.quantized.offsets[j]);
        }
    }
    double operator()(std::size_t row) const {
        if (!scaled_.empty()) return code_dot(scaled_.data(), data_.quantized.codes.data() + row * d_, d_) + constant_;
        const float* x = data_.floats.data() + row * d_; double sum = 0;
        for (std::size_t j = 0; j < d_; ++j) sum += v_[j] * double(x[j]);
        return sum;
    }
private:
    Matrix data_; std::size_t d_; std::span<const double> v_;
    std::vector<double> scaled_; double constant_ = 0;
};
double dot(std::span<const double> a, std::span<const double> b) {
    double sum = 0;
    for (std::size_t j = 0; j < a.size(); ++j) sum += a[j] * b[j];
    return sum;
}
}

std::size_t CellSlabs::bytes() const {
    return neighbors.capacity() * sizeof(Id) + (low.capacity() + high.capacity() + gram.capacity()) * sizeof(double);
}

CellSlabs build_cell_slabs(Matrix packed, std::size_t d, std::span<const Leaf> leaves,
                           std::size_t directions, std::size_t threads) {
    const std::size_t n = leaves.size();
    if (!directions || directions >= n || directions > max_directions)
        throw std::invalid_argument("cell directions must be within 1..min(leaves-1, 64)");
    CellSlabs slabs; slabs.directions = directions;
    slabs.neighbors.resize(n * directions); slabs.low.resize(n * directions); slabs.high.resize(n * directions);
    slabs.gram.resize(n * directions * directions);
    const double eps = error_factor(d);
    parallel_for(n, threads, [&](std::size_t l, std::size_t) {
        const auto& center = leaves[l].geometry.centroid;
        std::vector<std::pair<double, Id>> ranked; ranked.reserve(n - 1);
        for (std::size_t other = 0; other < n; ++other)
            if (other != l) ranked.emplace_back(centroid_l2(center, leaves[other].geometry.centroid), Id(other));
        std::partial_sort(ranked.begin(), ranked.begin() + directions, ranked.end());
        const std::size_t base = l * directions;
        std::vector<double> axes(directions * d);
        for (std::size_t e = 0; e < directions; ++e) {
            slabs.neighbors[base + e] = ranked[e].second;
            const auto& target = leaves[ranked[e].second].geometry.centroid;
            for (std::size_t j = 0; j < d; ++j) axes[e * d + j] = double(target[j]) - double(center[j]);
        }
        for (std::size_t e = 0; e < directions; ++e) for (std::size_t f = 0; f <= e; ++f) {
            double value = dot(std::span(axes).subspan(e * d, d), std::span(axes).subspan(f * d, d));
            slabs.gram[(base + e) * directions + f] = slabs.gram[(base + f) * directions + e] = value;
        }
        std::vector<double> low(directions, infinity), high(directions, -infinity);
        std::vector<float> point(d);
        for (std::size_t i = leaves[l].offset; i < leaves[l].offset + leaves[l].size; ++i) {
            decode(packed, d, i, point);
            for (std::size_t e = 0; e < directions; ++e) {
                const double* axis = axes.data() + e * d;
                double projection = 0, magnitude = 0;
                for (std::size_t j = 0; j < d; ++j) {
                    projection += (double(point[j]) - double(center[j])) * axis[j];
                    magnitude += (std::abs(double(point[j])) + std::abs(double(center[j]))) * std::abs(axis[j]);
                }
                double error = eps * magnitude;
                low[e] = std::min(low[e], std::nextafter(projection - error, -infinity));
                high[e] = std::max(high[e], std::nextafter(projection + error, infinity));
            }
        }
        std::copy(low.begin(), low.end(), slabs.low.begin() + base);
        std::copy(high.begin(), high.end(), slabs.high.begin() + base);
    });
    return slabs;
}

SlabBounds slab_bounds(const CellSlabs& slabs, std::size_t leaf, std::size_t used,
                       std::span<const double> centroid_distances, std::size_t d, std::size_t sweeps) {
    if (!used || used > slabs.directions) throw std::invalid_argument("slab directions exceed stored directions");
    const std::size_t stride = slabs.directions, base = leaf * stride;
    const double eps = error_factor(d), own = centroid_distances[leaf];
    auto gram = [&](std::size_t e, std::size_t f) { return slabs.gram[(base + e) * stride + f]; };
    std::array<double, max_directions> lo{}, hi{}, length{}, w{}, g{};
    std::array<bool, max_directions> active{};
    SlabBounds result;
    bool separated = false;
    for (std::size_t e = 0; e < used; ++e) {
        double length_squared = gram(e, e);
        if (!(length_squared > 0)) continue;
        // 2(q-c_l)·a_e = ||q-c_l||² + ||a_e||² - ||q-c_e||².
        double other = centroid_distances[slabs.neighbors[base + e]];
        double projection = (own + length_squared - other) * .5;
        double low = slabs.low[base + e], high = slabs.high[base + e];
        double slack = eps * (own + other + length_squared + std::abs(low) + std::abs(high));
        active[e] = true; length[e] = std::sqrt(length_squared);
        lo[e] = low - projection - slack; hi[e] = high - projection + slack;
        double gap = std::max({0.0, lo[e], -hi[e]});
        separated |= gap > 0;
        result.single = std::max(result.single, gap * gap / (length_squared * (1 + eps)));
    }
    result.joint = result.single;
    if (!separated || !sweeps) return result;
    auto certified = [&]() {
        // Recompute Gw from scratch so update drift cannot weaken the guarantee.
        double support = 0, support_error = 0, quadratic = 0, weight_norm = 0;
        for (std::size_t e = 0; e < used; ++e) {
            if (!w[e]) continue;
            double row = 0;
            for (std::size_t f = 0; f < used; ++f) if (w[f]) row += gram(e, f) * w[f];
            quadratic += w[e] * row;
            double bound = w[e] > 0 ? lo[e] : hi[e];
            support += w[e] * bound; support_error += std::abs(w[e] * bound);
            weight_norm += std::abs(w[e]) * length[e];
        }
        support -= eps * support_error;
        quadratic += 2 * eps * weight_norm * weight_norm;
        return support > 0 && quadratic > 0 ? support * support / quadratic : 0.0;
    };
    double previous = 0;
    for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
        for (std::size_t e = 0; e < used; ++e) {
            if (!active[e]) continue;
            double diagonal = gram(e, e), others = g[e] - diagonal * w[e];
            double next = lo[e] - others > 0 ? (lo[e] - others) / diagonal
                        : hi[e] - others < 0 ? (hi[e] - others) / diagonal : 0.0;
            double delta = next - w[e];
            if (delta == 0) continue;
            w[e] = next;
            for (std::size_t f = 0; f < used; ++f) g[f] += gram(f, e) * delta;
        }
        bool last = sweep + 1 == sweeps;
        if (!last && (sweep + 1) % certify_interval) continue;
        double value = certified();
        result.joint = std::max(result.joint, value);
        if (value - previous <= sweep_tolerance * value) break;
        previous = value;
    }
    return result;
}

HullDecision hull_distance(std::span<const float> query, Matrix packed, std::size_t d,
                           std::size_t offset, std::size_t size, std::span<const double> member_distances,
                           const HullStop& stop) {
    if (!size || member_distances.size() != size) throw std::invalid_argument("hull member shape");
    HullDecision result;
    auto nearest = std::size_t(std::min_element(member_distances.begin(), member_distances.end()) - member_distances.begin());
    result.upper = member_distances[nearest];
    auto decided = [&]() {
        if (stop.threshold && (result.upper <= *stop.threshold || result.lower > *stop.threshold)) return true;
        return result.upper - result.lower <= stop.relative_gap * result.upper;
    };
    if (decided()) return result;
    std::vector<float> point(d);
    std::vector<double> v(d), y(d);
    decode(packed, d, offset + nearest, point);
    for (std::size_t j = 0; j < d; ++j) v[j] = double(point[j]) - double(query[j]);
    double vv = dot(v, v);
    for (std::size_t iteration = 0; iteration < stop.max_iterations; ++iteration) {
        result.iterations = iteration + 1;
        double qv = 0;
        for (std::size_t j = 0; j < d; ++j) qv += double(query[j]) * v[j];
        RowDot along(packed, d, v);
        double minimum = infinity; std::size_t star = 0;
        for (std::size_t i = 0; i < size; ++i) {
            double value = along(offset + i);
            if (value < minimum) { minimum = value; star = i; }
        }
        // Separating hyperplane: every hull point y satisfies y·v >= support.
        double support = minimum - qv;
        if (support > 0) result.lower = std::max(result.lower, support * support / vv);
        if (decided()) return result;
        decode(packed, d, offset + star, point);
        for (std::size_t j = 0; j < d; ++j) y[j] = double(point[j]) - double(query[j]);
        double vy = dot(v, y), yy = dot(y, y), denominator = vv - 2 * vy + yy;
        double step = denominator > 0 ? (vv - vy) / denominator : 0;
        if (step <= 0) break;
        step = std::min(step, 1.0);
        for (std::size_t j = 0; j < d; ++j) v[j] += step * (y[j] - v[j]);
        vv = dot(v, v);
        result.upper = std::min(result.upper, vv);
        if (decided()) return result;
    }
    return result;
}
}
