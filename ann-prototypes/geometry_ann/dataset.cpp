// Strict fvecs/ivecs readers and deterministic clustered synthetic data for smoke benchmarks.
#include "dataset.h"
#include "internal.h"
#include <algorithm>
#include <bit>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
namespace geometry_ann {
namespace {
template<class T> std::vector<T> read_records(const std::string& path, std::size_t& dimension,
                                             std::size_t limit = std::numeric_limits<std::size_t>::max()) {
    if constexpr (std::endian::native != std::endian::little)
        throw std::runtime_error("fvecs/ivecs reader requires little endian host");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path);
    std::vector<T> values;
    std::size_t records = 0;
    while (records < limit) {
        std::int32_t width = 0;
        input.read(reinterpret_cast<char*>(&width), sizeof(width));
        if (input.eof() && input.gcount() == 0) break;
        constexpr std::int32_t maximum_dimension = 1 << 20;
        if (!input || width <= 0 || width > maximum_dimension) throw std::runtime_error("invalid record in " + path);
        if (!dimension) dimension = std::size_t(width);
        if (dimension != std::size_t(width)) throw std::runtime_error("dimension mismatch in " + path);
        auto start = values.size(); values.resize(start + dimension);
        input.read(reinterpret_cast<char*>(values.data() + start), std::streamsize(dimension * sizeof(T)));
        if (!input) throw std::runtime_error("truncated record in " + path);
        ++records;
    }
    if (values.empty()) throw std::runtime_error("empty dataset " + path);
    return values;
}
}
std::vector<Neighbor> exact(std::span<const float> base, std::span<const float> q, std::size_t k) {
    std::vector<Neighbor> neighbors;
    for (std::size_t i = 0; i < base.size() / q.size(); ++i)
        neighbors.push_back({l2(base.subspan(i * q.size(), q.size()), q), Id(i)});
    k = std::min(k, neighbors.size());
    std::partial_sort(neighbors.begin(), neighbors.begin() + k, neighbors.end()); neighbors.resize(k);
    return neighbors;
}
Dataset synthetic(std::size_t count, std::size_t query_count, std::size_t d, std::uint32_t seed) {
    constexpr std::size_t clusters = 24;
    constexpr float center_scale = 4, elongated_scale = 5;
    if (!count || !query_count || !d || count > std::numeric_limits<Id>::max())
        throw std::invalid_argument("invalid synthetic shape");
    Dataset data; data.dimension = d;
    std::mt19937 rng(seed); std::normal_distribution<float> normal;
    std::vector<float> centers(clusters * d);
    for (auto& value : centers) value = normal(rng) * center_scale;
    auto generate = [&](std::size_t n, std::vector<float>& destination) {
        destination.resize(n * d);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t cluster_id = rng() % clusters;
            for (std::size_t j = 0; j < d; ++j)
                destination[i * d + j] = centers[cluster_id * d + j] + normal(rng) *
                    (j == cluster_id % d ? elongated_scale : 1);
        }
    };
    generate(count, data.base); generate(query_count, data.queries);
    for (std::size_t i = 0; i < query_count; ++i) {
        std::vector<Id> ids;
        for (auto neighbor : exact(data.base, std::span(data.queries).subspan(i * d, d), 10)) ids.push_back(neighbor.id);
        data.truth.push_back(std::move(ids));
    }
    return data;
}
Dataset load_dataset(const std::string& base, const std::string& queries, const std::string& truth, std::size_t limit) {
    Dataset data;
    data.base = read_records<float>(base, data.dimension);
    data.queries = read_records<float>(queries, data.dimension, limit);
    std::size_t width = 0;
    auto ids = read_records<Id>(truth, width, limit);
    auto query_count = data.queries.size() / data.dimension;
    if (ids.size() / width != query_count || width < 10 || data.base.size() / data.dimension < 10)
        throw std::runtime_error("ground truth must match queries and include at least ten neighbors");
    for (std::size_t i = 0; i < query_count; ++i) {
        data.truth.emplace_back(ids.begin() + i * width, ids.begin() + i * width + 10);
        for (auto id : data.truth.back()) if (id >= data.base.size() / data.dimension)
            throw std::runtime_error("ground truth ID outside base dataset");
    }
    return data;
}
}
