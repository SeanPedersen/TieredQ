// Sampled Lloyd training and best-score adaptive bisection without point-level graphs.
#include "internal.h"
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
    for (std::size_t iteration = 0; iteration < o.iterations; ++iteration) {
        std::vector<double> sums(count * d, 0);
        std::vector<std::size_t> sizes(count, 0);
        for (Id id : sample) {
            auto c = nearest(id); ++sizes[c];
            for (std::size_t j = 0; j < d; ++j) sums[c * d + j] += data.value(id,j,d);
        }
        for (std::size_t c = 0; c < count; ++c) {
            if (!sizes[c]) continue;
            for (std::size_t j = 0; j < d; ++j) centers[c * d + j] = float(sums[c * d + j] / sizes[c]);
        }
    }
    std::vector<Group> groups(count);
    for (Id id : ids) groups[nearest(id)].push_back(id);
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
}
