// Router head-to-head over leaf centroids: which lists does each router hand to the scan, and at what cost?
// Quality: fraction of each query's true top-10 neighbours whose list is among the router's first L lists.
// Routers: exact fp32 scan, full compressed scan, two-level compressed (TieredQ IVF over centroids),
// graph walk with fp32 distances, graph walk with compressed distances (gathered or neighbour-inline codes).
// Every router re-ranks its first `rerank` lists exactly in fp32 before reporting the order.
#include <tieredq/tieredq.h>
#include <cblas.h>
#include <omp.h>
#include <immintrin.h>
#include <sched.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t dim = 1024, truth_k = 10, degree = 16, knn = 48, rerank = 256;
std::size_t supers = 256;  // set from the centroid count: about 128 centroids per super

template<class T> std::vector<T> read_raw(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<T> out(std::size_t(in.tellg()) / sizeof(T)); in.seekg(0);
    in.read(reinterpret_cast<char*>(out.data()), out.size() * sizeof(T)); return out;
}
// fvecs / ivecs: each row is a 4-byte count followed by that many values.
template<class T> std::vector<T> read_vecs(const std::string& path, std::size_t keep, std::size_t rows) {
    auto raw = read_raw<std::int32_t>(path); std::vector<T> out;
    for (std::size_t i = 0, r = 0; i < raw.size() && r < rows; i += std::size_t(raw[i]) + 1, ++r) {
        if (std::size_t(raw[i]) < keep) throw std::runtime_error("rows too narrow in " + path);
        const T* row = reinterpret_cast<const T*>(raw.data() + i + 1); out.insert(out.end(), row, row + keep);
    }
    return out;
}
float l2(const float* a, const float* b) {
#ifdef __AVX512F__
    __m512 wide = _mm512_setzero_ps();
    for (std::size_t j = 0; j < dim; j += 16) { __m512 d = _mm512_sub_ps(_mm512_loadu_ps(a + j), _mm512_loadu_ps(b + j)); wide = _mm512_fmadd_ps(d, d, wide); }
    return _mm512_reduce_add_ps(wide);
#endif
    __m256 acc = _mm256_setzero_ps();
    for (std::size_t j = 0; j < dim; j += 8) { __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j)); acc = _mm256_fmadd_ps(d, d, acc); }
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s); return _mm_cvtss_f32(s);
}
using Ranked = std::vector<std::pair<float, std::uint32_t>>;  // (distance, list), ascending

struct Centroids {
    std::vector<float> rows; std::size_t count;
    const float* row(std::size_t i) const { return rows.data() + i * dim; }
};
// Re-rank the first `head` entries of an approximate order exactly; the tail keeps the router's order.
void rerank_head(const Centroids& c, const float* q, Ranked& order) {
    std::size_t head = std::min(rerank, order.size());
    for (std::size_t i = 0; i < head; ++i) order[i].first = l2(q, c.row(order[i].second));
    std::sort(order.begin(), order.begin() + head);
}

// Exact 48-NN by GEMM, then relative-neighbourhood pruning to `degree`, then reverse edges fill free slots.
std::vector<std::uint32_t> build_graph(const Centroids& c) {
    const std::size_t n = c.count, block = 1024;
    std::vector<float> norms(n); for (std::size_t i = 0; i < n; ++i) norms[i] = cblas_sdot(dim, c.row(i), 1, c.row(i), 1);
    std::vector<std::vector<std::pair<float, std::uint32_t>>> near(n);
    std::vector<float> dots(block * n);
    for (std::size_t b = 0; b < n; b += block) {
        std::size_t m = std::min(block, n - b);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, dim, 1.f, c.row(b), dim, c.rows.data(), dim, 0.f, dots.data(), n);
        #pragma omp parallel for schedule(dynamic, 16)
        for (std::size_t i = 0; i < m; ++i) {
            std::vector<std::pair<float, std::uint32_t>> all; all.reserve(n);
            for (std::size_t j = 0; j < n; ++j) if (j != b + i) all.emplace_back(norms[b + i] + norms[j] - 2 * dots[i * n + j], std::uint32_t(j));
            std::partial_sort(all.begin(), all.begin() + knn, all.end()); near[b + i].assign(all.begin(), all.begin() + knn);  // a copy, so the n-wide scratch is not retained
        }
    }
    std::vector<std::vector<std::uint32_t>> adj(n);
    #pragma omp parallel for schedule(dynamic, 64)
    for (std::size_t i = 0; i < n; ++i) {
        for (auto [d, j] : near[i]) {
            if (adj[i].size() == degree) break;
            bool redundant = false;
            for (auto k : adj[i]) if (l2(c.row(k), c.row(j)) < d) { redundant = true; break; }
            if (!redundant) adj[i].push_back(j);
        }
    }
    auto pruned = adj;
    for (std::size_t i = 0; i < n; ++i) for (auto j : pruned[i])
        if (adj[j].size() < degree && std::find(adj[j].begin(), adj[j].end(), i) == adj[j].end()) adj[j].push_back(std::uint32_t(i));
    for (std::size_t i = 0; i < n; ++i) for (auto [d, j] : near[i]) {
        if (adj[i].size() == degree) break;
        if (std::find(adj[i].begin(), adj[i].end(), j) == adj[i].end()) adj[i].push_back(j);
    }
    std::vector<std::uint32_t> edges(n * degree);
    for (std::size_t i = 0; i < n; ++i) std::copy(adj[i].begin(), adj[i].end(), edges.begin() + i * degree);
    return edges;
}

// Best-first beam search; `score_batch(ids, count, out)` returns distances (smaller is closer) for neighbour ids of `node`.
using ScoreNeighbours = std::function<void(std::uint32_t node, const std::uint32_t* ids, float* out)>;
Ranked beam_search(std::size_t n, const std::vector<std::uint32_t>& edges, const std::vector<std::uint32_t>& seeds,
                   const std::vector<float>& seed_distances, std::size_t beam, const ScoreNeighbours& score,
                   std::vector<std::uint32_t>& visited, std::uint32_t stamp, std::size_t& expansions) {
    using Entry = std::pair<float, std::uint32_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<>> frontier;
    std::priority_queue<Entry> best;  // max-heap of the beam
    for (std::size_t s = 0; s < seeds.size(); ++s) {
        if (visited[seeds[s]] == stamp) continue;
        visited[seeds[s]] = stamp; frontier.emplace(seed_distances[s], seeds[s]); best.emplace(seed_distances[s], seeds[s]);
    }
    while (best.size() > beam) best.pop();
    float distances[degree];
    while (!frontier.empty()) {
        auto [d, node] = frontier.top(); frontier.pop();
        if (best.size() == beam && d > best.top().first) break;
        ++expansions;
        const std::uint32_t* nbrs = edges.data() + std::size_t(node) * degree;
        score(node, nbrs, distances);
        for (std::size_t e = 0; e < degree; ++e) {
            std::uint32_t next = nbrs[e];
            if (visited[next] == stamp) continue;
            visited[next] = stamp;
            if (best.size() < beam || distances[e] < best.top().first) {
                frontier.emplace(distances[e], next); best.emplace(distances[e], next);
                if (best.size() > beam) best.pop();
            }
        }
    }
    Ranked out; out.reserve(best.size());
    while (!best.empty()) { out.push_back(best.top()); best.pop(); }
    std::reverse(out.begin(), out.end());
    (void)n; return out;
}


// Per-dimension scalar codes (8 or 4 bits) with an AVX2 single-code distance: |q - x|^2 = |a|^2 - 2 a.(s*c) + |s*c|^2,
// a = q - low. The last term is stored per code; a.(s*c) is a dot of the query weights w = a*s with the code values.
struct ScalarCodes {
    unsigned bits = 8; std::size_t bytes = 0; std::vector<float> low, step; std::vector<std::uint8_t> codes; std::vector<float> tail;
    ScalarCodes(const float* rows, std::size_t n, unsigned b) : bits(b), bytes(dim * b / 8), low(dim, 1e30f), step(dim), codes(n * dim * b / 8), tail(n) {
        std::vector<float> high(dim, -1e30f);
        for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < dim; ++j) { low[j] = std::min(low[j], rows[i * dim + j]); high[j] = std::max(high[j], rows[i * dim + j]); }
        const float levels = float((1u << bits) - 1);
        for (std::size_t j = 0; j < dim; ++j) step[j] = std::max(high[j] - low[j], 1e-12f) / levels;
        for (std::size_t i = 0; i < n; ++i) {
            double t = 0;
            for (std::size_t j = 0; j < dim; ++j) {
                unsigned v = unsigned(std::clamp(std::nearbyint((rows[i * dim + j] - low[j]) / step[j]), 0.f, levels));
                double x = v * double(step[j]); t += x * x;
                if (bits == 8) codes[i * bytes + j] = std::uint8_t(v);
                else codes[i * bytes + j / 2] |= std::uint8_t(v << (4 * (j & 1)));
            }
            tail[i] = float(t);
        }
    }
    const std::uint8_t* row(std::size_t i) const { return codes.data() + i * bytes; }
    struct Query { std::vector<float> w; float head; };
    Query query(const float* q) const {
        Query out{std::vector<float>(dim), 0};
        for (std::size_t j = 0; j < dim; ++j) { float a = q[j] - low[j]; out.w[j] = a * step[j]; out.head += a * a; }
        return out;
    }
    float distance(const Query& q, const std::uint8_t* code, std::size_t id) const {
#ifdef __AVX512F__
        __m512 wide0 = _mm512_setzero_ps(), wide1 = _mm512_setzero_ps();
        if (bits == 8) {
            for (std::size_t j = 0; j < dim; j += 32) {
                __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(code + j));
                wide0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_castsi256_si128(b))), _mm512_loadu_ps(q.w.data() + j), wide0);
                wide1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_extracti128_si256(b, 1))), _mm512_loadu_ps(q.w.data() + j + 16), wide1);
            }
        } else {
            const __m128i mask = _mm_set1_epi8(0x0f);
            for (std::size_t j = 0; j < dim; j += 32) {
                __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(code + j / 2));
                __m128i lo = _mm_and_si128(b, mask), hi = _mm_and_si128(_mm_srli_epi16(b, 4), mask);
                wide0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(lo, hi))), _mm512_loadu_ps(q.w.data() + j), wide0);
                wide1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(lo, hi))), _mm512_loadu_ps(q.w.data() + j + 16), wide1);
            }
        }
        return q.head - 2 * _mm512_reduce_add_ps(_mm512_add_ps(wide0, wide1)) + tail[id];
#endif
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        if (bits == 8) {
            for (std::size_t j = 0; j < dim; j += 16) {
                __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(code + j));
                acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b)), _mm256_loadu_ps(q.w.data() + j), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(b, 8))), _mm256_loadu_ps(q.w.data() + j + 8), acc1);
            }
        } else {
            // Byte k holds dimensions 2k (low nibble) and 2k+1 (high); interleave back into dimension order.
            const __m128i mask = _mm_set1_epi8(0x0f);
            for (std::size_t j = 0; j < dim; j += 16) {
                __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(code + j / 2));
                __m128i v = _mm_unpacklo_epi8(_mm_and_si128(b, mask), _mm_and_si128(_mm_srli_epi16(b, 4), mask));
                acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v)), _mm256_loadu_ps(q.w.data() + j), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(v, 8))), _mm256_loadu_ps(q.w.data() + j + 8), acc1);
            }
        }
        __m256 acc = _mm256_add_ps(acc0, acc1);
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        return q.head - 2 * _mm_cvtss_f32(s) + tail[id];
    }
};
// Lloyd k-means of the centroids into `groups` supers (fp32, GEMM assignment).
std::vector<std::uint32_t> kmeans_groups(const float* rows, std::size_t n, std::size_t groups, std::size_t iterations, std::vector<float>& means) {
    std::mt19937 rng(7); std::vector<std::uint32_t> order(n); std::iota(order.begin(), order.end(), 0u); std::shuffle(order.begin(), order.end(), rng);
    means.assign(groups * dim, 0);
    for (std::size_t g = 0; g < groups; ++g) std::copy_n(rows + std::size_t(order[g]) * dim, dim, means.data() + g * dim);
    std::vector<std::uint32_t> label(n); std::vector<float> dots(n * groups), norms(groups);
    for (std::size_t it = 0; it <= iterations; ++it) {
        for (std::size_t g = 0; g < groups; ++g) norms[g] = cblas_sdot(dim, means.data() + g * dim, 1, means.data() + g * dim, 1);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, groups, dim, 1.f, rows, dim, means.data(), dim, 0.f, dots.data(), groups);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t best = 0; float bd = 1e30f;
            for (std::size_t g = 0; g < groups; ++g) { float d = norms[g] - 2 * dots[i * groups + g]; if (d < bd) { bd = d; best = g; } }
            label[i] = std::uint32_t(best);
        }
        if (it == iterations) break;
        std::vector<double> sum(groups * dim, 0); std::vector<std::size_t> count(groups, 0);
        for (std::size_t i = 0; i < n; ++i) { ++count[label[i]]; for (std::size_t j = 0; j < dim; ++j) sum[label[i] * dim + j] += rows[i * dim + j]; }
        for (std::size_t g = 0; g < groups; ++g) if (count[g]) for (std::size_t j = 0; j < dim; ++j) means[g * dim + j] = float(sum[g * dim + j] / count[g]);
    }
    return label;
}
struct Quality { std::vector<double> covered; double ms = 0, expansions = 0; };
const std::vector<std::size_t> depths = {32, 64, 128, 256, 512, 1024, 2048};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 9) { std::fprintf(stderr, "usage: duel centroids.f32 labels.u32 queries.fvecs truth.ivecs bits_per_dim threads queries search_cpu\n"); return 2; }
    const double bits_per_dim = std::stod(argv[5]);
    const int build_threads = std::stoi(argv[6]); const std::size_t query_rows = std::stoul(argv[7]); const int search_cpu = std::stoi(argv[8]);
    omp_set_num_threads(build_threads);
    Centroids c{read_raw<float>(argv[1]), 0}; c.count = c.rows.size() / dim;
    auto labels = read_raw<std::uint32_t>(argv[2]);
    auto queries = read_vecs<float>(argv[3], dim, query_rows);
    auto truth = read_vecs<std::int32_t>(argv[4], truth_k, query_rows);
    supers = std::max<std::size_t>(256, c.count / 128);
    const std::size_t nq = truth.size() / truth_k, n = c.count;
    std::printf("centroids=%zu queries=%zu bits/dim=%.2f\n", n, nq, bits_per_dim);

    auto t0 = Clock::now();
    auto config = dv::TieredQConfig::target_code_bits(dim, std::size_t(bits_per_dim * dim)); config.metric = dv::TieredQMetric::l2; config.max_scan_bits = 4;
    auto codec = dv::TieredQ::train(config, c.rows.data(), n, 1);
    dv::TieredQIndex flat_codes(codec, dv::TieredQIndexMode::fast); flat_codes.add(c.rows.data(), n);
    dv::TieredQIVFIndex two_level(codec, supers, 1); two_level.build(c.rows.data(), n);
    auto rows = codec.encode(c.rows.data(), n); const std::size_t row_bytes = codec.bytes_per_vector();
    std::printf("codec+indexes %.1fs  code bytes/centroid=%zu  ivf storage=%.1f MiB\n",
                std::chrono::duration<double>(Clock::now() - t0).count(), row_bytes, two_level.storage_bytes() / 1048576.0);
    t0 = Clock::now();
    auto edges = build_graph(c);
    std::printf("graph build %.1fs (degree %zu, exact %zu-NN + RNG prune)\n", std::chrono::duration<double>(Clock::now() - t0).count(), degree, knn);
    // Graph entry: the nearest few of 256 super-centroid medoids (k-means of the centroids via the IVF's own lists).
    auto assignment = two_level.list_assignment();
    std::vector<std::vector<std::uint32_t>> members(two_level.lists());
    for (std::size_t i = 0; i < n; ++i) members[assignment[i]].push_back(std::uint32_t(i));
    std::vector<std::uint32_t> medoids;
    for (auto& m : members) {
        if (m.empty()) continue;
        std::vector<float> mean(dim, 0); for (auto i : m) for (std::size_t j = 0; j < dim; ++j) mean[j] += c.row(i)[j] / m.size();
        medoids.push_back(*std::min_element(m.begin(), m.end(), [&](auto a, auto b) { return l2(mean.data(), c.row(a)) < l2(mean.data(), c.row(b)); }));
    }
    ScalarCodes sq8(c.rows.data(), n, 8), sq4(c.rows.data(), n, 4);
    // Own two-level router: fp32 k-means supers ranked exactly, member codes stored contiguously per super.
    std::vector<float> super_means; auto super_of = kmeans_groups(c.rows.data(), n, supers, 6, super_means);
    std::vector<std::uint32_t> grouped(n); std::vector<std::size_t> group_begin(supers + 1, 0);
    for (auto g : super_of) ++group_begin[g + 1];
    std::partial_sum(group_begin.begin(), group_begin.end(), group_begin.begin());
    { auto fill = group_begin; for (std::size_t i = 0; i < n; ++i) grouped[fill[super_of[i]]++] = std::uint32_t(i); }
    std::vector<float> grouped_rows(n * dim); for (std::size_t i = 0; i < n; ++i) std::copy_n(c.row(grouped[i]), dim, grouped_rows.data() + i * dim);
    ScalarCodes gsq8(grouped_rows.data(), n, 8), gsq4(grouped_rows.data(), n, 4);
    grouped_rows.clear(); grouped_rows.shrink_to_fit();
    omp_set_num_threads(1);
    // Timed routing runs on one pinned core, like the index benchmarks.
    cpu_set_t only; CPU_ZERO(&only); CPU_SET(search_cpu, &only);
    if (sched_setaffinity(0, sizeof(only), &only)) throw std::runtime_error("cannot pin the search thread");
    std::printf("supers=%zu search_cpu=%d\n", supers, search_cpu);

    // True lists per query (with multiplicity): the lists of its true top-10 neighbours.
    auto coverage = [&](const float*, std::size_t qi, const Ranked& order, std::vector<double>& covered) {
        std::vector<std::uint32_t> rank(n, UINT32_MAX);
        for (std::size_t i = 0; i < order.size(); ++i) rank[order[i].second] = std::uint32_t(i);
        for (std::size_t di = 0; di < depths.size(); ++di) {
            std::size_t hit = 0;
            for (std::size_t t = 0; t < truth_k; ++t) hit += rank[labels[truth[qi * truth_k + t]]] < depths[di];
            covered[di] += double(hit) / truth_k / nq;
        }
    };
    auto run = [&](const char* name, const std::function<Ranked(const float*, std::size_t&)>& route) {
        Quality quality; quality.covered.assign(depths.size(), 0);
        std::size_t warm = 0; route(queries.data(), warm);
        std::vector<Ranked> orders(nq); std::size_t expansions = 0;
        auto start = Clock::now();
        for (std::size_t qi = 0; qi < nq; ++qi) orders[qi] = route(queries.data() + qi * dim, expansions);
        quality.ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count() / nq;
        for (std::size_t qi = 0; qi < nq; ++qi) coverage(queries.data() + qi * dim, qi, orders[qi], quality.covered);
        std::printf("%-34s %7.3f ms %7.0f exp |", name, quality.ms, double(expansions) / nq);
        for (auto v : quality.covered) std::printf(" %.4f", v);
        std::printf("\n"); std::fflush(stdout);
    };
    std::printf("%-34s %10s %11s |", "router", "route", "expansions"); for (auto d : depths) std::printf(" L=%-4zu", d); std::printf("\n");

    run("exact fp32 scan", [&](const float* q, std::size_t&) {
        Ranked order(n); for (std::size_t i = 0; i < n; ++i) order[i] = {l2(q, c.row(i)), std::uint32_t(i)};
        std::partial_sort(order.begin(), order.begin() + depths.back(), order.end()); order.resize(depths.back()); order.shrink_to_fit(); return order;
    });
    auto from_hits = [&](const float* q, const std::vector<dv::TieredQHit>& hits) {
        Ranked order; order.reserve(hits.size()); for (auto& h : hits) order.emplace_back(-h.score, std::uint32_t(h.id));
        rerank_head(c, q, order); return order;
    };
    for (std::size_t k : {1024ul, 2048ul})
        run(("compressed full scan k=" + std::to_string(k)).c_str(), [&](const float* q, std::size_t&) { return from_hits(q, flat_codes.search_fast(q, k)); });
    for (std::size_t nprobe : {16ul, 64ul})
        for (std::size_t k : {2048ul})
            run(("two-level nprobe=" + std::to_string(nprobe) + " k=" + std::to_string(k)).c_str(),
                [&](const float* q, std::size_t&) { return from_hits(q, two_level.search(q, k, nprobe)); });

    run("tieredq ivf all lists k=2048", [&](const float* q, std::size_t&) { return from_hits(q, two_level.search(q, 2048, supers)); });
    for (const ScalarCodes* sq : {&gsq8, &gsq4})
        for (std::size_t nprobe : {8ul, 16ul, 32ul, 64ul, 128ul}) {
            run(("own two-level sq" + std::to_string(sq->bits) + " nprobe=" + std::to_string(nprobe)).c_str(), [&](const float* q, std::size_t&) {
                Ranked groups(supers); for (std::size_t g = 0; g < supers; ++g) groups[g] = {l2(q, super_means.data() + g * dim), std::uint32_t(g)};
                std::partial_sort(groups.begin(), groups.begin() + nprobe, groups.end());
                auto weights = sq->query(q); Ranked order;
                for (std::size_t p = 0; p < nprobe; ++p) for (std::size_t i = group_begin[groups[p].second]; i < group_begin[groups[p].second + 1]; ++i)
                    order.emplace_back(sq->distance(weights, sq->row(i), i), grouped[i]);
                std::size_t keep = std::min(depths.back(), order.size());
                std::partial_sort(order.begin(), order.begin() + keep, order.end()); order.resize(keep);
                rerank_head(c, q, order); return order;
            });
        }
    for (const ScalarCodes* sq : {&sq8, &sq4})
        run(("full sq" + std::to_string(sq->bits) + " scan").c_str(), [&](const float* q, std::size_t&) {
            auto weights = sq->query(q); Ranked order(n);
            for (std::size_t i = 0; i < n; ++i) order[i] = {sq->distance(weights, sq->row(i), i), std::uint32_t(i)};
            std::partial_sort(order.begin(), order.begin() + depths.back(), order.end()); order.resize(depths.back()); order.shrink_to_fit();
            rerank_head(c, q, order); return order;
        });
    std::vector<std::uint32_t> visited(n, 0); std::uint32_t stamp = 0;
    const std::size_t entry_seeds = 8;
    auto seeds_for = [&](const float* q, std::vector<std::uint32_t>& seeds, std::vector<float>& sd, const std::function<float(std::uint32_t)>& dist) {
        Ranked m; for (auto id : medoids) m.emplace_back(l2(q, c.row(id)), id);
        std::partial_sort(m.begin(), m.begin() + entry_seeds, m.end());
        seeds.clear(); sd.clear(); for (std::size_t s = 0; s < entry_seeds; ++s) { seeds.push_back(m[s].second); sd.push_back(dist(m[s].second)); }
    };
    for (std::size_t beam : {512ul, 1024ul, 2048ul, 4096ul, 8192ul}) {
        run(("graph fp32 beam=" + std::to_string(beam)).c_str(), [&](const float* q, std::size_t& exp) {
            std::vector<std::uint32_t> seeds; std::vector<float> sd;
            seeds_for(q, seeds, sd, [&](std::uint32_t id) { return l2(q, c.row(id)); });
            auto order = beam_search(n, edges, seeds, sd, beam, [&](std::uint32_t, const std::uint32_t* ids, float* out) {
                for (std::size_t e = 0; e < degree; ++e) out[e] = l2(q, c.row(ids[e]));
            }, visited, ++stamp, exp);
            return order;
        });
        for (const ScalarCodes* sq : {&sq8, &sq4}) {
            run(("graph sq" + std::to_string(sq->bits) + " beam=" + std::to_string(beam)).c_str(), [&](const float* q, std::size_t& exp) {
                auto weights = sq->query(q);
                std::vector<std::uint32_t> seeds; std::vector<float> sd;
                seeds_for(q, seeds, sd, [&](std::uint32_t id) { return sq->distance(weights, sq->row(id), id); });
                auto order = beam_search(n, edges, seeds, sd, beam, [&](std::uint32_t, const std::uint32_t* ids, float* out) {
                    for (std::size_t e = 0; e < degree; ++e) _mm_prefetch(reinterpret_cast<const char*>(sq->row(ids[e])), _MM_HINT_T0);
                    for (std::size_t e = 0; e < degree; ++e) out[e] = sq->distance(weights, sq->row(ids[e]), ids[e]);
                }, visited, ++stamp, exp);
                rerank_head(c, q, order); return order;
            });
        }
    }
}
