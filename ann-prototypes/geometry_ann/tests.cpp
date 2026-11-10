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
        // Exercise SIMD blocks, tails, cancellation, overflow and underflow.
        for (std::size_t d : {1, 7, 8, 15, 16, 17, 31, 32, 33, 128, 1024}) {
            for (float scale : {1e-25f, 1e-10f, 1.f, 1e10f, 1e25f}) {
                std::vector<float> a(d), b(d);
                for (int trial = 0; trial < 50; ++trial) {
                    for (std::size_t j = 0; j < d; ++j) {
                        a[j] = normal(rng) * scale;
                        b[j] = trial % 2 ? std::nextafter(a[j], 0.f) : normal(rng) * scale;
                    }
                    require(centroid_l2(a, b) == l2(a, b), "SIMD centroid distance changed double metric");
                    require(scan_lower_bound(a, b) <= l2(a, b), "FP32 filter exceeds reference distance");
                }
            }
        }
        for (std::size_t d : {1,7,8,16,31,32,33,128,1024}) {
            std::vector<std::int8_t> code(d); std::vector<float> scales(d), offsets(d), query(d), decoded(d);
            for (float magnitude : {1e-25f,1.f,1e25f}) for (int trial = 0; trial < 20; ++trial) {
                for (std::size_t j=0;j<d;++j) {
                    code[j]=std::int8_t(int(rng()%256)-128); scales[j]=magnitude*(.01f+float(rng()%100)/100);
                    offsets[j]=normal(rng)*magnitude; decoded[j]=float(code[j])*scales[j]+offsets[j]; query[j]=trial%2 ? std::nextafter(decoded[j],0.f) : normal(rng)*magnitude;
                }
                require(sq8_centroid_l2(query,code,scales,offsets)==sq8_l2(query,code,scales,offsets), "SQ8 training SIMD changed the metric");
                require(sq8_l2(query,code,scales,offsets)==l2(query,decoded), "SQ8 double metric differs from decoded FP32");
                require(sq8_lower_bound(query,code,scales,offsets)<=sq8_l2(query,code,scales,offsets), "SQ8 SIMD filter rejects a competitive vector");
            }
        }
        {
            constexpr std::size_t qd=33, qn=193;
            std::vector<float> source(qd*qn), decoded(qd*qn), scales(qd,.013f), offsets(qd);
            for (auto& offset : offsets) offset=normal(rng)*.05f;
            std::vector<std::int8_t> codes(qd*qn);
            for (std::size_t j=0;j<source.size();++j) {
                source[j]=normal(rng)*.3f; codes[j]=std::int8_t(std::clamp(std::floor((source[j]-offsets[j%qd])/scales[j%qd]+.5f),-128.f,127.f));
                decoded[j]=float(codes[j])*scales[j%qd]+offsets[j%qd];
            }
            BuildOptions build; build.coarse=4;build.children=4;build.leaf_budget=16;build.graph_degree=8;
            Index quantized({},qd,build,{codes,scales,offsets});
            Index decoded_reference(decoded,qd,build);
            require(quantized.raw_bytes()==codes.size(), "SQ8 retained an FP32 corpus");
            for (std::size_t q=0;q<10;++q) {
                auto query=std::span(source).subspan(q*qd,qd);auto expected=brute(decoded,query,10);
                for (auto routing : {Routing::flat,Routing::graph,Routing::hierarchy}) {
                    SearchOptions search;search.bounds=BoundsMode::none;search.routing=routing;search.leaf_scans=quantized.leaf_count();
                    auto result=quantized.search(query,search);
                    require(result.neighbors.size()==expected.size(), "SQ8 result count changed");
                    for (std::size_t i=0;i<expected.size();++i)
                        require(result.neighbors[i].id==expected[i].id && result.neighbors[i].distance==expected[i].distance,"SQ8 differs from exhaustive decoded search");
                    search.vector_scans=11;search.metadata_evaluations=7;
                    auto limited=quantized.search(query,search);
                    auto reference=decoded_reference.search(query,search);
                    require(limited.neighbors.size()==reference.neighbors.size() && limited.vectors_scanned==reference.vectors_scanned,
                            "native SQ8 training changed partition budgets");
                    for (std::size_t i=0;i<reference.neighbors.size();++i)
                        require(limited.neighbors[i].id==reference.neighbors[i].id && limited.neighbors[i].distance==reference.neighbors[i].distance,
                                "native SQ8 training differs from decoded reference");
                    require(limited.vectors_scanned<=11 && limited.metadata_evaluations<=7,"SQ8 exceeded hard caps");
                }
            }
            // Native INT8 envelopes must preserve exhaustive results and hard caps.
            auto shaped=build; shaped.geometry_metadata=true; shaped.shared_directions=8; shaped.neighbor_support=true;
            Index shaped_index({},qd,shaped,{codes,scales,offsets});
            for (std::size_t q=0;q<10;++q) {
                auto query=std::span(source).subspan(q*qd,qd); auto expected=brute(decoded,query,10);
                for (auto mode:{BoundsMode::radius,BoundsMode::shared,BoundsMode::shared_residual})
                for (auto route:{Routing::flat,Routing::graph,Routing::hierarchy}) {
                    SearchOptions search; search.bounds=mode; search.routing=route; search.leaf_scans=shaped_index.leaf_count();
                    auto found=shaped_index.search(query,search);
                    require(found.neighbors.size()==expected.size(),"SQ8 geometry lost candidates");
                    for (std::size_t i=0;i<expected.size();++i)
                        require(found.neighbors[i].id==expected[i].id && found.neighbors[i].distance==expected[i].distance,
                                "SQ8 geometry differs from exhaustive decoded result");
                    search.vector_scans=17; search.metadata_evaluations=9;
                    auto limited=shaped_index.search(query,search);
                    require(limited.vectors_scanned<=17 && limited.metadata_evaluations<=9,"SQ8 geometry exceeded hard caps");
                }
            }
            SearchOptions support;support.bounds=BoundsMode::support;support.routing=Routing::flat;
            support.leaf_scans=shaped_index.leaf_count();
            std::vector<std::vector<Id>> diagnostic_truth;
            for (std::size_t q=0;q<10;++q) {
                auto query=std::span(source).subspan(q*qd,qd);auto expected=brute(decoded,query,10);
                auto found=shaped_index.search(query,support);
                std::vector<Id> ids;
                for (std::size_t i=0;i<expected.size();++i) {
                    require(found.neighbors[i].id==expected[i].id && found.neighbors[i].distance==expected[i].distance,
                            "INT8 edge-support search differs from exhaustive result");
                    ids.push_back(expected[i].id);
                }
                diagnostic_truth.push_back(ids);
                support.vector_scans=40;auto limited=shaped_index.search(query,support);
                require(limited.vectors_scanned<=40,"edge support exceeded vector cap");
                support.vector_scans=std::numeric_limits<std::size_t>::max();
            }
            for (const auto& d:shaped_index.diagnose_geometry(std::span(source).first(10*qd),diagnostic_truth))
                require(d.unsafe_members==0 && d.oracle_vectors_9<=d.oracle_vectors_10 &&
                        d.oracle_vectors_10<=d.ranked_vectors_10 && d.oracle_vectors_9<=d.ranked_vectors_9,
                        "oracle coverage or support diagnostic invalid");
            support.routing=Routing::graph;
            bool unsupported=false;
            try {shaped_index.search(std::span(source).first(qd),support);}catch(const std::invalid_argument&){unsupported=true;}
            require(unsupported,"edge support accepted graph search without complete centroid cache");
            bool rejected=false;
            try { SearchOptions missing; missing.bounds=BoundsMode::radius; quantized.search(std::span(source).first(qd),missing); } catch (const std::invalid_argument&) { rejected=true; }
            require(rejected,"SQ8 accepted missing radius metadata");
        }
        check_index(data, dimension, false); check_index(data, dimension, true);
        std::vector<float> duplicates(31 * dimension, 2); check_index(duplicates, dimension, false);
        std::vector<float> singleton(dimension, 0); check_index(singleton, dimension, false);
        std::vector<float> wide(population * 33);
        for (auto& value : wide) value = normal(rng);
        check_index(wide, 33, true);
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
