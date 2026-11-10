// Full-member support slabs along centroid-graph edges. Query projections reuse centroid distances.
#include "internal.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace geometry_ann {
namespace {
double error_factor(std::size_t d) {return 32*std::numeric_limits<double>::epsilon()*double(d+8);}
}
std::vector<EdgeSupport> build_edge_supports(Matrix packed,std::size_t d,
        std::span<const Leaf> leaves,std::span<const Id> edges,std::size_t degree) {
    if (edges.size()!=leaves.size()*degree) throw std::invalid_argument("support graph shape");
    std::vector<EdgeSupport> result(edges.size());
    for (std::size_t l=0;l<leaves.size();++l) {
        const auto& leaf=leaves[l];
        std::vector<double> directions(degree*d);
        for (std::size_t e=0;e<degree;++e) {
            auto neighbor=edges[l*degree+e];
            if (neighbor>=leaves.size()) throw std::invalid_argument("support neighbor ID");
            auto& support=result[l*degree+e];
            support.length_squared=l2(leaf.geometry.centroid,leaves[neighbor].geometry.centroid);
            support.low=std::numeric_limits<double>::infinity();support.high=-support.low;
            for (std::size_t j=0;j<d;++j)
                directions[e*d+j]=double(leaves[neighbor].geometry.centroid[j])-leaf.geometry.centroid[j];
        }
        for (std::size_t i=leaf.offset;i<leaf.offset+leaf.size;++i) {
            auto point=decoded_row(packed,d,Id(i));
            for (std::size_t e=0;e<degree;++e) {
                double projection=0,magnitude=0;
                for (std::size_t j=0;j<d;++j) {
                    double axis=directions[e*d+j];
                    projection+=(double(point[j])-leaf.geometry.centroid[j])*axis;
                    magnitude+=(std::abs(double(point[j]))+std::abs(double(leaf.geometry.centroid[j])))*std::abs(axis);
                }
                double error=error_factor(d)*magnitude;
                auto& support=result[l*degree+e];
                double low=std::nextafter(projection-error,-std::numeric_limits<double>::infinity());
                double high=std::nextafter(projection+error,std::numeric_limits<double>::infinity());
                if (low<support.low) {support.low=low;support.low_witness=Id(i);}
                if (high>support.high) {support.high=high;support.high_witness=Id(i);}
            }
        }
    }
    return result;
}
double edge_support_lower_bound(std::span<const EdgeSupport> supports,
        std::span<const Id> neighbors,std::span<const double> distances,
        double centroid_distance,std::size_t dimension) {
    double bound=0,eps=error_factor(dimension);
    for (std::size_t e=0;e<supports.size();++e) {
        const auto& s=supports[e];
        if (s.length_squared==0) continue;
        // 2(q-ci)·(cj-ci) = ||q-ci||² + ||cj-ci||² - ||q-cj||².
        double other=distances[neighbors[e]];
        double projection=(centroid_distance+s.length_squared-other)*.5;
        double slack=eps*(centroid_distance+other+s.length_squared+std::abs(s.low)+std::abs(s.high));
        double gap=std::max({0.0,s.low-projection-slack,projection-s.high-slack});
        bound=std::max(bound,gap*gap/(s.length_squared*(1+eps)));
    }
    return bound;
}
}
