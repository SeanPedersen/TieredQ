// Calibration-only geometry diagnostics. Temporary inverse IDs are outside timed search.
#include "search_state.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
namespace geometry_ann {
std::vector<GeometryDiagnostic> Index::diagnose_geometry(std::span<const float> queries,
        std::span<const std::vector<Id>> truth) const {
    if (queries.size()!=truth.size()*dimension_) throw std::invalid_argument("diagnostic query shape");
    std::vector<Id> position(ids_.size());
    for (std::size_t i=0;i<ids_.size();++i) position[ids_[i]]=Id(i);
    auto point_distance=[&](std::span<const float> query,std::size_t packed) {
        return codes_.empty()?centroid_l2(query,std::span(vectors_).subspan(packed*dimension_,dimension_)):
            sq8_centroid_l2(query,std::span(codes_).subspan(packed*dimension_,dimension_),code_scales_,code_offsets_);
    };
    std::vector<GeometryDiagnostic> result;
    for (std::size_t qi=0;qi<truth.size();++qi) {
        auto query=queries.subspan(qi*dimension_,dimension_);
        std::vector<std::size_t> truth_leaves; std::vector<double> member_distances;
        double reference=0;
        for (Id id:truth[qi]) {
            if (id>=position.size()) throw std::invalid_argument("diagnostic truth ID");
            auto packed=position[id];
            truth_leaves.push_back(leaf_containing(leaves_,packed));
            double distance=codes_.empty()?l2(query,std::span(vectors_).subspan(std::size_t(packed)*dimension_,dimension_)):
                sq8_l2(query,std::span(codes_).subspan(std::size_t(packed)*dimension_,dimension_),code_scales_,code_offsets_);
            member_distances.push_back(distance); reference=std::max(reference,distance);
        }
        if (truth_leaves.size()!=10) throw std::invalid_argument("oracle coverage requires 10 truth IDs");
        std::vector<std::size_t> profits(leaves_.size());
        for (auto l:truth_leaves) ++profits[l];
        auto cost=cover_costs(profits,leaves_,10);
        auto cluster_count=std::size_t(std::count_if(profits.begin(),profits.end(),[](std::size_t p){return p>0;}));
        for (auto mode:{BoundsMode::none,BoundsMode::radius,BoundsMode::shared,BoundsMode::shared_residual,BoundsMode::support}) {
            if (mode!=BoundsMode::none && !geometry_metadata_) continue;
            if (mode==BoundsMode::support && edge_supports_.empty()) continue;
            if ((mode==BoundsMode::shared || mode==BoundsMode::shared_residual) && shared_geometry_.norm.empty()) continue;
            for (bool witness_ordering:{false,true}) {
            if (witness_ordering && mode!=BoundsMode::support) continue;
            std::vector<double> witness_upper;
            std::size_t witness_evaluations=0;
            if (witness_ordering) {
                witness_upper.assign(leaves_.size(),std::numeric_limits<double>::infinity());
                for (std::size_t l=0;l<leaves_.size();++l) {
                    std::vector<Id> witnesses;
                    for (std::size_t e=0;e<graph_degree_;++e) {
                        const auto& support=edge_supports_[l*graph_degree_+e];
                        witnesses.push_back(support.low_witness);witnesses.push_back(support.high_witness);
                    }
                    std::sort(witnesses.begin(),witnesses.end());witnesses.erase(std::unique(witnesses.begin(),witnesses.end()),witnesses.end());
                    witness_evaluations+=witnesses.size();
                    for (Id p:witnesses) witness_upper[l]=std::min(witness_upper[l],point_distance(query,p));
                }
            }
            for (double weight:{.0,.5,1.}) {
                if (witness_ordering && weight==0) continue;
                if (mode==BoundsMode::none && weight!=0) continue;
                SearchOptions options; options.bounds=mode; options.metadata_evaluations=leaves_.size();
                options.bound_weight=weight; options.ordering=Ordering::hybrid;
                SearchState state(query,options,variant_,vectors_,ids_,leaves_,&shared_geometry_,{codes_,code_scales_,code_offsets_});
                std::vector<Visit> visits; visits.reserve(leaves_.size());
                GeometryDiagnostic row{qi,0,0,0,weight,0,0,0,mode};
                for (std::size_t l=0;l<leaves_.size();++l) {
                    auto visit=*state.evaluate(leaves_[l].geometry,l,false);
                    visits.push_back(visit); row.positive+=visit.bound>0; row.above_reference+=visit.bound>reference;
                    if (reference>0) row.mean_bound_ratio+=visit.bound/reference;
                }
                if (mode==BoundsMode::support) {
                    std::vector<double> distances;
                    for (const auto& visit:visits) distances.push_back(visit.centroid_distance);
                    row.positive=0;row.above_reference=0;row.mean_bound_ratio=0;
                    for (auto& visit:visits) {
                        visit.bound=std::max(visit.bound,edge_support_lower_bound(
                            std::span(edge_supports_).subspan(visit.index*graph_degree_,graph_degree_),
                            std::span(graph_edges_).subspan(visit.index*graph_degree_,graph_degree_),distances,visit.centroid_distance,dimension_));
                        visit.priority=(1-weight)*visit.centroid_distance+weight*visit.bound;
                        row.positive+=visit.bound>0;row.above_reference+=visit.bound>reference;
                        if (reference>0)row.mean_bound_ratio+=visit.bound/reference;
                    }
                }
                if (witness_ordering) for (auto& visit:visits)
                    visit.priority=(1-weight)*visit.centroid_distance+weight*witness_upper[visit.index];
                row.witness_ordering=witness_ordering;row.witness_evaluations=witness_evaluations;
                for (std::size_t t=0;t<truth_leaves.size();++t)
                    row.unsafe_members+=visits[truth_leaves[t]].bound>member_distances[t];
                std::sort(visits.begin(),visits.end(),[](const Visit& a,const Visit& b){return VisitGreater{}(b,a);});
                std::vector<std::size_t> rank(leaves_.size());
                for (std::size_t l=0;l<visits.size();++l) rank[visits[l].index]=l+1;
                for (auto leaf:truth_leaves) {
                    row.mean_truth_cluster_rank+=double(rank[leaf]);
                    row.max_truth_cluster_rank=std::max(row.max_truth_cluster_rank,double(rank[leaf]));
                }
                row.mean_truth_cluster_rank/=double(truth_leaves.size()); row.mean_bound_ratio/=double(leaves_.size());
                row.truth_clusters=cluster_count; row.oracle_vectors_9=std::min(cost[9],cost[10]);row.oracle_vectors_10=cost[10];
                std::size_t spent=0,covered=0;
                for (const auto& visit:visits) {
                    spent+=leaves_[visit.index].size;covered+=profits[visit.index];
                    if (covered>=9 && !row.ranked_vectors_9) row.ranked_vectors_9=spent;
                    if (covered==10) {row.ranked_vectors_10=spent;break;}
                }
                result.push_back(row);
            }
            }
        }
    }
    return result;
}
std::size_t leaf_containing(std::span<const Leaf> leaves,std::size_t packed) {
    auto it=std::upper_bound(leaves.begin(),leaves.end(),packed,[](std::size_t p,const Leaf& l){return p<l.offset;});
    if (it==leaves.begin()) throw std::invalid_argument("packed row precedes first leaf");
    return std::size_t(std::prev(it)-leaves.begin());
}
std::vector<std::size_t> cover_costs(std::span<const std::size_t> profits,std::span<const Leaf> leaves,std::size_t k) {
    constexpr auto unreachable=std::numeric_limits<std::size_t>::max()/2;
    std::vector<std::size_t> cost(k+1,unreachable);cost[0]=0;
    for (std::size_t l=0;l<leaves.size();++l) if (profits[l]) {
        auto next=cost;
        for (std::size_t n=0;n<cost.size();++n) {
            auto reached=std::min(k,n+profits[l]);
            next[reached]=std::min(next[reached],cost[n]+leaves[l].size);
        }
        cost=std::move(next);
    }
    return cost;
}
}
