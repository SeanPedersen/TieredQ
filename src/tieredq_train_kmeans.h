// K-means centroid updates. Grouping preserves the input order within each centroid's sum.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dv::tieredq_detail {
inline void update_centroids_reference(const float* sample,size_t rows,size_t dimensions,const uint32_t* ids,
                                      std::vector<float>& centroids) {
  const size_t clusters=centroids.size()/dimensions;
  std::vector<double> sums(centroids.size()); std::vector<size_t> counts(clusters);
  for (size_t i=0;i<rows;++i) {
    const size_t id=ids[i]; ++counts[id];
    for (size_t j=0;j<dimensions;++j) sums[id*dimensions+j]+=sample[i*dimensions+j];
  }
  for (size_t id=0;id<clusters;++id) if (counts[id])
    for (size_t j=0;j<dimensions;++j) centroids[id*dimensions+j]=float(sums[id*dimensions+j]/counts[id]);
}
// value(row, j) reads a sample coordinate; rows are summed in input order per centroid, as the reference does.
template<class Value>
void update_centroids_with(Value value,size_t rows,size_t dimensions,const uint32_t* ids,std::vector<float>& centroids) {
  const size_t clusters=centroids.size()/dimensions;
  constexpr size_t minimum_parallel_dimensions=1536;
  if (dimensions<minimum_parallel_dimensions) {
    std::vector<double> sums(centroids.size()); std::vector<size_t> counts(clusters);
    for (size_t i=0;i<rows;++i) {
      const size_t id=ids[i]; ++counts[id];
      for (size_t j=0;j<dimensions;++j) sums[id*dimensions+j]+=value(i,j);
    }
    for (size_t id=0;id<clusters;++id) if (counts[id])
      for (size_t j=0;j<dimensions;++j) centroids[id*dimensions+j]=float(sums[id*dimensions+j]/counts[id]);
    return;
  }
  std::vector<size_t> offsets(clusters+1),grouped(rows);
  for (size_t i=0;i<rows;++i) ++offsets[ids[i]+1];
  for (size_t id=0;id<clusters;++id) offsets[id+1]+=offsets[id];
  auto cursors=offsets;
  for (size_t i=0;i<rows;++i) grouped[cursors[ids[i]]++]=i;
  constexpr size_t dimension_tile=64;
  const size_t tiles=(dimensions+dimension_tile-1)/dimension_tile;
#pragma omp parallel for schedule(static)
  for (size_t task=0;task<clusters*tiles;++task) {
    const size_t id=task/tiles,begin=(task%tiles)*dimension_tile;
    const size_t width=std::min(dimension_tile,dimensions-begin),count=offsets[id+1]-offsets[id];
    if (!count) continue;
    std::array<double,dimension_tile> sums{};
    for (size_t p=offsets[id];p<offsets[id+1];++p)
      for (size_t j=0;j<width;++j) sums[j]+=value(grouped[p],begin+j);
    for (size_t j=0;j<width;++j) centroids[id*dimensions+begin+j]=float(sums[j]/count);
  }
}
inline void update_centroids(const float* sample,size_t rows,size_t dimensions,const uint32_t* ids,
                             std::vector<float>& centroids) {
  update_centroids_with([&](size_t row,size_t j) { return sample[row*dimensions+j]; },rows,dimensions,ids,centroids);
}
} // namespace dv::tieredq_detail
