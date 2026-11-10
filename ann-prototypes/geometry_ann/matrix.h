// Non-owning training matrix. SQ8 rows decode into registers, never a corpus-sized FP32 buffer.
#pragma once
#include "index.h"
#include <ranges>
#include <type_traits>
namespace geometry_ann {
double centroid_l2(std::span<const float>,std::span<const float>);
double sq8_centroid_l2(std::span<const float>,std::span<const std::int8_t>,std::span<const float>,std::span<const float>);
struct Matrix {
    std::span<const float> floats;
    QuantizedInput quantized;
    Matrix(std::span<const float> values,QuantizedInput codes={}) : floats(values),quantized(codes) {}
    template<std::ranges::contiguous_range Range>
        requires std::is_same_v<std::ranges::range_value_t<Range>,float>
    Matrix(const Range& values) : floats(values) {}
    std::size_t size() const { return quantized.codes.empty()?floats.size():quantized.codes.size(); }
    const float* data() const { return floats.data(); }
    float value(Id id,std::size_t j,std::size_t d) const {
        if (quantized.codes.empty()) return floats[std::size_t(id)*d+j];
        float result=float(quantized.codes[std::size_t(id)*d+j])*quantized.scales[j];
        if (!quantized.offsets.empty()) result+=quantized.offsets[j];
        return result;
    }
    double distance(std::span<const float> query,Id id,std::size_t d) const {
        if (quantized.codes.empty()) return centroid_l2(query,floats.subspan(std::size_t(id)*d,d));
        return sq8_centroid_l2(query,quantized.codes.subspan(std::size_t(id)*d,d),quantized.scales,quantized.offsets);
    }
};
inline std::vector<float> decoded_row(Matrix data,std::size_t d,Id id) {
    std::vector<float> result(d);
    for (std::size_t j=0;j<d;++j) result[j]=data.value(id,j,d);
    return result;
}
}
