// Strict fvecs/ivecs readers and deterministic clustered synthetic data for smoke benchmarks.
#include "dataset.h"
#include "internal.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <filesystem>
#include <iomanip>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
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
QuantizedDataset load_quantized(const std::string& path, const std::string& scale_path, std::size_t count, std::size_t d, const std::string& offset_path) {
    if (!count || !d || count > std::numeric_limits<std::size_t>::max() / d)
        throw std::runtime_error("invalid SQ8 shape");
    QuantizedDataset result;
    result.scales.resize(d); std::ifstream scales(scale_path,std::ios::binary);
    scales.read(reinterpret_cast<char*>(result.scales.data()),std::streamsize(d*sizeof(float)));
    if (!scales || scales.peek() != std::char_traits<char>::eof()) throw std::runtime_error("invalid SQ8 scales file");
    if (!offset_path.empty()) {
        result.offsets.resize(d); std::ifstream offsets(offset_path,std::ios::binary);
        offsets.read(reinterpret_cast<char*>(result.offsets.data()),std::streamsize(d*sizeof(float)));
        if (!offsets || offsets.peek() != std::char_traits<char>::eof()) throw std::runtime_error("invalid SQ8 offsets file");
    }
    for (std::size_t j=0;j<d;++j) {
        double offset=result.offsets.empty()?0:result.offsets[j];
        if (!std::isfinite(result.scales[j]) || result.scales[j]<=0 || !std::isfinite(offset) ||
            std::abs(offset)+128*double(result.scales[j])>std::numeric_limits<float>::max())
            throw std::runtime_error("invalid SQ8 parameter range");
    }
    int fd = open(path.c_str(),O_RDONLY); if (fd < 0) throw std::runtime_error("cannot open SQ8 codes");
    struct stat info{}; auto bytes=count*d;
    if (fstat(fd,&info) || info.st_size < 0 || std::size_t(info.st_size) != bytes) {
        close(fd); throw std::runtime_error("SQ8 code size mismatch");
    }
    void* mapping=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    if (mapping==MAP_FAILED) throw std::runtime_error("cannot map SQ8 codes");
    result.mapping=std::shared_ptr<void>(mapping,[bytes](void* p){munmap(p,bytes);});
    result.codes={static_cast<const std::int8_t*>(mapping),bytes};return result;
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
    generate(count, data.base_storage); data.base = data.base_storage; data.base_rows = count; generate(query_count, data.queries);
    for (std::size_t i = 0; i < query_count; ++i) {
        std::vector<Id> ids;
        for (auto neighbor : exact(data.base, std::span(data.queries).subspan(i * d, d), 10)) ids.push_back(neighbor.id);
        data.truth.push_back(std::move(ids));
    }
    return data;
}
Dataset load_dataset(const std::string& base, const std::string& queries, const std::string& truth, std::size_t limit) {
    Dataset data;
    if (std::filesystem::path(base).extension() == ".rawview") {
        // A small descriptor allows mapping contiguous HDF5 payloads without a
        // second 30+ GB export. The preparation script validates dtype/layout.
        std::ifstream descriptor(base);
        std::string magic, filename, extra; std::size_t rows = 0, offset = 0;
        if (!(descriptor >> magic >> rows >> data.dimension >> offset >> std::quoted(filename)) ||
            magic != "geometry_raw_f32_v1" || !rows || !data.dimension || offset % alignof(float) ||
            rows > std::numeric_limits<std::size_t>::max() / data.dimension / sizeof(float) ||
            (descriptor >> extra)) throw std::runtime_error("invalid rawview descriptor");
        auto path = std::filesystem::path(filename);
        if (path.is_relative()) path = std::filesystem::path(base).parent_path() / path;
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open mapped base");
        struct stat statbuf{};
        std::size_t bytes = rows * data.dimension * sizeof(float);
        if (fstat(fd, &statbuf) || statbuf.st_size < 0 || offset > std::size_t(statbuf.st_size) ||
            bytes > std::size_t(statbuf.st_size) - offset) {
            close(fd); throw std::runtime_error("rawview exceeds backing file");
        }
        long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) { close(fd); throw std::runtime_error("invalid page size"); }
        std::size_t aligned = offset / std::size_t(page_size) * std::size_t(page_size);
        std::size_t length = offset - aligned + bytes;
        void* mapping = mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, off_t(aligned));
        close(fd);
        if (mapping == MAP_FAILED) throw std::runtime_error("cannot map base");
        data.base_mapping = std::shared_ptr<void>(mapping, [length](void* p) { munmap(p, length); });
        data.base = {reinterpret_cast<const float*>(static_cast<const char*>(mapping) + offset - aligned), rows * data.dimension};
    } else {
        data.base_storage = read_records<float>(base, data.dimension);
        data.base = data.base_storage;
    }
    data.base_rows = data.base.size() / data.dimension;
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
std::vector<std::uint32_t> load_labels(const std::string& path, std::size_t rows) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open labels " + path);
    if (std::size_t(input.tellg()) != rows * sizeof(std::uint32_t)) throw std::runtime_error("labels do not match base rows: " + path);
    std::vector<std::uint32_t> labels(rows);
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(labels.data()), std::streamsize(rows * sizeof(std::uint32_t))))
        throw std::runtime_error("cannot read labels " + path);
    return labels;
}
}
