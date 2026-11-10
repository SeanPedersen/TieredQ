// Mapped inputs must match vecs inputs, retain ownership after moves, and reject bad views.
#include "dataset.h"
#include <cstdlib>
#include <algorithm>
#include <tuple>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace geometry_ann;
namespace {
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
struct Temporary {
    std::filesystem::path path;
    Temporary() {
        auto name = (std::filesystem::temp_directory_path() / "geometry-mmap-XXXXXX").string();
        if (!mkdtemp(name.data())) throw std::runtime_error("mkdtemp failed");
        path = name;
    }
    ~Temporary() { std::error_code error; std::filesystem::remove_all(path, error); }
};
template<class T> void records(const std::filesystem::path& path, const std::vector<T>& values, std::int32_t d) {
    std::ofstream out(path, std::ios::binary);
    for (std::size_t i = 0; i < values.size(); i += d) {
        out.write(reinterpret_cast<const char*>(&d), 4);
        out.write(reinterpret_cast<const char*>(values.data() + i), d * sizeof(T));
    }
}
}
int main() {
    try {
        Temporary temp;
        std::vector<float> base(36), query = {1.3f, 2.1f, 3.2f};
        for (std::size_t i = 0; i < base.size(); ++i) base[i] = float(i) / 7;
        std::vector<Id> truth = {0,1,2,3,4,5,6,7,8,9};
        records(temp.path / "base.fvecs", base, 3);
        records(temp.path / "q.fvecs", query, 3);
        records(temp.path / "truth.ivecs", truth, 10);
        { std::ofstream file(temp.path / "base bytes.bin", std::ios::binary);
          file.write("123456789012", 12); file.write(reinterpret_cast<const char*>(base.data()), base.size() * 4); }
        auto view = temp.path / "base.rawview";
        auto descriptor = [&](std::size_t n, std::size_t d, std::size_t offset) {
            std::ofstream(view) << "geometry_raw_f32_v1 " << n << ' ' << d << ' ' << offset << ' '
                                << std::quoted("base bytes.bin") << '\n';
        };
        descriptor(12, 3, 12);
        auto load = [&](const auto& path) { return load_dataset(path, temp.path / "q.fvecs", temp.path / "truth.ivecs", 1); };
        auto mapped = load(view), legacy = load(temp.path / "base.fvecs");
        auto moved = std::move(mapped);
        require(moved.base_storage.empty() && moved.base_mapping, "mapped input was copied");
        require(std::equal(moved.base.begin(), moved.base.end(), base.begin(), base.end()), "mapped values changed");
        BuildOptions build; build.coarse = 2; build.children = 2; build.leaf_budget = 4;
        Index a(moved.base, 3, build), b(legacy.base, 3, build);
        SearchOptions search; search.routing = Routing::flat; search.leaf_scans = 4;
        auto actual = a.search(query, search), expected = b.search(query, search);
        require(actual.neighbors.size() == expected.neighbors.size(), "mapped result count changed");
        for (std::size_t i = 0; i < actual.neighbors.size(); ++i)
            require(actual.neighbors[i].id == expected.neighbors[i].id && actual.neighbors[i].distance == expected.neighbors[i].distance,
                    "mapped index differs from vecs index");
        moved.release_base();
        require(moved.base.empty() && !moved.base_mapping && moved.count() == 12, "base release lost row count");
        for (auto [n,d,offset] : {std::tuple<std::size_t,std::size_t,std::size_t>{13,3,12}, {12,3,2}, {0,3,12}}) {
            descriptor(n,d,offset); bool rejected = false;
            try { load(view); } catch (const std::runtime_error&) { rejected = true; }
            require(rejected, "invalid mapped view accepted");
        }
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
