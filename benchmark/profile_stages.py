"""Optional phase counters injected only into a separate diagnostic source/build tree."""

COUNTERS = ("projection", "centroid", "lut", "scan", "selection")
HEADER = r'''
namespace bench_profile {
inline std::atomic<double> projection_s{0}, centroid_s{0}, lut_s{0}, scan_s{0}, selection_s{0};
class Phase {
  std::atomic<double>& counter;
  Span::Clock::time_point start=Span::Clock::now();
  bool active=true;
 public:
  explicit Phase(std::atomic<double>& value):counter(value) {}
  void stop() {
    if (!active) return;
    counter.fetch_add(std::chrono::duration<double>(Span::Clock::now()-start).count());
    active=false;
  }
  ~Phase() { stop(); }
};
}
'''


def instrument(source, module, header, replace_once):
    header.write_text(header.read_text() + HEADER)
    path = source / "src/tieredq.cpp"
    text = replace_once(path.read_text(),
        "void TieredQ::project_query_tile(const float* vectors,size_t rows,TieredQQuery* out) const {",
        "void TieredQ::project_query_tile(const float* vectors,size_t rows,TieredQQuery* out) const {\n"
        "  bench_profile::Phase projection(bench_profile::projection_s);")
    text = replace_once(text, "  std::vector<float> lanes(d*query_weight_tile,0);",
        "  projection.stop();\n  bench_profile::Phase centroid(bench_profile::centroid_s);\n"
        "  std::vector<float> lanes(d*query_weight_tile,0);")
    text = replace_once(text, "  auto q=query_weights(vector);",
        "  auto q=query_weights(vector);\n  bench_profile::Phase lut(bench_profile::lut_s);")
    path.write_text(f'#include "{header}"\n' + text)
    path = source / "src/tieredq_scan_lookup_x86.h"
    text = replace_once(path.read_text(), "  const auto output=project_fast_query(cache.projection,vector);",
        "  bench_profile::Phase projection(bench_profile::projection_s);\n"
        "  const auto output=project_fast_query(cache.projection,vector);\n"
        "  projection.stop();\n  bench_profile::Phase lut(bench_profile::lut_s);")
    path.write_text(f'#include "{header}"\n' + text)
    path = source / "src/tieredq_search_fast.h"
    text = replace_once(path.read_text(),
        "        scan(queries.data()+first,rows,begin,count,scratch.data(),stride);",
        "        { bench_profile::Phase scan_phase(bench_profile::scan_s);\n"
        "          scan(queries.data()+first,rows,begin,count,scratch.data(),stride); }\n"
        "        bench_profile::Phase selection_phase(bench_profile::selection_s);")
    text = replace_once(text,
        "      for (size_t row=0;row<rows;++row) partial[(first+row)*segments+segment]=top[row].finish();",
        "      { bench_profile::Phase selection_phase(bench_profile::selection_s);\n"
        "        for (size_t row=0;row<rows;++row) partial[(first+row)*segments+segment]=top[row].finish(); }")
    path.write_text(f'#include "{header}"\n' + text)
    reset = " ".join(f"bench_profile::{name}_s.store(0);" for name in COUNTERS)
    module = replace_once(module, "bench_profile::total_s.store(0);", "bench_profile::total_s.store(0); " + reset)
    reads = ", ".join(f"bench_profile::{name}_s.load()" for name in COUNTERS)
    return replace_once(module, "bench_profile::total_s.load());", "bench_profile::total_s.load(), " + reads + ");")
