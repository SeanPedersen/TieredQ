"""Add phase counters only to an isolated copy of the measured codec snapshot."""
import argparse
from pathlib import Path

HEADER = r'''// Diagnostic phase counters; never compiled into the primary timing extension.
#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <vector>
namespace experiment_profile {
enum Stage { preparation, scan, selection, merge, count };
inline std::array<std::atomic<double>, count> seconds{};
inline std::atomic<bool> enabled{false};
inline void reset() { for (auto& x:seconds) x.store(0); enabled.store(true); }
inline std::vector<double> read() {
  enabled.store(false);
  std::vector<double> result;
  for (auto& x:seconds) result.push_back(x.load());
  return result;
}
class Phase {
  using Clock=std::chrono::steady_clock;
  Stage stage_;
  bool active_;
  Clock::time_point start_;
 public:
  explicit Phase(Stage stage):stage_(stage),active_(enabled.load()),start_(active_ ? Clock::now() : Clock::time_point{}) {}
  void stop() {
    if (!active_) return;
    seconds[stage_].fetch_add(std::chrono::duration<double>(Clock::now()-start_).count());
    active_=false;
  }
  ~Phase() { stop(); }
};
}
'''


def replace(text, old, new, expected=1):
    if text.count(old) != expected:
        raise ValueError(f'profiling context changed: {old!r}')
    return text.replace(old, new)


def instrument(root):
    header = (root / 'benchmark/experiments/native/profile.h').resolve()
    header.write_text(HEADER)
    path = root / 'src/tieredq_index.cpp'
    text = path.read_text()
    text = replace(text, '  [[maybe_unused]] const auto backend=',
                   '  experiment_profile::Phase preparation_phase(experiment_profile::preparation);\n  [[maybe_unused]] const auto backend=')
    text = replace(text, '    tieredq_detail::search_fast_tiles(prepared',
                   '    preparation_phase.stop();\n    tieredq_detail::search_fast_tiles(prepared', expected=2)
    text = replace(text, '  tieredq_detail::search_fast_tiles(prepared,cache.count,k,1',
                   '  preparation_phase.stop();\n  tieredq_detail::search_fast_tiles(prepared,cache.count,k,1')
    text = replace(text, '  auto prepared=prepare_search_queries(queries,rows,dimensions,\n      [&](const float* vector) { return codec_.query(vector); });',
                   '  experiment_profile::Phase preparation_phase(experiment_profile::preparation);\n  auto prepared=prepare_search_queries(queries,rows,dimensions,\n      [&](const float* vector) { return codec_.query(vector); });\n  preparation_phase.stop();')
    text = replace(text, '''        codec_.score_planes_batch(payload_.data()+doc*stride,n,prepared.data()+first,count,
                                  block_scores.data(),n,TieredQKernel::automatic,false);''',
                   '''        { experiment_profile::Phase scan_phase(experiment_profile::scan);
          codec_.score_planes_batch(payload_.data()+doc*stride,n,prepared.data()+first,count,
                                  block_scores.data(),n,TieredQKernel::automatic,false); }
        experiment_profile::Phase selection_phase(experiment_profile::selection);''')
    text = replace(text, '      for (size_t i=0;i<count;++i) {\n        const auto hits=heaps[i].finish();',
                   '      experiment_profile::Phase selection_phase(experiment_profile::selection);\n      for (size_t i=0;i<count;++i) {\n        const auto hits=heaps[i].finish();')
    text = replace(text, '    const size_t chunk_index=row/chunk,local=row%chunk;',
                   '    experiment_profile::Phase merge_phase(experiment_profile::merge);\n    const size_t chunk_index=row/chunk,local=row%chunk;')
    path.write_text(f'#include "{header}"\n'+text)
    path = root / 'src/tieredq_search_fast.h'
    text = path.read_text()
    text = replace(text, '        scan(queries.data()+first,rows,begin,count,scratch.data(),stride);',
                   '        { experiment_profile::Phase scan_phase(experiment_profile::scan);\n          scan(queries.data()+first,rows,begin,count,scratch.data(),stride); }\n        experiment_profile::Phase selection_phase(experiment_profile::selection);')
    text = replace(text, '      for (size_t row=0;row<rows;++row) partial[(first+row)*segments+segment]=top[row].finish();',
                   '      { experiment_profile::Phase selection_phase(experiment_profile::selection);\n        for (size_t row=0;row<rows;++row) partial[(first+row)*segments+segment]=top[row].finish(); }')
    text = replace(text, '  for (size_t row=0;row<queries.size();++row) {',
                   '  experiment_profile::Phase merge_phase(experiment_profile::merge);\n  for (size_t row=0;row<queries.size();++row) {')
    path.write_text(f'#include "{header}"\n'+text)
    print('Instrumented isolated diagnostic snapshot:', root, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    instrument(parser.parse_args().root)
