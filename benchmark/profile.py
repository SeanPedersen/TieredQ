"""Separate instrumented build measures native preprocessing without perturbing timings."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROFILE_QUERY_LIMIT = 32
MILLISECONDS_PER_SECOND = 1000
HEADER = r'''
// Diagnostic counters live only in the separately built profiling extension.
#pragma once
#include <atomic>
#include <chrono>
namespace bench_profile {
inline std::atomic<double> prep_s{0}, total_s{0};
struct Span {
  using Clock = std::chrono::steady_clock;
  Clock::time_point start = Clock::now();
  void prepared() { prep_s.fetch_add(std::chrono::duration<double>(Clock::now()-start).count()); }
  ~Span() { total_s.fetch_add(std::chrono::duration<double>(Clock::now()-start).count()); }
};
}
'''


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Profiling patch context changed: {old}")
    return text.replace(old, new)


def build(full_score, stages=False):
    root = HERE / ".deps" / ("profile-stages" if stages else "profile")
    source, native = root/"source", root/"native"
    native.mkdir(parents=True, exist_ok=True)
    header = native/"profile.h"
    header.write_text(HEADER)
    for name in ("src", "include", "cmake"):
        shutil.copytree(HERE.parent/name, source/name, dirs_exist_ok=True)
    shutil.copyfile(HERE.parent/"CMakeLists.txt", source/"CMakeLists.txt")
    path = source/"src/tieredq_index.cpp"
    text = replace_once(path.read_text(), "[[maybe_unused]] const auto backend=", "bench_profile::Span profile;\n  [[maybe_unused]] const auto backend=")
    # Each dispatch branch measures the actual projection/rotation/LUT path it executes.
    text = text.replace("    tieredq_detail::search_fast_tiles(prepared", "    profile.prepared();\n    tieredq_detail::search_fast_tiles(prepared")
    text = text.replace("  tieredq_detail::search_fast_tiles(prepared,cache.count,k,1", "  profile.prepared();\n  tieredq_detail::search_fast_tiles(prepared,cache.count,k,1")
    path.write_text(f'#include "{header}"\n' + text)
    rq = root/"rabitq"
    from setup_adapters import prepare
    overlay = prepare(HERE/".deps", full_score=full_score)
    shutil.copytree(overlay/"include", rq/"include", dirs_exist_ok=True)
    flat = (HERE/"codec_adapters/rabitq/rabitq_flat.h").read_text()
    flat = replace_once(flat, "void search(const float* vector,size_t k,uint32_t* ids) const {", "void search(const float* vector,size_t k,uint32_t* ids) const {\n    bench_profile::Span profile;")
    flat = replace_once(flat, "query.set_g_add(distance,center_ip);", "query.set_g_add(distance,center_ip);\n    profile.prepared();")
    (native/"rabitq").mkdir(exist_ok=True)
    (native/"rabitq/rabitq_flat.h").write_text(f'#include "{header}"\n' + flat)
    shutil.copyfile(HERE/"codec_adapters/rabitq/reference.cpp",native/"rabitq/reference.cpp")
    module = (HERE/"codec_adapters/module.cpp").read_text()
    module = replace_once(module, "#include <nanobind/nanobind.h>", f'#include <nanobind/nanobind.h>\n#include "{header}"')
    module = replace_once(module, 'NB_MODULE(codec_native, m) {', '''NB_MODULE(codec_native, m) {
  m.def("profile_reset", []{ bench_profile::prep_s.store(0); bench_profile::total_s.store(0); });
  m.def("profile_read", []{ return nb::make_tuple(bench_profile::prep_s.load(), bench_profile::total_s.load()); });''')
    if stages:
        from profile_stages import instrument
        module = instrument(source, module, header, replace_once)
    (native/"module.cpp").write_text(module)
    cmake = (HERE/"codec_adapters/CMakeLists.txt").read_text().replace("add_subdirectory(../.. tieredq)", f'add_subdirectory("{source}" tieredq)')
    (native/"CMakeLists.txt").write_text(cmake)
    build_dir = root/"build"
    subprocess.run(["cmake", "-S", str(native), "-B", str(build_dir), f"-DRABITQ_SOURCE={rq}", f"-DBENCH_FULL_SCORE={'ON' if full_score else 'OFF'}",
                    f"-DPython_EXECUTABLE={sys.executable}", "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "-j", "8"], check=True)
    from build_policy import audit_compile_commands
    audit_compile_commands(json.loads((build_dir/"compile_commands.json").read_text()))
    return build_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data", type=Path, default=HERE/"data")
    parser.add_argument("--legacy-l2-lift", action="store_true")
    parser.add_argument("--stages", action="store_true", help="Add intrusive worker-time counters for TieredQ scan and selection")
    args = parser.parse_args()
    with (args.output/"results.csv").open() as handle:
        results = [r for r in csv.DictReader(handle) if int(r["seed"]) == 0]
    build_dir = build(any("full-score" in r["config"] for r in results), stages=args.stages)
    sys.path.insert(0, str(build_dir))
    os.environ["BENCH_SEED"] = "0"
    import codec_native
    import numpy as np
    from threadpoolctl import threadpool_limits
    from data import prepare, codec_vectors
    from measure import L2Search, measure, REPEATS, WARMUPS
    from l2_transform import L2Transform
    rows = []
    with threadpool_limits(limits=1):
        for dataset in sorted({r["dataset"] for r in results}):
            selected = [r for r in results if r["dataset"] == dataset]
            base, queries, truth = prepare(args.data, dataset, int(selected[0]["base_n"]), int(selected[0]["queries"]))
            for codec in ("tieredq", "rabitq"):
                lift = dataset == "sift" and codec == "tieredq"
                transform = L2Transform.fit(base) if lift and not args.legacy_l2_lift else None
                vectors, probes = (transform.base_vectors(base), transform.queries(queries)) if transform else codec_vectors(base, queries, lift)
                for bits in sorted({int(r["target_bits"]) for r in selected}):
                    codec_native.set_threads(8)
                    index = codec_native.Index(vectors, bits, 0, codec == "tieredq", dataset == "sift")
                    search, q = (L2Search(index, float(np.max(np.linalg.norm(base, axis=1))), transform),queries) if lift else (index,probes)
                    for config in [r for r in selected if r["codec"] == codec and int(r["target_bits"]) == bits]:
                        codec_native.profile_reset()
                        measure(search, q[:PROFILE_QUERY_LIMIT], truth[:PROFILE_QUERY_LIMIT], int(config["threads"]), int(config["batch"]))
                        query_calls = REPEATS * min(PROFILE_QUERY_LIMIT, len(q)) + WARMUPS
                        counters = codec_native.profile_read()
                        prep, total = counters[:2]
                        rows.append(dict(dataset=dataset,codec=codec,bits=bits,threads=config["threads"],batch=config["batch"],
                                         native_preprocessing_ms=prep*MILLISECONDS_PER_SECOND/query_calls,
                                         native_total_ms=total*MILLISECONDS_PER_SECOND/query_calls, preprocessing_fraction=prep/total))
                        if args.stages:
                            from profile_stages import COUNTERS
                            rows[-1].update({f"{name}_worker_ms":value*MILLISECONDS_PER_SECOND/query_calls if codec == "tieredq" else None
                                             for name,value in zip(COUNTERS,counters[2:],strict=True)})
                    print(f"Profiled {dataset} {codec} {bits}-bit", flush=True)
    with (args.output/"preprocessing.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
