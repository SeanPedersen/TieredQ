"""Builds the low-RAM nested index through the Python API with its defaults only, from the memory-mapped SQ8 store:
TieredQNestedIVFIndex.build(codes=, scales=, offsets=) and write_tier(path, codes=, ...) to the NVMe SSD, then save().
Usage: build.py BASE.i8 SCALES.f32 OFFSETS.f32 DIMENSIONS INDEX.tq TIER.rows OUT.json [OPTIONS_JSON]
OPTIONS_JSON adds build() keywords (empty: defaults only)."""
import json, resource, sys, time
import numpy as np
from tieredq import TieredQNestedIVFIndex

base, scales_path, offsets_path, dimensions, index_path, tier_path, out = sys.argv[1:8]
options = json.loads(sys.argv[8]) if len(sys.argv) > 8 else {}
dimensions = int(dimensions)
codes = np.memmap(base, dtype=np.int8, mode='r').reshape(-1, dimensions)
scales, offsets = np.fromfile(scales_path, dtype=np.float32), np.fromfile(offsets_path, dtype=np.float32)
store = dict(codes=codes, scales=scales, offsets=offsets)

started = time.perf_counter()
index = TieredQNestedIVFIndex.build(**store, **options)
built = time.perf_counter()
index.write_tier(tier_path, **store)
tiered = time.perf_counter()
index.save(index_path)
saved = time.perf_counter()
json.dump(dict(options=options, rows=len(index), lists=index.lists, has_router=index.has_router, prefix_bits=index.prefix_bits,
               total_bits=index.total_bits, tier_rows=index.tier_rows, tier_bytes_per_vector=index.tier_bytes_per_vector,
               storage_bytes=index.storage_bytes, build_s=built - started, tier_s=tiered - built, save_s=saved - tiered,
               total_s=tiered - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
          open(out, 'w'), indent=2)
