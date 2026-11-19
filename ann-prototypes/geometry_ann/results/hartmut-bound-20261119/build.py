"""Builds the low-RAM nested index through the Python API from the memory-mapped SQ8 store, as #52's defaults study:
TieredQNestedIVFIndex.build(codes=, scales=, offsets=, seed=, coarse_training=) and write_tier(path, codes=, ...) to the
NVMe SSD, then save().
Usage: build.py BASE.i8 SCALES.f32 OFFSETS.f32 DIMENSIONS INDEX.tq TIER.rows OUT.json SEED [COARSE_TRAINING_JSON]
COARSE_TRAINING_JSON holds TieredQCoarseTraining fields (empty: the automatic list training)."""
import json, resource, sys, time
import numpy as np
from tieredq import TieredQCoarseTraining, TieredQNestedIVFIndex

base, scales_path, offsets_path, dimensions, index_path, tier_path, out, seed = sys.argv[1:9]
training = json.loads(sys.argv[9]) if len(sys.argv) > 9 else {}
dimensions = int(dimensions)
codes = np.memmap(base, dtype=np.int8, mode='r').reshape(-1, dimensions)
scales, offsets = np.fromfile(scales_path, dtype=np.float32), np.fromfile(offsets_path, dtype=np.float32)
store = dict(codes=codes, scales=scales, offsets=offsets)
options = dict(seed=int(seed), **(dict(coarse_training=TieredQCoarseTraining(**training)) if training else {}))

started = time.perf_counter()
index = TieredQNestedIVFIndex.build(**store, **options)
built = time.perf_counter()
index.write_tier(tier_path, **store)
tiered = time.perf_counter()
index.save(index_path)
saved = time.perf_counter()
json.dump(dict(training=training, seed=int(seed), rows=len(index), lists=index.lists, has_router=index.has_router,
               prefix_bits=index.prefix_bits, total_bits=index.total_bits, storage_bytes=index.storage_bytes, build_stats=index.build_stats,
               build_s=built - started, tier_s=tiered - built, save_s=saved - tiered, total_s=tiered - started,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
          open(out, 'w'), indent=2)
