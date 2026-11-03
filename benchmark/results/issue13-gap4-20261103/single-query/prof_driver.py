"""Single-query (1 thread, batch 1) loop over a flat TieredQ index for perf: prints 'profile ready' after the build."""
import sys, time
from pathlib import Path
import numpy as np
import codec_native
from data import prepare
dataset, data, code_bits, rounds = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
base, queries, truth = prepare(data, dataset)
import os
base = np.ascontiguousarray(base[:int(os.environ.get("DOCS", len(base)))])
index = codec_native.Index(base, -(-code_bits // base.shape[1]), 0, True, False, True, code_bits, True)
ids = np.empty((1, 10), dtype=np.uint32)
print("profile ready", file=sys.stderr, flush=True)
start = time.perf_counter()
for _ in range(rounds):
    for q in range(len(queries)):
        index.search(queries[q:q + 1], ids, 1)
seconds = time.perf_counter() - start
print(f"{dataset} {code_bits} bits: {seconds / (rounds * len(queries)) * 1e3:.3f} ms/query", flush=True)
