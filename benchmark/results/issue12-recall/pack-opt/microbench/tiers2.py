"""Learned tiers and memory split (resident vs serving model) per dataset and budget, seed 0; run from benchmark/."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from threadpoolctl import threadpool_limits
from data import prepare, codec_vectors
from l2_transform import L2Transform

sys.path.insert(0, "build-native")
import codec_native

rows = []
with threadpool_limits(limits=1):
    # Same inputs as run.py: SIFT uses the fitted L2 lift (its storage counts toward bytes per vector).
    for dataset in sys.argv[2:]:
        base, queries, _ = prepare(Path("data"), dataset, None, 1000)
        transform = L2Transform.fit(base) if dataset == "sift" else None
        vectors = transform.base_vectors(base) if transform else codec_vectors(base, queries, False)[0]
        for bits in range(1, 9):
            codec_native.set_threads(8)
            index = codec_native.Index(vectors, bits, 0, True, dataset == "sift")
            layout = index.diagnostics()
            with tempfile.TemporaryDirectory() as temporary:
                storage = index.storage(str(Path(temporary) / "index.bin"))
            row = dict(dataset=dataset, bits=bits, n=len(vectors), storage_bytes_per_vector=(storage + (transform.storage_bytes if transform else 0)) / len(vectors),
                       tiers=[(t["bits"], t["dimensions"]) for t in layout["tiers"]],
                       **{k: v for k, v in layout.items() if k != "tiers"})
            print(json.dumps(row), flush=True)
            rows.append(row)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
