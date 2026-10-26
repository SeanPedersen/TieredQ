"""Learned tiers and memory split (resident vs serving model) per dataset and budget, seed 0; run from benchmark/."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from threadpoolctl import threadpool_limits
from data import prepare, codec_vectors

sys.path.insert(0, "build-native")
import codec_native

rows = []
with threadpool_limits(limits=1):
    for dataset in ("sift", "glove", "openai"):
        base, queries, _ = prepare(Path("data"), dataset, None, 1000)
        vectors, _ = codec_vectors(base, queries, dataset == "sift")
        for bits in range(1, 9):
            codec_native.set_threads(8)
            index = codec_native.Index(vectors, bits, 0, True, dataset == "sift")
            layout = index.diagnostics()
            with tempfile.TemporaryDirectory() as temporary:
                storage = index.storage(str(Path(temporary) / "index.bin"))
            row = dict(dataset=dataset, bits=bits, n=len(vectors), storage_bytes_per_vector=storage / len(vectors),
                       tiers=[(t["bits"], t["dimensions"]) for t in layout["tiers"]],
                       **{k: v for k, v in layout.items() if k != "tiers"})
            print(json.dumps(row), flush=True)
            rows.append(row)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
