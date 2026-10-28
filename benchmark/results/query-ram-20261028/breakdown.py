"""Component allocation breakdown of reloaded TieredQ indexes (diagnostic tree only)."""
import json, sys
from pathlib import Path
import codec_native
from data import prepare
out = {}
for dataset, bits in [("openai", 1), ("gemini", 1), ("glove", 1), ("openai", 2), ("glove", 4)]:
    base, _, _ = prepare(Path(sys.argv[1]), dataset)
    codec_native.set_threads(8)
    index = codec_native.Index(base, bits, 0, True, False)
    a = index.allocations(f"/tmp/breakdown-{dataset}.bin")
    n = len(base); scale = 8 / n
    out[f"{dataset}-{bits}"] = a
    print(dataset, bits, "disk", round(a["serialized_bytes"]*scale,1), "resident", round(a["resident_bytes"]*scale,1))
    for part in ("model", "cache"):
        print("  ", part, {k: round(v*scale, 1) for k, v in sorted(a[part].items(), key=lambda kv: -kv[1]) if v*scale >= 0.5})
    Path(f"/tmp/breakdown-{dataset}.bin").unlink()
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
