"""Saved-index SHA-256 and build peak for fixed configurations; compares training changes byte for byte."""
import hashlib, json, sys, tempfile, time
from pathlib import Path
import codec_native
from data import prepare
from measure_memory import reset_peak, memory_status
data, label = Path(sys.argv[1]), sys.argv[2]
for dataset, bits in [tuple(item.split(":")) for item in (sys.argv[3] if len(sys.argv) > 3 else "openai:1,gemini:2,glove:4").split(",")]:
    bits = int(bits)
    base, _, _ = prepare(data, dataset)
    codec_native.set_threads(8)
    baseline = reset_peak()
    start = time.perf_counter()
    index = codec_native.Index(base, bits, 0, True, False)
    build_s = time.perf_counter() - start
    peak = memory_status()[1] - baseline
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "index.bin"; index.save(str(path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(dict(label=label, dataset=dataset, bits=bits, sha256=digest[:16], build_peak_mib=round(peak/2**20, 1), build_s=round(build_s, 3))), flush=True)
