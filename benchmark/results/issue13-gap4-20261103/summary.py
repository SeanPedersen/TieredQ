"""Issue #13 4-bit gap: median over rounds per variant at its equal-recall budget, against the RaBitQ 4 bits/dim cells."""
import collections, glob, json, statistics
# RaBitQ 4 bits/dim reference cells (issue13-refs-20261102/final/tables.md): stored bits, QPS 8t.b32, QPS 1t.b32, p50 1t.b1 ms.
RABITQ = {"openai": (6305, 3917, 539, 1.83), "openai1m": (6304, 375, 51, 19.27)}
cells = collections.defaultdict(lambda: collections.defaultdict(list))
for path in glob.glob("timing/r*/*/*.json"):
    _, _, variant, name = path.split("/"); dataset, code = name[5:-5].rsplit("-c", 1)
    for c in json.load(open(path)):
        mode = f"{c['threads']}t.b{c['batch']}"
        cells[(dataset, variant, int(code))][mode].append(c["p50_ms"] if mode == "1t.b1" else c["qps"])
        cells[(dataset, variant, int(code))]["bits"].append(8 * c["effective_bytes_per_vector"])
        cells[(dataset, variant, int(code))]["recall"].append(c["recall10"])
print("| Set | Variant | Code bits | R@10 s0 | Stored bits TQ / RaBitQ | QPS 8t.b32 TQ / ref | QPS 1t.b32 TQ / ref | p50 1t.b1 ms TQ / ref |")
print("|---|---|---:|---:|---|---|---|---|")
for (dataset, variant, code) in sorted(cells, key=lambda k: (k[0], "base mb6 mb5 f8".split().index(k[1]))):
    m = {k: statistics.median(v) for k, v in cells[(dataset, variant, code)].items()}
    ref = RABITQ[dataset]
    cols = [(m["bits"], ref[0], 0), (m["8t.b32"], ref[1], 0), (m["1t.b32"], ref[2], 0), (m["1t.b1"], ref[3], 2)]
    print(f"| {dataset} | {variant} | {code} | {m['recall']:.4f} | " + " | ".join(f"{a:,.{d}f} / {b:,.{d}f} ({a / b:.2f}x)" for a, b, d in cols) + " |")
