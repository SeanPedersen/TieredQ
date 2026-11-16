"""Summarizes the #55 gate into REPORT.md: kernel scan times, then per dataset and kind (flat, nested) whether builds
wrote identical files, build time and storage per ABBA build, whether every serve returned the ids and scores of the
first A serve, and per nprobe single-query QPS / p50 / p95 and batch QPS of B over A (each the mean of its two ABBA
serves, with the A-A spread as the noise reference), plus serving RSS."""
import json
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
ABBA = ("1A", "2B", "3B", "4A")


def load(path):
    return json.loads(path.read_text())


def change(b, a):
    return f"{b / a - 1:+.1%}"


def mean(values):
    return sum(values) / len(values)


lines = ["# #55 non-regression gate: aae87c7 (A) vs fba7ef5 (B)", ""]
lines += ["## Kernel scans (whole-run block, ns per row, CPU 2)", "", "| model | A 1 | B 2 | B 3 | A 4 | B / A | B tiles |",
          "|---|---|---|---|---|---|---|"]
for model in ("768-175", "1024-100"):
    runs = {tag: json.loads((RESULTS / f"kernel-{model}-{tag}/run.log").read_text().splitlines()[-1]) for tag in ABBA}
    a = mean([runs["1A"]["block_ns_per_row"], runs["4A"]["block_ns_per_row"]])
    b = mean([runs["2B"]["block_ns_per_row"], runs["3B"]["block_ns_per_row"]])
    tiles = mean([runs["2B"]["tile_ns_per_row"], runs["3B"]["tile_ns_per_row"]])
    lines.append(f"| {model} | " + " | ".join(f"{runs[t]['block_ns_per_row']:.2f}" for t in ABBA) + f" | {change(b, a)} | {tiles:.2f} |")

for dataset in ("arxiv", "qwen"):
    for kind in ("flat", "nested"):
        lines += ["", f"## {dataset} {kind}", ""]
        if kind == "flat":
            builds = {t: RESULTS / f"{dataset}-flat-build-{t}" for t in ABBA}
        else:
            builds = {v: RESULTS / f"{dataset}-nested-build-{v}" for v in "AB"}
        if not all((p / "build.json").exists() for p in builds.values()):
            lines.append("(incomplete)")
            continue
        hashes = {t: tuple(v["sha256"] for v in load(p / "files.json").values()) for t, p in builds.items()}
        reference = next(iter(hashes.values()))
        lines.append(f"Saved files identical across builds: {all(h == reference for h in hashes.values())}")
        lines += ["", "| build | build s | tier s | storage MiB | peak RSS GiB |", "|---|---|---|---|---|"]
        for tag, path in builds.items():
            b = load(path / "build.json")
            lines.append(f"| {tag} | {b['build_s']:.1f} | {b['tier_s']:.1f} | {b['storage_bytes'] / 2**20:.1f} | "
                         f"{b['peak_rss_bytes'] / 2**30:.2f} |")
        serves = {t: RESULTS / f"{dataset}-{kind}-serve-{t}" for t in ABBA}
        if not all((p / "serve.json").exists() for p in serves.values()):
            lines.append("(serves incomplete)")
            continue
        arrays = {t: np.load(p / "serve.npz") for t, p in serves.items()}
        first = arrays["1A"]
        same = {t: all(np.array_equal(a[key], first[key]) for key in first.files) for t, a in arrays.items()}
        lines += ["", "Ids and scores identical to serve 1A: " + ", ".join(f"{t} {same[t]}" for t in ABBA)]
        reports = {t: load(p / "serve.json") for t, p in serves.items()}
        lines += ["", "| nprobe | single QPS A | B | B / A | A-A spread | p50 us A | B | p95 us A | B |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for i, point in enumerate(reports["1A"]["points"]):
            by = {v: [reports[t]["points"][i] for t in ABBA if t.endswith(v)] for v in "AB"}
            qps = {v: mean([p["single_qps"] for p in by[v]]) for v in "AB"}
            p50 = {v: mean([p["p50_us"] for p in by[v]]) for v in "AB"}
            p95 = {v: mean([p["p95_us"] for p in by[v]]) for v in "AB"}
            spread = abs(by["A"][0]["single_qps"] / by["A"][1]["single_qps"] - 1)
            lines.append(f"| {point['nprobe']} | {qps['A']:.0f} | {qps['B']:.0f} | {change(qps['B'], qps['A'])} | {spread:.1%} | "
                         f"{p50['A']:.0f} | {p50['B']:.0f} | {p95['A']:.0f} | {p95['B']:.0f} |")
        batch = {v: mean([reports[t]["batch_qps"] for t in ABBA if t.endswith(v)]) for v in "AB"}
        rss = {v: mean([reports[t]["served_rss_bytes"] for t in ABBA if t.endswith(v)]) for v in "AB"}
        a_spread = abs(reports["1A"]["batch_qps"] / reports["4A"]["batch_qps"] - 1)
        lines += ["", f"Batch QPS (32 threads, nprobe {reports['1A']['batch_nprobe']}): A {batch['A']:.0f}, B {batch['B']:.0f} "
                  f"({change(batch['B'], batch['A'])}, A-A spread {a_spread:.1%}). Serving RssAnon: A {rss['A'] / 2**30:.3f} GiB, "
                  f"B {rss['B'] / 2**30:.3f} GiB."]
        for extra in sorted(RESULTS.glob(f"{dataset}-{kind}-serve-own-B")):
            own = np.load(extra / "serve.npz")
            lines.append(f"B serving its own build: ids and scores identical to 1A: "
                         f"{all(np.array_equal(own[key], first[key]) for key in first.files)}")
Path("REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
