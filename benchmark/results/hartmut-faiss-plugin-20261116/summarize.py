"""Summarizes the #55 acceptance study into REPORT.md.
- Parity: per dataset, seed and nprobe, queries (of 50) whose (id, score) sets and score lists equal TieredQIVFIndex's.
- Plugin against TieredQIVFIndex: measured RAM ratio, and per TieredQ point the plugin's QPS at TieredQ's recall
  (log-QPS interpolated along the plugin's own curve), per seed.
- Every index: RAM, recall ceiling, and VIBE-style best QPS and QPS/GiB among the settings reaching each target recall
  (the #41 metric), as the mean over seeds with the seed range."""
import json
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
# Optional: further studies' results on the same partitions (TieredQ and plugin at other budgets, more Faiss codecs).
EXTRA = [Path(arg) for arg in sys.argv[2:]]
TARGETS = {"arxiv": (0.85, 0.90, 0.95), "qwen": (0.80, 0.85, 0.90, 0.95)}
DEFAULT_BITS = {"arxiv": "1.75", "qwen": "1"}
GIB = 2**30


def load(path):
    return json.loads(path.read_text())


def interpolate_qps(points, recall):
    """QPS at a recall along a curve (log-QPS linear in recall); None outside it."""
    ordered = sorted(points, key=lambda p: p["recall"])
    for low, high in zip(ordered, ordered[1:]):
        if low["recall"] <= recall <= high["recall"] and high["recall"] > low["recall"]:
            t = (recall - low["recall"]) / (high["recall"] - low["recall"])
            return float(np.exp((1 - t) * np.log(low["qps"]) + t * np.log(high["qps"])))
    exact = [p["qps"] for p in ordered if p["recall"] == recall]
    return exact[0] if exact else None


def best_reaching(points, ram, target):
    reaching = [p for p in points if p["recall"] >= target]
    if not reaching:
        return None
    best = max(reaching, key=lambda p: p["qps"])
    return best["qps"], best["qps"] / (ram / GIB), best["nprobe"]


def spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return "—"
    mean = np.mean(values)
    if len(values) == 1:
        return f"{mean:,.0f}"
    return f"{mean:,.0f} ({min(values):,.0f}–{max(values):,.0f})"


def frontier(dataset, curves):
    """Every index of both studies at each target: best QPS/GiB, and the lowest RAM that reaches the target."""
    entries = {}
    for name, reports in curves.items():
        entries[f"{name} {DEFAULT_BITS[dataset]}b" if name in ("tieredq", "plugin") else name] = reports
    out = []
    control = EXTRA[0] / f"{dataset}-b{DEFAULT_BITS[dataset]}-s1"
    if (control / "serve-tieredq/serve.json").exists():
        rerun, base = load(control / "serve-tieredq/serve.json"), load(RESULTS / f"{dataset}-s1/serve-tieredq/serve.json")
        same = [p["recall"] for p in rerun["points"] if p["nprobe"] in {q["nprobe"] for q in base["points"]}] == \
               [p["recall"] for p in base["points"] if p["nprobe"] in {q["nprobe"] for q in rerun["points"]}]
        out += [f"Partition control ({control.name} against {dataset}-s1): equal TieredQ recall at every shared nprobe: {same}", ""]
    runs = [r for extra in EXTRA for r in extra.glob(f"{dataset}-b*-s*")]
    for run in runs:
        bits = run.name.split("-b")[1].split("-s")[0]
        if bits == DEFAULT_BITS[dataset]:
            continue  # the partition control repeats the base study
        for serve in run.glob("serve-*/serve.json"):
            name = serve.parent.name.removeprefix("serve-")
            entries.setdefault(f"{name} {bits}b" if name in ("tieredq", "plugin") else name, []).append(load(serve))
    targets = TARGETS[dataset]
    out += [f"### Recall-per-RAM frontier with the high-bit runs ({dataset}; mean over seeds)", "",
            "| index | seeds | RAM GiB | ceiling | " + " | ".join(f"r ≥ {t}: QPS / QPS/GiB" for t in targets) + " |",
            "|---|---|---|---|" + "---|" * len(targets)]
    lowest = {t: None for t in targets}
    for name, reports in sorted(entries.items(), key=lambda e: np.mean([r["ram_bytes"] for r in e[1]])):
        ram = np.mean([r["ram_bytes"] for r in reports]) / GIB
        cells = []
        for target in targets:
            found = [best_reaching(r["points"], r["ram_bytes"], target) for r in reports]
            if any(f is None for f in found):
                cells.append("—")
                continue
            cells.append(f"{np.mean([f[0] for f in found]):,.0f} / {np.mean([f[1] for f in found]):,.0f}")
            if lowest[target] is None or ram < lowest[target][1]:
                lowest[target] = (name, ram, np.mean([f[0] for f in found]))
        ceiling = np.mean([max(p["recall"] for p in r["points"]) for r in reports])
        out.append(f"| {name} | {len(reports)} | {ram:.3f} | {ceiling:.3f} | " + " | ".join(cells) + " |")
    out += ["", "Lowest RAM reaching each target: " + "; ".join(
        f"r ≥ {t}: " + ("none" if lowest[t] is None else f"{lowest[t][0]} ({lowest[t][1]:.3f} GiB, {lowest[t][2]:,.0f} QPS)")
        for t in targets), ""]
    return out


lines = ["# #55 acceptance: IndexIVFTieredQ against TieredQIVFIndex and Faiss's codecs", ""]
for dataset in ("arxiv", "qwen"):
    seeds = sorted(p for p in RESULTS.glob(f"{dataset}-s*") if (p / "serve-tieredq/serve.json").exists())
    if not seeds:
        continue
    lines += [f"## {dataset} ({len(seeds)} partition seeds)", ""]
    builds = {s.name: load(s / "build/build.json") for s in seeds}
    first = builds[seeds[0].name]
    lines.append(f"Lists {first['lists']}, TieredQ code bits {first['code_bits']}. Build seconds per seed (k-means, TieredQ, "
                 f"plugin add): " + "; ".join(f"{b['kmeans_s']:.0f} / {b['tieredq_build_s']:.0f} / {b['plugin_add_s']:.0f}"
                                              for b in builds.values()))
    lines += ["", "### Parity with TieredQIVFIndex (exact coarse ranking; queries of 50 with equal ids+scores / scores)", "",
              "| seed | " + " | ".join(f"np {p['nprobe']}" for p in load(seeds[0] / "parity/parity.json")["points"]) + " |",
              "|---|" + "---|" * len(load(seeds[0] / "parity/parity.json")["points"])]
    for s in seeds:
        parity = load(s / "parity/parity.json")["points"]
        lines.append(f"| {s.name} | " + " | ".join(f"{p['same_hits']} / {p['same_scores']}" for p in parity) + " |")
    lines += ["", "### Plugin against TieredQIVFIndex (single core)", "",
              "| seed | RAM TieredQ GiB | plugin GiB | ratio | plugin QPS / TieredQ QPS at TieredQ's recall, per nprobe |",
              "|---|---|---|---|---|"]
    for s in seeds:
        tieredq, plugin = load(s / "serve-tieredq/serve.json"), load(s / "serve-plugin/serve.json")
        ratios = []
        for point in tieredq["points"]:
            qps = interpolate_qps(plugin["points"], point["recall"])
            ratios.append(f"np{point['nprobe']} r{point['recall']:.3f}: " + ("—" if qps is None else f"{qps / point['qps']:.3f}"))
        lines.append(f"| {s.name} | {tieredq['ram_bytes'] / GIB:.3f} | {plugin['ram_bytes'] / GIB:.3f} | "
                     f"{plugin['ram_bytes'] / tieredq['ram_bytes']:.3f} | " + ", ".join(ratios) + " |")
    lines += ["", "### Same-nprobe curves (seed means)", "", "| index | RAM GiB | " +
              " | ".join(f"np {p['nprobe']} recall / QPS" for p in load(seeds[0] / "serve-tieredq/serve.json")["points"]) + " |",
              "|---|---|" + "---|" * len(load(seeds[0] / "serve-tieredq/serve.json")["points"])]
    names = sorted({p.name.removeprefix("serve-") for s in seeds for p in s.glob("serve-*")},
                   key=lambda n: ({"tieredq": 0, "plugin": 1}.get(n, 2), n))
    curves = {}
    for name in names:
        reports = [load(s / f"serve-{name}/serve.json") for s in seeds if (s / f"serve-{name}/serve.json").exists()]
        curves[name] = reports
        cells = []
        for i in range(len(reports[0]["points"])):
            recall = np.mean([r["points"][i]["recall"] for r in reports]); qps = np.mean([r["points"][i]["qps"] for r in reports])
            cells.append(f"{recall:.3f} / {qps:,.0f}")
        lines.append(f"| {name} | {np.mean([r['ram_bytes'] for r in reports]) / GIB:.3f} | " + " | ".join(cells) + " |")
    lines += ["", "### Best QPS and QPS/GiB at fixed recall (#41; mean over seeds, range in parentheses)", "",
              "| index | RAM GiB | ceiling | " + " | ".join(f"r ≥ {t}: QPS | QPS/GiB" for t in TARGETS[dataset]) + " |",
              "|---|---|---|" + "---|---|" * len(TARGETS[dataset])]
    for name, reports in curves.items():
        ceiling = np.mean([max(p["recall"] for p in r["points"]) for r in reports])
        cells = []
        for target in TARGETS[dataset]:
            found = [best_reaching(r["points"], r["ram_bytes"], target) for r in reports]
            if any(f is None for f in found):
                cells += ["—", "—"]
                continue
            cells += [spread([f[0] for f in found]), spread([f[1] for f in found])]
        lines.append(f"| {name} | {np.mean([r['ram_bytes'] for r in reports]) / GIB:.3f} | {ceiling:.3f} | " + " | ".join(cells) + " |")
    lines.append("")
    if EXTRA:
        lines += frontier(dataset, curves)
Path("REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
