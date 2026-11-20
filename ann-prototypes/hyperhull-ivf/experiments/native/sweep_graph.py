"""Wall-time frontier of the C++ router with / without graph expansion (hullbench on one pinned core).

Usage: sweep_graph.py <hullbench> <export_dir> <label> [extra env K=V ...]
Prints the fastest config reaching each recall target, for H=0 (no graph) and H>0.
"""
import itertools
import os
import subprocess
import sys

TARGETS = (0.90, 0.95, 0.97, 0.99)
CORE = "20"


def configs() -> list[str]:
    out = []
    for T, F in [(16, 25), (16, 50), (32, 50), (32, 100), (64, 100), (64, 200), (128, 200), (128, 400), (256, 400),
                 (256, 800), (256, 1200), (256, 1792)]:
        for V, R in [(20, 150), (30, 200), (60, 400), (100, 800)]:
            out.append(f"{T},{F},{V},{R}")
            for H, P, K, E in itertools.product((1, 2), (5, 10, 20), (8, 16), (5, 10, 20)):
                if P * K <= 320:
                    out.append(f"{T},{F},{V},{R},0,{H},{P},{K},{E}")
    return out


def main() -> None:
    bench, export, label = sys.argv[1:4]
    env = {**os.environ, "HH_PREFIX_DIMS": "128", **dict(kv.split("=", 1) for kv in sys.argv[4:])}
    rows = []
    cfgs = configs()
    for s in range(0, len(cfgs), 200):
        out = subprocess.run(["taskset", "-c", CORE, bench, export, "2", *cfgs[s : s + 200]], env=env,
                             capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            parts = line.split()
            if parts and parts[0][0].isdigit() and "," in parts[0]:
                rows.append((parts[0], float(parts[1]), float(parts[2])))
        print(f"{label}: {len(rows)}/{len(cfgs)}", file=sys.stderr, flush=True)
    for graph in (False, True):
        pool = [r for r in rows if (r[0].count(",") > 3) == graph]
        line = []
        for t in TARGETS:
            ok = [r for r in pool if r[1] >= t]
            best = min(ok, key=lambda r: r[2]) if ok else None
            line.append(f"R≥{t}: {best[2]:6.1f} µs ({best[1]:.3f}, {best[0]})" if best else f"R≥{t}: -")
        print(f"{label:10s} {'graph' if graph else 'plain'}  " + "  ".join(line), flush=True)


if __name__ == "__main__":
    main()
