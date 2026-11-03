"""Summarise issue #13 nibble pruning-port rows: median over rounds per set/budget/nprobe/arm."""
import csv, statistics, sys
from collections import defaultdict
ARMS = ["exact-z0", "exact-z8", "nibble-off", "nibble-z0", "nibble-z8"]
cells = defaultdict(list)
for row in csv.reader(open(sys.argv[1]), delimiter="\t"):
    s, arm, _, budget, nlist, nprobe = row[:6]
    dim = int(row[18]) // 4
    cells[(s, int(float(budget)) // dim, int(nprobe), arm)].append((float(row[13]), float(row[14]), float(row[10])))
keys = sorted({k[:3] for k in cells}, key=lambda k: (["openai100k", "gemini", "msmarco", "openai1m"].index(k[0]), k[1], k[2]))
print("| Set, bits/dim, nprobe | " + " | ".join(ARMS) + " |")
print("|---" * (len(ARMS) + 1) + "|")
for k in keys:
    out = []
    for a in ARMS:
        v = cells[k + (a,)]
        q1, q8, r = (statistics.median(x[i] for x in v) for i in range(3))
        out.append(f"{q1:,.0f} / {q8:,.0f}, {r:.4f}")
    print(f"| {k[0]}, {k[1]}, {k[2]} | " + " | ".join(out) + " |")
