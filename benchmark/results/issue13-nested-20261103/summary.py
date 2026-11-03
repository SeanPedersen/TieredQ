"""Issue #13 option 2: flat nested search vs flat RaBitQ in ivf_eval (one list, nprobe 1); median over timing rounds.
Stored bits: prefix IVF (disk_bits) plus the suffix-only RAM tier. Usage: summary.py TIME_TSV"""
import collections, csv, statistics, sys
rows = collections.defaultdict(list)
for r in csv.reader(open(sys.argv[1]), delimiter="\t"):
    label, mode = r[0], r[1]
    if mode == "rabitq":  # label rabitq budget nlist nprobe disk code recall q1 q8 build hash
        key, bits, recall, q1, q8 = (label, ""), float(r[5]), float(r[7]), float(r[8]), float(r[9])
    else:  # label tieredq-nested budget nlist nprobe disk code k recall tier_bytes rows q1 q8 build hash
        key, bits, recall, q1, q8 = (label, r[7]), float(r[5]) + 8 * float(r[9]), float(r[8]), float(r[11]), float(r[12])
    rows[key].append((bits, recall, q1, q8))
med = {k: (v[0][0], v[0][1], statistics.median(x[2] for x in v), statistics.median(x[3] for x in v)) for k, v in rows.items()}
print("| Arm | k' | Stored bits (vs RaBitQ) | R@10 seed 0 | QPS 1t.b1 (vs RaBitQ) | QPS 8t.b32 (vs RaBitQ) |")
print("|---|---:|---|---:|---|---|")
for (label, k), (bits, recall, q1, q8) in med.items():
    ref = med[("rabitq2" if label.startswith("t2") or label == "rabitq2" else "rabitq4", "")]
    print(f"| {label} | {k or '-'} | {bits:,.0f} ({bits / ref[0]:.2f}x) | {recall:.4f} | {q1:,.0f} ({q1 / ref[2]:.2f}x) | {q8:,.0f} ({q8 / ref[3]:.2f}x) |")
