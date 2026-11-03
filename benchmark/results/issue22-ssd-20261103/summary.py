"""Issue #22 SSD comparison: median QPS over rounds per config. RaBitQ IVF all in RAM; TieredQ nested with the tier on
SSD (direct I/O, queue depth 32). RAM bits = the IVF index per vector; SSD bytes = tier row; reads = k' per query."""
import collections, csv, statistics, sys
rows = collections.defaultdict(list)
for r in csv.reader(open(sys.argv[1]), delimiter="\t"):
    if r[1] == "rabitq":  # label rabitq budget nlist nprobe disk code recall q1 q8 build hash
        key = (r[0], int(r[4]), 0); rows[key].append((float(r[5]), 0, float(r[7]), float(r[8]), float(r[9])))
    else:  # label tieredq-nested budget nlist nprobe disk code k recall tier_bytes rows q1 q8 build hash
        key = (r[0], int(r[4]), int(r[7])); rows[key].append((float(r[5]), int(r[9]), float(r[8]), float(r[11]), float(r[12])))
print("| Config | nprobe | RAM bits/vector | SSD B/vector | Reads/query | R@10 | QPS 1t·b1 | QPS 8t·b32 |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for (label, nprobe, k), v in sorted(rows.items(), key=lambda kv: kv[1][0][2]):
    ram, tier, recall = v[0][:3]
    q1, q8 = (statistics.median(x[i] for x in v) for i in (3, 4))
    name = f"RaBitQ {label[2:]}-bit" if label.startswith("rq") else f"TieredQ nested p{label[2:]}, k'={k}"
    print(f"| {name} | {nprobe} | {ram:,.0f} | {tier or '-'} | {k or 0} | {recall:.4f} | {q1:,.0f} | {q8:,.0f} |  rounds 8t {[round(x[4]) for x in v]}")
