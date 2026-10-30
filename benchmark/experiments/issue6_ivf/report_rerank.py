"""Issue #19.1 / #22: pick timing cells from the untimed IVF + rerank recall pass, and tabulate the timed result.

Rows are job_rerank.sh output: label codec budget lists nprobe disk_bits code_bits rerank_k cand_recall recall reads
fp32_bits qps_1t_b1 qps_8t_b32 build_s list_hash.

  select RECALL_DIR [TARGETS...] [--lists L]
                                -> job_rerank.sh CONFIGS lines: per configuration and k', the smallest nprobe that
                                   reaches each target (larger nprobe or k' at equal recall only costs time); --lists 0
                                   for runs on the codec's own lists, whose rows print the codec list count
  report DIR... [--targets ...] -> markdown: per codec the least index RAM per target, and per memory level the
                                   fastest configuration per target (index RAM vs QPS)
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

COLUMNS = ("label codec budget lists nprobe disk_bits code_bits rerank_k cand_recall recall reads fp32_bits "
           "qps_1t qps_8t build_s list_hash").split()
NUMERIC = {"budget", "disk_bits", "code_bits", "cand_recall", "recall", "reads", "fp32_bits", "qps_1t", "qps_8t",
           "build_s"}
INTEGER = {"lists", "nprobe", "rerank_k"}
DEFAULT_TARGETS = (0.90, 0.95, 0.99)
# Matched budgets share a memory level when their disk bits/vector are within this fraction of each other.
LEVEL_TOLERANCE = 0.05
TUNING_ENV = {"z8-p3": "PRUNE_Z=8 PRUNE_PREFIX=3", "z8-p4": "PRUNE_Z=8 PRUNE_PREFIX=4", "max4": "MAX_BITS=4"}


def parse(value: str, column: str):
    if column in INTEGER:
        return int(value)
    return float(value) if column in NUMERIC else value


def load_rows(paths: list[Path]) -> list[dict]:
    files = [f for p in paths for f in (sorted(p.glob("*.tsv")) if p.is_dir() else [p])]
    rows = []
    for path in files:
        for fields in csv.reader(path.open(), delimiter="\t"):
            rows.append({c: parse(v, c) for c, v in zip(COLUMNS, fields)})
    return rows


def config_key(row: dict) -> tuple:
    return row["label"], row["lists"], row["codec"], row["budget"]


def minimal_cells(rows: list[dict], target: float) -> list[dict]:
    """Cells reaching target that no other cell of the same configuration dominates in both nprobe and k'."""
    reached = [r for r in rows if r["recall"] >= target]
    return [r for r in reached if not any(
        o is not r and o["nprobe"] <= r["nprobe"] and o["rerank_k"] <= r["rerank_k"] for o in reached)]


def select(recall_dir: Path, targets: list[float], lists_arg: int | None) -> None:
    configs = defaultdict(list)
    for row in load_rows([recall_dir]):
        configs[config_key(row)].append(row)
    for (label, lists, codec, budget), rows in sorted(configs.items()):
        cells = defaultdict(set)
        for target in targets:
            for row in minimal_cells(rows, target):
                cells[row["rerank_k"]].add(row["nprobe"])
        mode = "rabitq" if codec == "rabitq" else "tieredq"
        env = TUNING_ENV.get(label, "")
        for depth, nprobes in sorted(cells.items()):
            nprobe_list = ",".join(str(n) for n in sorted(nprobes))
            print(f"{label} {lists if lists_arg is None else lists_arg} {mode} {budget:g} RERANK={depth} NPROBES={nprobe_list} {env}".rstrip())


def memory_levels(rows: list[dict]) -> dict[tuple, int]:
    """Memory level (0 = smallest) per (lists, budget configuration), grouping budgets of near-equal disk bits."""
    sizes = sorted({(r["lists"], r["disk_bits"]) for r in rows})
    levels, current = {}, defaultdict(lambda: (-1, 0.0))
    for lists, bits in sizes:
        level, anchor = current[lists]
        if level < 0 or bits > anchor * (1 + LEVEL_TOLERANCE):
            current[lists] = level, anchor = level + 1, bits
        levels[lists, bits] = level
    return levels


def family(codec: str) -> str:
    return "RaBitQ" if codec == "rabitq" else "TieredQ"


def describe(row: dict) -> str:
    setting = "" if row["label"] in ("default", "rabitq") else f" {row['label']}"
    return f"{row['budget']:g}{setting}, nprobe {row['nprobe']}, k' {row['rerank_k']}"


def cell(row: dict | None) -> list[str]:
    if row is None:
        return ["—"] * 6
    return [describe(row), f"{row['disk_bits']:,.0f}", f"{row['recall']:.3f}", f"{row['qps_1t']:,.0f}",
            f"{row['qps_8t']:,.0f}", f"{row['reads']:.0f}"]


def report(paths: list[Path], targets: list[float]) -> None:
    rows = [r for r in load_rows(paths) if r["qps_1t"] > 0]
    fp32_bytes = {r["fp32_bits"] / 8 for r in rows}
    print(f"Rerank tier: {', '.join(f'{b:,.0f}' for b in sorted(fp32_bytes))} B/vector FP32, reported separately "
          "(SSD or RAM); index RAM = serialized compressed IVF.\n")
    for lists in sorted({r["lists"] for r in rows}):
        group = [r for r in rows if r["lists"] == lists]
        print(f"### {lists} lists: least index RAM per recall@10 target (then fastest 1-thread)\n")
        print("| Target | Codec | Config (budget, nprobe, k') | Index bits/vec | Recall | QPS 1t | QPS 8t×32 | Reads/q |")
        print("|---|---|---|---|---|---|---|---|")
        for target in targets:
            for name in ("TieredQ", "RaBitQ"):
                ok = [r for r in group if family(r["codec"]) == name and r["recall"] >= target]
                best = min(ok, key=lambda r: (r["disk_bits"], -r["qps_1t"])) if ok else None
                print(f"| {target:.2f} | {name} | " + " | ".join(cell(best)) + " |")
        print()
        levels = memory_levels(group)
        print(f"### {lists} lists: fastest configuration per memory level and target (index RAM vs QPS)\n")
        print("| Target | Level | Codec | Config | Index bits/vec | Recall | QPS 1t | QPS 8t×32 | Reads/q |")
        print("|---|---|---|---|---|---|---|---|---|")
        for target in targets:
            for level in sorted(set(levels.values())):
                for name in ("TieredQ", "RaBitQ"):
                    ok = [r for r in group if family(r["codec"]) == name and r["recall"] >= target
                          and levels[lists, r["disk_bits"]] == level]
                    best = max(ok, key=lambda r: r["qps_1t"]) if ok else None
                    print(f"| {target:.2f} | {level} | {name} | " + " | ".join(cell(best)) + " |")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("select")
    pick.add_argument("recall_dir", type=Path)
    pick.add_argument("targets", type=float, nargs="*", default=list(DEFAULT_TARGETS))
    pick.add_argument("--lists", type=int, help="LISTS value for the emitted lines (default: the rows' list count)")
    table = sub.add_parser("report")
    table.add_argument("paths", type=Path, nargs="+")
    table.add_argument("--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS))
    args = parser.parse_args()
    if args.command == "select":
        select(args.recall_dir, args.targets, args.lists)
    else:
        report(args.paths, args.targets)


if __name__ == "__main__":
    sys.exit(main())
