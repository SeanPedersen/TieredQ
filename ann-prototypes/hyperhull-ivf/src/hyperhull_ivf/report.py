"""Plain-text tables for experiment results."""

from .evaluate import CurvePoint, cost_at_recall

RECALL_TARGETS = (0.90, 0.95, 0.99, 1.0)


def _fmt(value: float | None) -> str:
    return f"{value:7.0f}" if value is not None else "      -"


def print_header(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}", flush=True)


def print_curve(name: str, curve: list[CurvePoint], label: str) -> None:
    cells = "  ".join(f"{label}={p.param:g}: R={p.recall:.3f} tot={p.total:.0f}" for p in curve)
    print(f"  {name:<24} {cells}", flush=True)


def print_cost_table(curves: dict[str, list[CurvePoint]], baseline: str, n_vectors: int) -> None:
    head = "".join(f"   R>={t:<5} scan/total      " for t in RECALL_TARGETS)
    print(f"\n  cost to reach recall (scan = exact float distances; total = scan + routing + bounds + codes·bits/32; % vs {baseline}; brute force = {n_vectors})")
    print(f"  {'method':<26}{head}")
    base_totals = {t: cost_at_recall(curves[baseline], t, "total") for t in RECALL_TARGETS}
    for name, curve in curves.items():
        row = ""
        for t in RECALL_TARGETS:
            scanned, total = cost_at_recall(curve, t, "scanned"), cost_at_recall(curve, t, "total")
            delta = f"{100 * (total / base_totals[t] - 1):+.0f}%" if total is not None and base_totals[t] else ""
            row += f"  {_fmt(scanned)} /{_fmt(total)} {delta:>6}"
        print(f"  {name:<26}{row}", flush=True)
