"""Time complete calls and evaluate recall outside measured regions."""
import time
import numpy as np

TOP_K = 10
# Recall@depth reported for every search at least that deep.
RECALL_DEPTHS = (10, 100)
WARMUPS = 3
REPEATS = 3


def recall(truth, ids, depth=TOP_K):
    if truth.shape[1] < depth or ids.shape[1] < depth:
        raise ValueError(f"recall@{depth} needs truth and results at least {depth} deep")
    return float(np.mean([len(set(a[:depth]) & set(b[:depth])) / depth for a, b in zip(truth, ids, strict=True)]))


def recalls(truth, ids):
    return {f"recall{depth}": recall(truth, ids, depth) for depth in RECALL_DEPTHS if depth <= ids.shape[1]}


def measure(index, queries, truth, threads, batch, k=TOP_K):
    output = np.empty((len(queries), k), dtype=np.uint32)
    for i in range(WARMUPS):
        index.search(queries[i % len(queries):i % len(queries)+1], output[:1], threads)
    durations = []
    total = 0.0
    for _ in range(REPEATS):
        for start in range(0, len(queries), batch):
            stop = min(start + batch, len(queries))
            tick = time.perf_counter()
            index.search(queries[start:stop], output[start:stop], threads)
            elapsed = time.perf_counter() - tick
            total += elapsed
            # Batch response latency, never mislabel amortized throughput as latency.
            durations.extend([elapsed * 1000] * (stop-start))
    return recalls(truth, output) | {"qps": len(queries)*REPEATS/total,
            "p50_ms": float(np.percentile(durations, 50)),
            "p95_ms": float(np.percentile(durations, 95))}


class L2Search:
    """Include the common query lift in each measured end-to-end call."""
    def __init__(self, index, scale, transform=None):
        self.index, self.scale, self.transform = index, scale, transform

    def search(self, queries, output, threads):
        probes = self.transform.queries(queries) if self.transform is not None else np.ascontiguousarray(
            np.column_stack([queries / self.scale, np.ones(len(queries))]), dtype=np.float32)
        self.index.search(probes, output, threads)
