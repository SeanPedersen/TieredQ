"""Correctness of the native TieredQ Partition on clustered random data: the order is a permutation, fine cells nest
in coarse lists, every row sits in its nearest nonempty fine cell of its list and in its nearest coarse list, and
the result does not depend on the thread count. Also prints build stage seconds per thread count.

Usage: uv run python experiments/native/check_partition.py [threads ...]
"""

import sys
import time

import numpy as np

from hyperhull_native import Partition

N_ROWS, DIM, N_BLOBS = 20_000, 64, 50
N_COARSE, FINE = 64, 8


def clustered_rows(rng):
    centers = rng.normal(size=(N_BLOBS, DIM)).astype(np.float32) * 3
    return (centers[rng.integers(N_BLOBS, size=N_ROWS)] + rng.normal(size=(N_ROWS, DIM))).astype(np.float32)


def check_layout(p: Partition) -> np.ndarray:
    order, coarse, fine = p.order, p.coarse_offsets, p.fine_offsets
    assert np.array_equal(np.sort(order), np.arange(N_ROWS)), "order is not a permutation"
    assert coarse[0] == 0 and coarse[-1] == N_ROWS and fine[-1] == N_ROWS
    assert np.all(np.diff(coarse) >= 0) and np.all(np.diff(fine) >= 0)
    assert np.array_equal(fine[::FINE], coarse), "fine cells must nest in coarse lists"
    return order


def misassigned(p: Partition, stored: np.ndarray) -> int:
    fine, centroids = p.fine_offsets, p.fine_centroids
    wrong = 0
    for cell_list in range(N_COARSE):
        cells = slice(cell_list * FINE, (cell_list + 1) * FINE)
        sizes = np.diff(fine[cell_list * FINE : (cell_list + 1) * FINE + 1])
        for f in range(FINE):
            rows = stored[fine[cell_list * FINE + f] : fine[cell_list * FINE + f + 1]]
            if not len(rows):
                continue
            d = ((rows[:, None, :] - centroids[cells][None]) ** 2).sum(-1)
            d[:, sizes == 0] = np.inf
            wrong += int((d.argmin(1) != f).sum())
    return wrong


def coarse_misassigned(p: Partition, stored: np.ndarray) -> int:
    lists = np.repeat(np.arange(N_COARSE), np.diff(p.coarse_offsets))
    d = ((stored[:, None, :] - p.coarse_centroids[None]) ** 2).sum(-1)
    return int((d.argmin(1) != lists).sum())


def main() -> None:
    threads = [int(a) for a in sys.argv[1:]] or [1, 8]
    rng = np.random.default_rng(0)
    x = clustered_rows(rng)
    orders = []
    for n_threads in threads:
        t = time.perf_counter()
        p = Partition(x, coarse_lists=N_COARSE, fine_per_coarse=FINE, n_threads=n_threads)
        stats = {k: round(v, 3) for k, v in p.build_stats.items()}
        print(f"threads={n_threads}: build {time.perf_counter() - t:.2f}s {stats}")
        orders.append(check_layout(p))
    assert all(np.array_equal(o, orders[0]) for o in orders), "partition depends on the thread count"
    stored = x[orders[-1]]
    print("layout ok, thread-count independent; rows not in their nearest fine cell:", misassigned(p, stored),
          "/ coarse list:", coarse_misassigned(p, stored))


if __name__ == "__main__":
    main()
