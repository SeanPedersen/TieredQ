"""VIBE-equivalent single-core run of one index configuration on VIBE arxiv-nomic-768-normalized (#41, #43, #42).

Follows ~/experiments/TieredQ-vibe-arxiv-20261106/vibe_like_thp.py (itself mirroring vibe/runner.py): index size is the
USS growth across fit() in KiB, every query is a separate timed call for k = 100 on one pinned core, QPS is the best of
RUNS passes, and Recall@100 counts returned ids whose exact distance is within the 100th true distance + 1e-3.
fit() runs on BUILD_CORES, so build times are not VIBE's single-core figures; query timing and index size are.

Changes from that harness, for the b0148dd API:
- tieredq-ivf takes lists None (automatic), max_scan_bits (0: automatic, 4 up to 2 bits/dim; 6: the pre-b0148dd cap),
  router (None automatic, True/False forced) and sample_seed: train_lists() on a seeded sample of the automatic
  build's size (max(32,768, 64 per list)), which moves the partition the way the C++ seed does.
- tieredq-nested-ivf is the low-RAM default (1-bit prefix, 8-bit fine suffix tier): the tier is written to TIER_DIR on
  the NVMe SSD and read with direct I/O (32 in flight), so it adds neither USS nor page cache.
- Each row also records list padding (#34) and AnonHugePages / RssAnon after fit (#42).
- Follow-up additions: K (env, default 100; recall@K against the K-th true distance), the private bytes that grew
  across fit() per mapping (smaps: [heap], each anonymous region, each file), and for nested indexes the USS after
  each step (build, write_tier, attach_tier) and the serialized size (index + refinement). WARMUP_ROWS > 0 first
  builds and drops a small index (flat only), so the measured growth excludes one-time process state.
THP=off disables transparent huge pages for this process (prctl), as every #41 run did.

Usage: BUILD_CORES=0,1,.. QUERY_CORE=2 [TIER_DIR=..] vibe_arxiv.py DATASET.hdf5 OUT.jsonl ALGO 'BUILD_JSON' 'QUERY_JSON_LIST'
"""

import ctypes
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import psutil

K = int(os.environ.get("K", "100"))
RUNS = 5
GROWTH_REPORT_KIB = 1024  # mappings that grew less are summed into "other"
WARMUP_ROWS = int(os.environ.get("WARMUP_ROWS", "0"))  # > 0: build a small index of this many rows before measuring
RECALL_EPSILON = 1e-3
KIB = 1024
ABSENT_ID = 2**32 - 1
LIST_ALIGNMENT = 32  # fast_list_alignment: each list starts on this many documents
MIN_LIST_SAMPLE, SAMPLE_ROWS_PER_LIST = 32768, 64
BUILD_CORES = {int(c) for c in os.environ["BUILD_CORES"].split(",")}
QUERY_CORE = int(os.environ["QUERY_CORE"])
THP = os.environ.get("THP", "host")
PR_SET_THP_DISABLE = 41
BUILD_THREADS = len(BUILD_CORES)


def serve_single_threaded():
    # A warm multi-thread OpenMP pool made single queries 2.8x slower (wake-ups); queries must run as in VIBE.
    for library, setter in (("libgomp.so.1", "omp_set_num_threads"), ("libopenblas.so.0", "openblas_set_num_threads")):
        try:
            getattr(ctypes.CDLL(library), setter)(1)
        except OSError:
            pass
    os.sched_setaffinity(0, {QUERY_CORE})


def uss_kib():
    return psutil.Process().memory_full_info().uss / KIB


def memory_kib():
    fields = {}
    for line in open("/proc/self/smaps_rollup"):
        name, _, rest = line.partition(":")
        if name in ("AnonHugePages", "Anonymous"):
            fields[name] = int(rest.split()[0])
    for line in open("/proc/self/status"):
        if line.startswith("RssAnon:"):
            fields["RssAnon"] = int(line.split()[1])
    return fields


def private_mappings_kib():
    """Private (clean + dirty) KiB per mapping: files and [heap]/[stack] by name, anonymous regions by start address."""
    mappings, key = {}, None
    for line in open("/proc/self/smaps"):
        fields = line.split()
        if "-" in fields[0] and not fields[0].endswith(":"):
            key = fields[5] if len(fields) > 5 else f"anon@{fields[0].split('-')[0]}"
        elif fields[0] in ("Private_Clean:", "Private_Dirty:"):
            mappings[key] = mappings.get(key, 0) + int(fields[1])
    return mappings


def mapping_growth(before, after):
    grown = {key: after[key] - before.get(key, 0) for key in after}
    report = {key: kib for key, kib in sorted(grown.items(), key=lambda item: -item[1]) if kib >= GROWTH_REPORT_KIB}
    report["other"] = sum(kib for key, kib in grown.items() if key not in report)
    return report


def padding(list_sizes):
    padded = -(-list_sizes // LIST_ALIGNMENT) * LIST_ALIGNMENT
    return dict(live_rows=int(list_sizes.sum()), padding_rows=int((padded - list_sizes).sum()),
                empty_lists=int((list_sizes == 0).sum()))


class TieredQIVF:
    def __init__(self, bits_per_dim, lists=None, max_scan_bits=0, router=None, sample_seed=None):
        self.bits_per_dim, self.lists, self.max_scan_bits = bits_per_dim, lists, max_scan_bits
        self.router, self.sample_seed = router, sample_seed

    def fit(self, X):
        from tieredq import TieredQIVFIndex

        self.index = TieredQIVFIndex(dimensions=X.shape[1], code_bits=round(self.bits_per_dim * X.shape[1]),
                                     lists=self.lists, metric="ip", max_scan_bits=self.max_scan_bits, router=self.router)
        if self.sample_seed is not None:
            rows = max(MIN_LIST_SAMPLE, SAMPLE_ROWS_PER_LIST * self.lists)
            pick = np.sort(np.random.default_rng(self.sample_seed).choice(len(X), rows, replace=False))
            sample = np.ascontiguousarray(X[pick])
            self.index.train_lists(sample)
            del sample
        self.index.build(X)

    def info(self):
        sizes = np.bincount(self.index.list_assignment(), minlength=self.index.lists)
        return dict(lists=self.index.lists, has_router=self.index.has_router, storage_bytes=self.index.storage_bytes,
                    **padding(sizes))

    def set_query_arguments(self, nprobe):
        self.nprobe = nprobe

    def query(self, q, n):
        _, ids = self.index.search(q, k=n, nprobe=self.nprobe)
        ids = ids.reshape(-1)
        return ids[ids != ABSENT_ID]


class TieredQNestedIVF:
    """The low-RAM default: 1-bit prefix IVF in RAM, candidates rescored from an 8-bit tier on SSD."""

    def __init__(self, prefix_bits_per_dim=None, lists=None):
        self.prefix_bits_per_dim, self.lists = prefix_bits_per_dim, lists
        self.tier_path = Path(os.environ["TIER_DIR"]) / f"tier-{os.getpid()}.rows"

    def fit(self, X):
        from tieredq import TieredQNestedIVFIndex

        prefix = round(self.prefix_bits_per_dim * X.shape[1]) if self.prefix_bits_per_dim else None
        steps = {"start": uss_kib()}
        self.index = TieredQNestedIVFIndex.build(X, prefix_bits=prefix, lists=self.lists, metric="ip")
        steps["build"] = uss_kib()
        self.index.write_tier(self.tier_path, X)
        steps["write_tier"] = uss_kib()
        self.index.attach_tier(self.tier_path, reads="direct")
        steps["attach_tier"] = uss_kib()
        self.steps_kib = {name: kib - steps["start"] for name, kib in steps.items()}

    def info(self):
        return dict(lists=self.index.lists, has_router=self.index.has_router, storage_bytes=self.index.storage_bytes,
                    serialized_bytes=len(self.index.serialize()), uss_steps_kib=self.steps_kib,
                    prefix_bits=self.index.prefix_bits, total_bits=self.index.total_bits,
                    tier_bytes=self.tier_path.stat().st_size, tier_bytes_per_vector=self.index.tier_bytes_per_vector)

    def set_query_arguments(self, query_args):
        self.nprobe, self.candidates = query_args

    def query(self, q, n):
        _, ids = self.index.search(q, k=n, nprobe=self.nprobe, candidates=max(n, self.candidates))
        ids = ids.reshape(-1)
        return ids[ids != ABSENT_ID]


class RabitqIVF:
    """VIBE's vibe/algorithms/rabitq/module.py RabitqIVF, verbatim in behavior."""

    def __init__(self, num_clusters, total_bits, fast_quantization=False):
        self.num_clusters, self.total_bits, self.fast_quantization = num_clusters, total_bits, fast_quantization

    def fit(self, X):
        import faiss
        from rabitqlib import IvfIndex

        faiss.omp_set_num_threads(BUILD_THREADS)
        coarse = faiss.index_factory(X.shape[1], f"IVF{self.num_clusters},Flat", faiss.METRIC_INNER_PRODUCT)
        coarse.train(X)
        centroids = np.ascontiguousarray(coarse.quantizer.reconstruct_n(0, coarse.nlist), dtype=np.float32)
        _, cluster_ids = coarse.quantizer.search(X, 1)
        cluster_ids = np.ravel(cluster_ids).astype(np.uint32)
        self.index = IvfIndex(dim=X.shape[1], max_elements=X.shape[0], num_clusters=self.num_clusters,
                              nbits=self.total_bits, metric="ip")
        self.index.build(X, centroids, cluster_ids, num_threads=BUILD_THREADS,
                         fast_quantization=self.fast_quantization)

    def info(self):
        return {}

    def set_query_arguments(self, nprobe):
        self.nprobe = nprobe

    def query(self, q, n):
        ids, _ = self.index.search(np.expand_dims(q, 0), k=n, nprobe=self.nprobe, num_threads=1)
        return ids[0][ids[0] < self.index.max_elements]


ALGORITHMS = {"tieredq-ivf": TieredQIVF, "tieredq-nested-ivf": TieredQNestedIVF, "rabitq-ivf": RabitqIVF}


def recall_at_k(train, test, true_distances, results):
    hits = []
    for q, ids, truth in zip(test, results, true_distances):
        distances = 1.0 - train[ids] @ q
        hits.append(np.sum(distances[:K] <= truth[K - 1] + RECALL_EPSILON))
    return float(np.mean(hits)) / K


def main():
    if THP == "off" and ctypes.CDLL("libc.so.6").prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0):
        sys.exit("prctl(PR_SET_THP_DISABLE) failed")
    dataset, out_path, name, build_json, query_json = sys.argv[1:6]
    with h5py.File(dataset, "r") as f:
        assert f.attrs["distance"] == "normalized", f.attrs["distance"]
        train = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        test = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        true_distances = f["distances"][:]
    build_args = json.loads(build_json)
    algo = ALGORITHMS[name](**build_args)

    os.sched_setaffinity(0, BUILD_CORES)
    if WARMUP_ROWS:  # one-time process state (module, BLAS buffers, OpenMP pool) then precedes the baseline
        warm = ALGORITHMS[name](**dict(build_args, lists=min(build_args.get("lists") or 256, 256)))
        warm.fit(np.ascontiguousarray(train[:WARMUP_ROWS]))
        del warm
    before, memory_before, mappings_before = uss_kib(), memory_kib(), private_mappings_kib()
    start = time.time()
    algo.fit(train)
    build_time = time.time() - start
    index_size = uss_kib() - before
    memory_after, mappings_after = memory_kib(), private_mappings_kib()
    info = dict(algo.info(), memory_before_kib=memory_before, memory_after_kib=memory_after,
                mapping_growth_kib=mapping_growth(mappings_before, mappings_after), k=K)
    serve_single_threaded()
    print(f"{name} {build_args}: built in {build_time:.1f}s, {index_size / KIB**2:.3f} GB, {info}", flush=True)

    for query_arg in json.loads(query_json):
        algo.set_query_arguments(query_arg)
        best_qps, results = 0.0, None
        for _ in range(RUNS):
            total, results = 0.0, []
            for q in test:
                t0 = time.perf_counter()
                ids = algo.query(q, K)
                total += time.perf_counter() - t0
                results.append(np.asarray(ids, dtype=np.int64))
            best_qps = max(best_qps, len(test) / total)
        recall = recall_at_k(train, test, true_distances, results)
        row = dict(algorithm=name, build=build_args, query=query_arg, qps=best_qps, recall=recall,
                   build_time=build_time, build_threads=BUILD_THREADS, index_size=index_size, thp=THP,
                   n=len(train), dim=train.shape[1], info=info)
        with open(out_path, "a") as out:
            out.write(json.dumps(row) + "\n")
        print(f"  query={query_arg}: recall={recall:.4f} qps={best_qps:.0f} "
              f"qps/GB={best_qps / (index_size / KIB**2):.0f}", flush=True)


if __name__ == "__main__":
    main()
