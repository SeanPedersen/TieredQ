"""Faiss reference construction and measurement, shared by calibration and held-out replay."""
import sys
import time

import faiss
import numpy as np

K = 10
TARGETS = (0.95, 0.99)
ADD_BATCH = 8192
HNSW_NEIGHBORS = 32
HNSW_CONSTRUCTION_EF = 100
QUANTIZER_SEARCH_EF = 128
LLOYD_ITERATIONS = 4
HNSW_SEARCH_BUDGETS = (10, 20, 40, 80, 160, 320, 640, 1280)


def build_reference(base: np.ndarray, args) -> tuple:
    """Include all training, assignment and insertion work in wall time."""
    start = time.perf_counter()
    dimension = base.shape[1]
    sq8 = bool(getattr(args,"int8_base",None))
    if sq8:
        scales = np.fromfile(args.int8_scales,dtype='<f4')
        if scales.shape != (dimension,) or not np.isfinite(scales).all() or np.any(scales<=0):
            raise ValueError("Invalid shared SQ8 scales")
        codes = np.memmap(args.int8_base,dtype='int8',mode='r',shape=base.shape)
        if args.int8_base.stat().st_size != base.size: raise ValueError("SQ8 shape mismatch")
        offsets = np.fromfile(args.int8_offsets,dtype='<f4') if getattr(args,'int8_offsets',None) else np.zeros(dimension,dtype='float32')
        if offsets.shape!=(dimension,) or not np.isfinite(offsets).all(): raise ValueError("Invalid shared SQ8 offsets")
    if args.variant == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, HNSW_NEIGHBORS)
        index.hnsw.efConstruction = HNSW_CONSTRUCTION_EF
        budgets = HNSW_SEARCH_BUDGETS
    else:
        quantizer = faiss.IndexFlatL2(dimension) if args.variant == "ivf" else faiss.IndexHNSWFlat(dimension, HNSW_NEIGHBORS)
        if args.variant == "ivf-hnsw":
            quantizer.hnsw.efConstruction = HNSW_CONSTRUCTION_EF
            quantizer.hnsw.efSearch = QUANTIZER_SEARCH_EF
        sq8 = bool(getattr(args,"int8_base",None))
        index = (faiss.IndexIVFScalarQuantizer(quantizer,dimension,args.nlist,faiss.ScalarQuantizer.QT_8bit,faiss.METRIC_L2,False)
                 if sq8 else faiss.IndexIVFFlat(quantizer, dimension, args.nlist, faiss.METRIC_L2))
        index.cp.niter = LLOYD_ITERATIONS
        index.cp.seed = args.seed
        given = getattr(args, "centroids", None)
        if given:
            # External partition: the quantizer holds the given centroids; nothing is trained.
            centroids = np.fromfile(given, dtype="<f4").reshape(args.nlist, dimension)
            quantizer.add(centroids)
            index.is_trained = True
            print(f"Using {args.nlist} given centroids from {given}", file=sys.stderr, flush=True)
        else:
            sample = np.random.default_rng(args.seed).choice(len(base), min(args.sample, len(base)), replace=False)
            print(f"Training {args.variant}: {len(sample)} vectors, {args.nlist} lists", file=sys.stderr, flush=True)
            training = codes[sample].astype(np.float32) if sq8 else np.ascontiguousarray(base[sample])
            if sq8:
                training *= scales
                training += offsets
            index.train(training)
            del training
        budgets = sorted({min(args.nlist, 2**power) for power in range(args.nlist.bit_length() + 1)})
    sq8 = bool(getattr(args,"int8_base",None))
    if sq8:
        faiss.copy_array_to_vector(np.concatenate([offsets-np.float32(128.5)*scales,np.float32(255)*scales]),index.sq.trained)
        prefix=index.coarse_code_size()
        if index.sa_code_size()!=prefix+dimension: raise ValueError("Unexpected SQ8 code size")
    labels = np.fromfile(args.assignments, dtype="<u4") if getattr(args, "assignments", None) else None
    if labels is not None and len(labels) != len(base): raise ValueError("one assignment label per row required")
    for offset in range(0, len(base), ADD_BATCH):
        if sq8:
            end=min(offset+ADD_BATCH,len(base))
            if labels is not None:
                assignments = labels[offset:end].astype(np.int64).reshape(-1, 1)
            else:
                decoded=codes[offset:end].astype(np.float32)
                decoded *= scales
                decoded += offsets
                _, assignments=index.quantizer.search(decoded,1)
                del decoded
            packed=np.empty((end-offset,prefix+dimension),dtype='uint8')
            for byte in range(prefix): packed[:,byte]=(assignments[:,0]>>(8*byte))&255
            packed[:,prefix:]=np.bitwise_xor(codes[offset:end].view('uint8'),np.uint8(128))
            index.add_sa_codes(packed)
        else:
            index.add(base[offset : offset + ADD_BATCH])
        print(f"Added {min(offset + ADD_BATCH, len(base))}/{len(base)} ({time.perf_counter() - start:.1f}s)",
              file=sys.stderr, flush=True)
    if sq8: codes._mmap.close()
    elapsed = time.perf_counter() - start
    print(f"Built {args.variant} in {elapsed:.3f}s", file=sys.stderr, flush=True)
    return index, budgets, elapsed


def measure(index, variant: str, queries: np.ndarray, truth: np.ndarray, budget: int) -> dict:
    """Warm each setting and count distances independently of Python timing overhead."""
    if variant == "hnsw":
        index.hnsw.efSearch = budget
    else:
        index.nprobe = budget
        if variant == "ivf-hnsw":
            # The coarse graph must return at least nprobe lists; the IVF probes exactly that many.
            faiss.downcast_index(index.quantizer).hnsw.efSearch = max(QUANTIZER_SEARCH_EF, budget)
    for query in queries[:20]:
        index.search(query.reshape(1, -1), K)
    latency, hits, evaluations, maximum = [], 0, 0, 0
    for query, expected in zip(queries, truth, strict=True):
        stats = faiss.cvar.hnsw_stats if variant == "hnsw" else faiss.cvar.indexIVF_stats
        stats.reset()
        start = time.perf_counter_ns()
        _, found = index.search(query.reshape(1, -1), K)
        latency.append((time.perf_counter_ns() - start) / 1000)
        hits += len(set(found[0]) & set(expected))
        evaluations += stats.ndis
        maximum = max(maximum, stats.ndis)
    ordered = sorted(latency)
    p50, p99 = (ordered[max(0, int(np.ceil(p * len(ordered))) - 1)] for p in (0.5, 0.99))
    result = dict(budget=budget, recall10=hits / (K * len(queries)), p50_us=p50, p99_us=p99,
                  mean_distance_evaluations=evaluations / len(queries), max_distance_evaluations=maximum)
    print(f"{variant} budget={budget} recall={result['recall10']:.4f} p50_us={p50:.1f}", file=sys.stderr, flush=True)
    return result


def select_targets(observations: list[dict], targets=TARGETS) -> dict:
    """Select using calibration measurements only; a missing target remains explicit."""
    return {str(target): min((row for row in observations if row["recall10"] >= target),
                             key=lambda row: row["p50_us"], default=None) for target in targets}
