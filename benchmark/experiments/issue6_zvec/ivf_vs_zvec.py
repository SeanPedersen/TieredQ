"""Issue #6: TieredQIVFIndex against zvec's IVF_RABITQ collection, both as shipped through Python.

Each codec builds its own coarse lists on the full base, then answers held-out queries one per call
(the single-query serving path). Rows: codec bits lists nprobe index_bits_per_vector recall@10 qps_1t_b1 build_s.
`index_bits_per_vector` is the serialized index: zvec's RaBitQ IVF file (`*.qindex.*`; the collection also keeps a
full FP32 copy in `*.index.*`, untouched by these queries and reported on stderr), TieredQ's `storage_bytes`, and for
`nested` the prefix IVF plus its in-RAM refinement tier (suffix rows; candidates rescored at the total width).
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

K = 10
WARMUP_QUERIES = 100
ZVEC_INSERT_BATCH = 1024
BITS_PER_BYTE = 8
ZVEC_QUANTIZED_INDEX_MARKER = '.qindex.'


def load(data, dimensions):
    def floats(name):
        return np.fromfile(data/name, dtype=np.float32).reshape(-1, dimensions)
    base, queries = floats('base.f32'), floats('queries.f32')
    truth = np.fromfile(data/'truth.u32', dtype=np.uint32).reshape(len(queries), -1)[:, :K]
    return np.ascontiguousarray(base), np.ascontiguousarray(queries), truth


def recall(ids, truth):
    return np.mean([len(set(row) & set(expected)) / K for row, expected in zip(ids, truth)])


def timed_qps(search_one, count):
    for i in range(min(WARMUP_QUERIES, count)):
        search_one(i)
    start = time.perf_counter()
    ids = [search_one(i) for i in range(count)]
    return np.array(ids), count / (time.perf_counter() - start)


class ZvecIVF:
    def __init__(self, workdir, dimensions, bits, lists):
        import zvec
        self.zvec = zvec
        self.path = workdir/f'zvec-b{bits}-l{lists}'
        shutil.rmtree(self.path, ignore_errors=True)
        schema = zvec.CollectionSchema(name=self.path.name, vectors=[zvec.VectorSchema(
            'embedding', zvec.DataType.VECTOR_FP32, dimension=dimensions,
            index_param=zvec.IvfRabitqIndexParam(metric_type=zvec.MetricType.IP, nlist=lists, total_bits=bits))])
        self.collection = zvec.create_and_open(path=str(self.path), schema=schema,
                                               option=zvec.CollectionOption(read_only=False, enable_mmap=True))

    def build(self, base):
        for offset in range(0, len(base), ZVEC_INSERT_BATCH):
            docs = [self.zvec.Doc(id=str(offset + i), vectors={'embedding': row})
                    for i, row in enumerate(base[offset:offset + ZVEC_INSERT_BATCH].tolist())]
            failed = [r for r in self.collection.insert(docs) if not r.ok()]
            if failed:
                sys.exit(f'zvec insert failed: {failed[0]}')
        self.collection.optimize(self.zvec.OptimizeOption())
        if self.collection.stats.index_completeness['embedding'] != 1:
            sys.exit('zvec index incomplete after optimize')
        files = sorted((str(p.relative_to(self.path)), p.stat().st_size) for p in self.path.rglob('*') if p.is_file())
        print('#zvec files', files, file=sys.stderr, flush=True)

    def index_bytes(self):
        return sum(p.stat().st_size for p in self.path.rglob(f'*{ZVEC_QUANTIZED_INDEX_MARKER}*'))

    def searcher(self, queries, nprobe):
        param = self.zvec.IvfRabitqQueryParam(nprobe=nprobe)
        prepared = [self.zvec.Query(field_name='embedding', vector=q, param=param) for q in queries.tolist()]

        def search_one(i):
            docs = self.collection.query(queries=prepared[i], topk=K)
            return [int(d.id) for d in docs] + [-1] * (K - len(docs))
        return search_one


class TieredQIVF:
    def __init__(self, dimensions, code_bits, lists, confidence):
        from tieredq import TieredQIVFIndex, TieredQPruning
        self.index = TieredQIVFIndex(dimensions=dimensions, code_bits=code_bits, lists=lists)
        self.pruning = TieredQPruning(confidence=confidence) if confidence else None

    def build(self, base):
        self.index.build(base)

    def index_bytes(self):
        return self.index.storage_bytes

    def searcher(self, queries, nprobe):
        rows = [queries[i:i + 1] for i in range(len(queries))]

        def search_one(i):
            if self.pruning is None:
                return self.index.search(rows[i], k=K, nprobe=nprobe)[1][0].tolist()
            return self.index.search(rows[i], k=K, nprobe=nprobe, pruning=self.pruning)[1][0].tolist()
        return search_one


class TieredQNested:
    def __init__(self, dimensions, total_bits, lists, prefix_bits, candidates):
        self.config = dict(prefix_bits=prefix_bits, total_bits=total_bits, lists=lists, tier_rows='suffix')
        self.candidates = candidates

    def build(self, base):
        from tieredq import TieredQNestedIVFIndex
        self.index = TieredQNestedIVFIndex.build(base, **self.config)
        self.index.attach_tier(self.index.encode_tier(base))
        self.rows = len(base)

    def index_bytes(self):
        return self.index.storage_bytes + self.index.tier_bytes_per_vector * self.rows

    def searcher(self, queries, nprobe):
        rows = [queries[i:i + 1] for i in range(len(queries))]

        def search_one(i):
            return self.index.search(rows[i], k=K, nprobe=nprobe, candidates=self.candidates)[1][0].tolist()
        return search_one


def sweep(label, bits, lists, index, base, queries, truth, nprobes, out):
    start = time.perf_counter()
    index.build(base)
    build_s = time.perf_counter() - start
    index_bits = index.index_bytes() * BITS_PER_BYTE / len(base)
    for nprobe in nprobes:
        ids, qps = timed_qps(index.searcher(queries, nprobe), len(queries))
        row = f'{label}\t{bits}\t{lists}\t{nprobe}\t{index_bits:.0f}\t{recall(ids, truth):.4f}\t{qps:.1f}\t{build_s:.1f}'
        print(row, flush=True)
        out.write(row + '\n')
        out.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--dimensions', type=int, default=768)
    parser.add_argument('--codec', choices=('zvec', 'tieredq', 'nested'), required=True)
    parser.add_argument('--bits', type=float, nargs='+', required=True,
                        help='zvec total_bits per dimension, or TieredQ code (nested: total) bits per dimension')
    parser.add_argument('--prefix-bits', type=float, default=1, help='nested: prefix code bits per dimension')
    parser.add_argument('--candidates', type=int, default=50, help='nested: candidates rescored per query')
    parser.add_argument('--confidence', type=float, default=0,
                        help='TieredQ opt-in approximate pruning confidence (0 = exact pruning)')
    parser.add_argument('--lists', type=int, default=256)
    parser.add_argument('--nprobe', type=int, nargs='+', default=[4, 8, 16, 32, 64, 128])
    parser.add_argument('--workdir', type=Path, default=Path('work'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    base, queries, truth = load(args.data, args.dimensions)
    args.workdir.mkdir(parents=True, exist_ok=True)
    with args.out.open('a') as out:
        for bits in args.bits:
            if args.codec == 'zvec':
                index = ZvecIVF(args.workdir, args.dimensions, int(bits), args.lists)
                label = args.codec
            elif args.codec == 'nested':
                index = TieredQNested(args.dimensions, round(bits * args.dimensions), args.lists,
                                      round(args.prefix_bits * args.dimensions), args.candidates)
                label = f'nested-p{args.prefix_bits:g}-c{args.candidates}'
            else:
                index = TieredQIVF(args.dimensions, round(bits * args.dimensions), args.lists, args.confidence)
                label = f'{args.codec}-z{args.confidence:g}' if args.confidence else args.codec
            sweep(label, bits, args.lists, index, base, queries, truth, args.nprobe, out)


if __name__ == '__main__':
    main()
