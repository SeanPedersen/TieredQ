"""Verify archived experiment inputs and expose the isolated exact-budget native adapter."""
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'storage-sweep'
SOURCE = ROOT / 'source'
ADAPTER = ROOT / 'matched-storage'
DATA = Path('/home/sean/experiments/TieredQ-issue4-current-20261026-tpj3ze/benchmark/data')
GEMINI = Path('/home/sean/experiments/TieredQ-gemini-turboquant-20261026-vDAm7R/TieredQ/benchmark/data/gemini-prepared.npz')
NATIVE_THREADS = 8
QUERY_COUNT = 1000
TOP_K = 10
sys.path[:0] = [str(ADAPTER / 'build'), str(SOURCE / 'benchmark')]
import codec_native
import numpy as np
from data import prepare
from build_policy import audit_compile_commands


@dataclass
class BuiltIndex:
    index: object
    build_s: float
    bytes_per_vector: float
    encode_vps: float


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_rows():
    return list(csv.DictReader((ROOT / 'output/results.csv').open()))


def verify_environment():
    expected_affinity = list(range(NATIVE_THREADS))
    assert sorted(os.sched_getaffinity(0)) == expected_affinity
    metadata = json.loads((ROOT / 'output/meta.json').read_text())
    for filename, expected in metadata['source_sha256'].items():
        assert sha256(SOURCE / filename) == expected, filename
    original = (SOURCE / 'benchmark/native/module.cpp').read_text()
    patched = original.replace('bool fast = true)', 'bool fast = true, size_t code_bits = 0)').replace(
        'bits * (width_ - (l2 ? 1 : 0))', 'code_bits ? code_bits : bits * (width_ - (l2 ? 1 : 0))').replace(
        '.def("diagnostics", &Index::diagnostics)',
        '.def(nb::init<Matrix,size_t,uint64_t,bool,bool,bool,size_t>(), nb::call_guard<nb::gil_scoped_release>())\n    .def("diagnostics", &Index::diagnostics)')
    assert patched == (ADAPTER / 'native/module.cpp').read_text()
    assert Path(codec_native.__file__).parent == ADAPTER / 'build'
    commands = audit_compile_commands(json.loads((ADAPTER / 'build/compile_commands.json').read_text()))
    return dict(source_commit=metadata['tieredq_commit'], source_hashes_match=True,
                adapter_sha256=sha256(ADAPTER / 'native/module.cpp'),
                binary_sha256=sha256(Path(codec_native.__file__)),
                compiler_commands=commands, affinity=expected_affinity,
                host=platform.node(), date=datetime.now(timezone.utc).isoformat())


def load_dataset(dataset):
    if dataset == 'gemini':
        with np.load(GEMINI) as arrays:
            base, queries, truth = (arrays[name] for name in ('base', 'queries', 'truth'))
    else:
        base, queries, truth = prepare(DATA, dataset, None, QUERY_COUNT)
    hashes = {name: hashlib.sha256(array.tobytes()).hexdigest()
              for name, array in [('base', base), ('queries', queries), ('truth', truth)]}
    expected = json.loads((ROOT / 'output/inputs.json').read_text())[dataset]['sha256']
    assert hashes == expected
    assert len(queries) == QUERY_COUNT
    return base, queries, truth, hashes


def build(base, code_bits, seed, directory):
    assert code_bits > 0
    codec_native.set_threads(NATIVE_THREADS)
    started = time.perf_counter()
    index = codec_native.Index(base, 1, seed, True, False, True, code_bits)
    elapsed = time.perf_counter() - started
    storage = index.storage(str(directory / 'selection-index.bin')) / len(base)
    return BuiltIndex(index, elapsed, storage, len(base) / index.encode_seconds)


def search_ids(index, queries, threads=NATIVE_THREADS, batch=32):
    ids = np.empty((len(queries), TOP_K), dtype=np.uint32)
    for offset in range(0, len(queries), batch):
        index.search(queries[offset:offset + batch], ids[offset:offset + batch], threads)
    return ids


def verify_control(dataset, base, queries, directory, rows):
    dimensions = base.shape[1]
    built = build(base, dimensions, 0, directory)
    ids = search_ids(built.index, queries)
    expected_ids = np.load(ROOT / f'output/cell-{dataset}-tieredq-1-0-t8-b32-ids.npy')
    assert np.array_equal(ids, expected_ids)
    baseline = next(r for r in rows if (r['dataset'], r['codec'], r['target_bits'], r['seed'], r['threads'], r['batch']) ==
                    (dataset, 'tieredq', '1', '0', '8', '32'))
    assert built.bytes_per_vector == float(baseline['effective_bytes_per_vector'])
    return dict(dataset=dataset, code_bits=dimensions, identical_ids=True,
                bytes_per_vector=built.bytes_per_vector)
