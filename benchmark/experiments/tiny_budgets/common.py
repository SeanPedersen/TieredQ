"""Shared data, recipe, and result contracts for the tiny-budget SIFT study."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

BENCHMARK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'native'))
from l2_transform import L2Transform
from model import parse_model

ORIGINAL_DIMENSIONS = 128
QUERY_COUNT = 1000
TRAINING_ROWS = 32768
BASE_ROWS = 1_000_000
TOP_K = 10
CURVE_K = 1000
RECALL_R = (1, 10, 20, 50, 100, 200, 500, 1000)
BATCH = 32
THREADS = 8
SEEDS = (0, 1, 2)
BUDGETS = (64, 80, 128, 256, 512)
CACHE_NAME = 'sift-1000000-1000-v2.npz'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def append_json(path, value):
    with path.open('a') as handle:
        handle.write(json.dumps(value) + '\n')


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_cache(data):
    with np.load(data / CACHE_NAME) as cache:
        base, queries, truth = cache['base'], cache['queries'], cache['truth']
    if base.shape != (BASE_ROWS, ORIGINAL_DIMENSIONS) or len(queries) != QUERY_COUNT:
        raise ValueError('unexpected SIFT split')
    return base, queries, truth


def read_transform(path):
    with np.load(path) as data:
        return L2Transform(float(data['scale']), data['base_map'], data['query_map'])


def output_arrays(rows, k=TOP_K):
    return np.empty((rows, k), np.float32), np.empty((rows, k), np.uint32)


def index_path(root, budget, seed, fast):
    return root / f'b{budget}-s{seed}-{"fast" if fast else "fp32"}.tq'
