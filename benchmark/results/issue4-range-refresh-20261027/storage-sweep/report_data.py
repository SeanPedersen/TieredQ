"""Load per-schedule measured operating points without interpolating or mixing timing rows."""
from dataclasses import dataclass
import csv
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
DATASET_NAMES = {'sift': 'SIFT1M', 'glove': 'GloVe-200', 'openai': 'OpenAI3-1536', 'gemini': 'DBpedia Gemini-768'}
CODEC_NAMES = {'tieredq': 'TieredQ fast', 'rabitq': 'RaBitQ'}
MAIN_BUDGETS = (1, 2, 4)


@dataclass(frozen=True)
class Point:
    dataset: str
    codec: str
    code_bits: int
    nominal_bits: float
    anchor_bits: int | None
    threads: int
    batch: int
    recall: float
    std: float
    storage: float
    qps: float
    p50: float
    p95: float
    build_s: float
    encode_vps: float
    supplemental: bool


def load_points(path, supplemental=False, default_anchor=None):
    rows = list(csv.DictReader(path.open()))
    groups = {}
    for row in rows:
        code_bits = int(row.get('code_bits') or round(float(row['target_bits']) * int(row['d'])))
        key = (row['dataset'], row['codec'], code_bits, int(row['threads']), int(row['batch']))
        groups.setdefault(key, []).append(row)
    result = []
    for (dataset, codec, code_bits, threads, batch), group in groups.items():
        assert len(group) == 3 and {r['seed'] for r in group} == {'0', '1', '2'}
        seed0 = next(r for r in group if r['seed'] == '0')
        recalls = [float(r['recall10']) for r in group]
        anchor = int(seed0['anchor_bits']) if seed0.get('anchor_bits') else default_anchor
        result.append(Point(dataset, codec, code_bits, float(seed0['target_bits']), anchor,
                            threads, batch, statistics.mean(recalls), statistics.pstdev(recalls),
                            float(seed0['effective_bytes_per_vector']), float(seed0['qps']),
                            float(seed0['p50_ms']), float(seed0['p95_ms']), float(seed0['build_s']),
                            float(seed0['encode_vps']), supplemental))
    return result


def all_points():
    return (load_points(ROOT / 'output/results.csv') +
            load_points(ROOT / 'matched-storage/results.csv', True, 2) +
            load_points(ROOT / 'storage-sweep/results.csv', True))


def serving_points(points):
    return [p for p in points if p.threads == 8 and p.batch == 32]


def anchor_point(points, dataset, bits):
    return next(p for p in points if p.dataset == dataset and p.codec == 'rabitq' and p.nominal_bits == bits and not p.supplemental)


def schedule(points, point, threads, batch):
    return next(p for p in points if (p.dataset, p.codec, p.code_bits, p.supplemental, p.threads, p.batch) ==
                (point.dataset, point.codec, point.code_bits, point.supplemental, threads, batch))
