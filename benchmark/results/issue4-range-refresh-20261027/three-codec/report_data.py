"""Combine audited main, RaBitQ-cap, and TurboQuant-cap measurements by scheduling row."""
from pathlib import Path
import importlib.util
import sys
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prior_report_data',HERE.parent/'storage-sweep/report_data.py')
prior=importlib.util.module_from_spec(spec);sys.modules[spec.name]=prior;spec.loader.exec_module(prior)
DATASET_NAMES=prior.DATASET_NAMES
CODEC_NAMES=prior.CODEC_NAMES|{'turboquant':'Qdrant TurboQuant+'}
MAIN_BUDGETS=prior.MAIN_BUDGETS
schedule=prior.schedule


def all_points():
    rows=prior.load_points(HERE/'results.csv')
    from dataclasses import replace
    return prior.all_points()+[replace(p,supplemental=p.codec=='tieredq') for p in rows]


def label(point):
    if not point.supplemental:return f'{point.nominal_bits:g}b'
    new=point.codec=='tieredq' and any(p.dataset==point.dataset and p.code_bits==point.code_bits for p in prior.load_points(HERE/'results.csv'))
    return f'{"Q" if new else "M"}{point.anchor_bits}'
