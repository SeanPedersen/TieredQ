"""Aggregate paired codebook results and verify fixed-model experiment invariants."""
import argparse
import json
from pathlib import Path

import numpy as np

from run import parse_model


def summarize(root):
    rows = [json.loads(line) for line in (root / 'results.jsonl').read_text().splitlines()]
    groups = {}
    for row in rows:
        key = (row['dataset'], row['bits_per_original_dimension'], row['rotate'], row['fast'])
        groups.setdefault(key, {}).setdefault(row['seed'], {})[row['variant']] = row
    output = []
    for key, seeds in sorted(groups.items()):
        gaussian, learned = [], []
        for seed, variants in sorted(seeds.items()):
            if set(variants) != {'gaussian', 'learned'}:
                raise ValueError(f'incomplete pair: {key}, seed {seed}')
            a, b = variants['gaussian'], variants['learned']
            for field in ['tiers', 'vector_storage_bytes', 'model_memory_bytes', 'base_rows', 'queries']:
                if a[field] != b[field]:
                    raise ValueError(f'changed {field}: {key}, seed {seed}')
            stem = f'{key[0]}-s{seed}-b{key[1]}-r{int(key[2])}'
            raw_a = (root / f'{stem}-gaussian.bin').read_bytes()
            raw_b = (root / f'{stem}-learned.bin').read_bytes()
            if len(raw_a) != len(raw_b) or parse_model(raw_a)['prefix'] != parse_model(raw_b)['prefix']:
                raise ValueError('model changed beyond codebooks')
            gaussian.append(a)
            learned.append(b)
        deltas = [b['recall10']-a['recall10'] for a,b in zip(gaussian, learned)]
        result = dict(dataset=key[0], bits=key[1], rotate=key[2], fast=key[3], seeds=sorted(seeds),
                      gaussian_recall=float(np.mean([a['recall10'] for a in gaussian])),
                      learned_recall=float(np.mean([b['recall10'] for b in learned])),
                      recall_delta_pp=float(np.mean(deltas)*100), recall_deltas_pp=[d*100 for d in deltas],
                      mse_change_percent=float(np.mean([100*(b['reconstruction_mse']/a['reconstruction_mse']-1) for a,b in zip(gaussian, learned)])),
                      qps_change_percent=float(np.mean([100*(b['qps']/a['qps']-1) for a,b in zip(gaussian, learned)])),
                      mean_fit_s=float(np.mean([b['fit_s'] for b in learned])),
                      mean_train_s=float(np.mean([b['train_s'] for b in learned])))
        output.append(result)
    (root / 'summary.json').write_text(json.dumps(output, indent=2)+'\n')
    print('dataset bits rotate fast Gaussian Learned delta_pp MSE_change% QPS_change%')
    for row in output:
        print(f"{row['dataset']:7} {row['bits']} {int(row['rotate'])} {int(row['fast'])} "
              f"{row['gaussian_recall']:.5f} {row['learned_recall']:.5f} "
              f"{row['recall_delta_pp']:+.3f} {row['mse_change_percent']:+.2f} {row['qps_change_percent']:+.2f}")
    print(f'Validated {len(rows)} rows and {len(rows)//2} paired evaluations.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    summarize(parser.parse_args().results)
