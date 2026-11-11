"""Prints every model's held-out recall@10 and single-query p50 per setting, cap 4 next to the automatic width."""
import json
import sys
from pathlib import Path

rows = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else 'results').glob('*/run.json'))]
for row in sorted(rows, key=lambda r: (r['dim'], r['kind'], r['bits_per_dim'], r['seed'], r['max_scan_bits'] == 4)):
    width = 'cap 4' if row['max_scan_bits'] == 4 else 'auto (8)'
    points = ', '.join(f"{'' if p['nprobe'] is None else 'np' + str(p['nprobe']) + ' '}{p['recall']:.4f} @ {p['p50_ms']:.2f} ms"
                       for p in row['points'])
    print(f"{row['dim']:>5}d {row['kind']:<4} {row['bits_per_dim']:g} b/d seed {row['seed']} {width:<8} "
          f"{row['storage_bytes'] / 2**20:7.1f} MiB  {points}")
