"""Prints each model's USS (GiB), USS per step, glibc heap, largest grown mappings and its Recall@100 / QPS / QPS/GB curve."""
import json, sys
from pathlib import Path

GIB_KIB = 2 ** 20
for result in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else 'results').glob('*/results.jsonl')):
    rows = [json.loads(line) for line in result.open()]
    info = rows[0]['info']
    size = rows[0]['index_size'] / GIB_KIB
    print(f"{result.parent.name}: {size:.4f} GiB, lists {info['lists']}, storage {info['storage_bytes'] / 2**30:.4f} GiB, "
          f"serialized {info['serialized_bytes'] / 2**30:.4f} GiB, build {rows[0]['build_time']:.1f} s")
    print(f"  USS per step KiB {info['uss_steps_kib']}  heap KiB {info.get('heap_kib')}")
    growth = info['mapping_growth_kib']
    arenas = [kib for key, kib in growth.items() if key.startswith('anon@') and 2000 <= kib <= 2600]
    print(f"  grown: {list(growth.items())[:6]} ... {len(arenas)} regions of 2.0-2.6 MiB = {sum(arenas) / 1024:.1f} MiB")
    for row in rows:
        print(f"    nprobe {row['query'][0]:>4} cand {row['query'][1]}: recall {row['recall']:.4f}  {row['qps']:.0f} QPS  "
              f"{row['qps'] / size:.0f} QPS/GB")
