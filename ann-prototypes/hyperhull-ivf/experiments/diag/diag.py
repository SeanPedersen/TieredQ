"""Tightness of outer-hull bounds on GloVe-100 (100k) at list and block granularity."""
import sys
from hyperhull_ivf.real_data import load_glove
from hyperhull_ivf.ivf import build_ivf
from hyperhull_ivf.hulls import subdivide_lists, build_hulls
from hyperhull_ivf.tightness import diagnose, print_diagnosis

d = load_glove(n_queries=100, n_base=100_000)
ivf = build_ivf(d.base, 256, 100_000)
for block_size in (64, 16):
    ivf2, layout = subdivide_lists(ivf, block_size)
    for r in (4, 16):
        print(f"block_size={block_size} R={r}")
        if block_size == 64:
            print_diagnosis("lists", diagnose(ivf2.vectors, ivf2.offsets, build_hulls(ivf2.vectors, ivf2.offsets, r), d.queries))
        print_diagnosis("blocks", diagnose(ivf2.vectors, layout.block_offsets, build_hulls(ivf2.vectors, layout.block_offsets, r), d.queries))
