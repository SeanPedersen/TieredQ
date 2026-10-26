"""Inspect the length-prefixed codec recipe without changing production serialization."""
import struct

import numpy as np


def parse_model(raw):
    """Read the current length-prefixed codec recipe and locate its level arrays."""
    data = bytes(raw)
    d, budget, centers, rotate, version, _, _, nt = struct.unpack_from('<8I', data)
    if version != 2:
        raise ValueError('unsupported model format')
    tiers = np.frombuffer(data, '<u4', nt * 5, 32).reshape(nt, 5).copy()
    pos = 32 + nt * 20
    arrays = []
    for _ in range(4):
        size, = struct.unpack_from('<Q', data, pos)
        pos += 8
        arrays.append(np.frombuffer(data, '<f4', size, pos).copy())
        pos += size * 4
    level_start = pos
    count, = struct.unpack_from('<I', data, pos)
    pos += 4
    levels = []
    for _ in range(count):
        size, = struct.unpack_from('<Q', data, pos)
        pos += 8
        levels.append(np.frombuffer(data, '<f4', size, pos).copy())
        pos += size * 4
    if pos != len(data) or count != 9:
        raise ValueError('unexpected model recipe')
    centroid, basis, mean, sigma = arrays
    return dict(prefix=data[:level_start], tiers=tiers, levels=levels,
                centers=centroid.reshape(centers, d), basis=basis.reshape(d, -1),
                mean=mean, sigma=sigma, budget=budget, rotate=rotate)

