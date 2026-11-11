"""Attributes resident heap a TieredQ build leaves behind (#43, #41). Builds a flat IVF or a nested IVF (+ write_tier)
on the first ROWS rows of VIBE arxiv on BUILD_CORES threads, after a small warm-up build so one-time process state
precedes the baseline. After each step it records USS, private KiB of [heap] and of all anonymous regions, and glibc's
per-arena statistics (malloc_info XML: in-use vs free bytes per arena). Then it calls malloc_trim(0) once more and
records the same: whatever that second trim returns was freed after the library's own trim (ordering), whatever
survives it is live or unreleasable fragmentation.

WARMUP=0 skips the warm-up build (a cold process, as VIBE measures). EXIT_AFTER=1 ends the process with os._exit right
after the last snapshot, so heaptrack's "leaked" allocations are the ones live at that point.

Usage: BUILD_CORES=.. heap_probe.py DATASET.hdf5 OUT.json flat|nested ROWS [LISTS]"""
import ctypes
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import psutil

KIB = 1024
WARMUP_ROWS, WARMUP_LISTS = 65536, 256
FLAT_BITS_PER_DIM = 1.75
libc = ctypes.CDLL("libc.so.6")


def private_kib():
    heap, anon, key = 0, 0, None
    for line in open("/proc/self/smaps"):
        fields = line.split()
        if "-" in fields[0] and not fields[0].endswith(":"):
            key = fields[5] if len(fields) > 5 else "anon"
        elif fields[0] in ("Private_Clean:", "Private_Dirty:"):
            if key == "[heap]":
                heap += int(fields[1])
            elif key == "anon":
                anon += int(fields[1])
    return heap, anon


def arenas():
    """Per-arena (in use, free, mmapped region total) KiB from malloc_info."""
    path = tempfile.mktemp()
    stream = libc.fopen(path.encode(), b"w")
    libc.malloc_info(0, ctypes.c_void_p(stream))
    libc.fclose(ctypes.c_void_p(stream))
    xml = Path(path).read_text()
    os.unlink(path)
    result = []
    for heap in re.findall(r'<heap nr="\d+">(.*?)</heap>', xml, re.S):
        free = sum(int(size) for size in re.findall(r'<total type="(?:fast|rest)" count="\d+" size="(\d+)"', heap))
        current = int(re.search(r'<system type="current" size="(\d+)"', heap).group(1))
        result.append(dict(system_kib=current // KIB, free_kib=free // KIB, in_use_kib=(current - free) // KIB))
    return result


def snapshot(label, baseline):
    heap, anon = private_kib()
    per_arena = arenas()
    return dict(step=label, uss_kib=psutil.Process().memory_full_info().uss / KIB - baseline["uss"],
                heap_private_kib=heap - baseline["heap"], anon_private_kib=anon - baseline["anon"], arenas=len(per_arena),
                arena_in_use_kib=sum(a["in_use_kib"] for a in per_arena), arena_free_kib=sum(a["free_kib"] for a in per_arena),
                main_arena=per_arena[0], thread_arenas_in_use_kib=sorted((a["in_use_kib"] for a in per_arena[1:]), reverse=True)[:6])


def build(kind, rows, lists, tier_path):
    from tieredq import TieredQIVFIndex, TieredQNestedIVFIndex

    if kind == "flat":
        index = TieredQIVFIndex(dimensions=rows.shape[1], code_bits=round(FLAT_BITS_PER_DIM * rows.shape[1]), lists=lists,
                                metric="ip")
        index.build(rows)
        return index, []
    index = TieredQNestedIVFIndex.build(rows, lists=lists, metric="ip")
    return index, [("write_tier", lambda: index.write_tier(tier_path, rows))]


def main():
    dataset, out_path, kind, count = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    lists = int(sys.argv[5]) if len(sys.argv) > 5 else None
    os.sched_setaffinity(0, {int(c) for c in os.environ["BUILD_CORES"].split(",")})
    with h5py.File(dataset, "r") as f:
        rows = np.ascontiguousarray(f["train"][:count], dtype=np.float32)
    tier_path = Path(os.environ.get("TIER_DIR", ".")) / f"probe-tier-{os.getpid()}.rows"
    if os.environ.get("WARMUP", "1") != "0":
        warm, warm_steps = build(kind, np.ascontiguousarray(rows[:WARMUP_ROWS]), WARMUP_LISTS, tier_path)
        for _, step in warm_steps:
            step()
        del warm
        tier_path.unlink(missing_ok=True)
        libc.malloc_trim(0)
    heap, anon = private_kib()
    baseline = dict(uss=psutil.Process().memory_full_info().uss / KIB, heap=heap, anon=anon)
    steps = [snapshot("start", baseline)]
    index, later = build(kind, rows, lists, tier_path)
    steps.append(snapshot("build", baseline))
    for label, step in later:
        step()
        steps.append(snapshot(label, baseline))
    libc.malloc_trim(0)
    steps.append(snapshot("second_trim", baseline))
    tier_path.unlink(missing_ok=True)
    report = dict(kind=kind, rows=count, lists=index.lists, storage_bytes=index.storage_bytes, steps=steps)
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    for step in steps:
        print(json.dumps(step))
    print(f"storage_bytes {index.storage_bytes / KIB:.0f} KiB, lists {index.lists}", flush=True)
    if os.environ.get("EXIT_AFTER") == "1":
        os._exit(0)


if __name__ == "__main__":
    main()
