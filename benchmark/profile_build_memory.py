"""RSS timeline of one TieredQ build, to attribute the build peak to training/encoding stages.

A sampler thread reads VmRSS every SAMPLE_S while the native constructor runs with the GIL released. The
native StageProfile (DV_TIEREDQ_PROFILE) prints per-stage seconds to stderr, which lets the timeline be
aligned to stages. RSS is reported as growth over the loaded FP32 corpus, as in measure_memory.py.
"""
import argparse
import json
import os
from pathlib import Path
import threading
import time

SAMPLE_S = 0.001
BUILD_THREADS = 8


def rss_bytes():
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def sample(stop, timeline, start, baseline):
    while not stop.is_set():
        timeline.append((time.perf_counter() - start, rss_bytes() - baseline))
        time.sleep(SAMPLE_S)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="openai")
    parser.add_argument("--bits", type=int, default=1)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["DV_TIEREDQ_PROFILE"] = "1"
    from threadpoolctl import threadpool_limits
    import codec_native
    from data import prepare
    from measure_memory import reset_peak, memory_status
    with threadpool_limits(limits=1):
        base, _, _ = prepare(args.data, args.dataset)
        codec_native.set_threads(BUILD_THREADS)
        baseline = reset_peak()
        timeline, stop = [], threading.Event()
        start = time.perf_counter()
        sampler = threading.Thread(target=sample, args=(stop, timeline, start, baseline))
        sampler.start()
        index = codec_native.Index(base, args.bits, 0, True, False)
        stop.set()
        sampler.join()
        peak = memory_status()[1] - baseline
    args.output.write_text(json.dumps(dict(dataset=args.dataset, bits=args.bits, base_n=len(base), d=base.shape[1],
                                           peak_bytes=peak, encode_s=index.encode_seconds,
                                           timeline=timeline), indent=1))
    print(f"peak {peak / 2**20:.1f} MiB, {peak * 8 / len(base):,.0f} bits/vector, {len(timeline)} samples")


if __name__ == "__main__":
    main()
