# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py>=3.10", "numpy>=1.26"]
# ///
"""Validate a full HDF5 corpus and export a mapped base descriptor plus held-out queries."""
import argparse
import hashlib
import json
import time
import resource
import sys
from pathlib import Path
import h5py
import numpy as np
from vecs_io import write_records


def main():
    started=time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--calibration', type=int, default=250)
    parser.add_argument('--test', type=int, default=750)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--int8', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with args.source.open('rb') as source:
        while block := source.read(8 << 20): digest.update(block)
    checksum = digest.hexdigest()
    if checksum != args.expected_sha256: raise ValueError(f'Source SHA256 mismatch: {checksum}')
    with h5py.File(args.source, 'r') as source:
        base = source['train']
        if len(base.shape) != 2 or base.dtype != np.dtype('<f4'):
            raise ValueError('Expected little-endian FP32 train matrix')
        n, d = base.shape
        if n <= 10 or n > np.iinfo(np.int32).max: raise ValueError('Invalid base size')
        queries = np.asarray(source['test'], dtype=np.float32)
        truth64 = np.asarray(source['neighbors'], dtype=np.int64)
        if min(args.calibration, args.test) <= 0 or args.calibration + args.test > len(queries):
            raise ValueError('Insufficient independent queries')
        if queries.shape[1] != d or truth64.shape[0] != len(queries) or truth64.shape[1] < 10:
            raise ValueError('Incompatible truth or queries')
        if not np.isfinite(queries).all() or np.any(truth64 < 0) or np.any(truth64 >= n):
            raise ValueError('Invalid queries or truth IDs')
        offset = base.id.get_offset()
        direct = offset is not None and base.chunks is None
        backing = args.source.resolve() if direct else (args.output / 'base.f32').resolve()
        norm_min, norm_max = float('inf'), 0.0
        minimum = np.full(d,np.inf,dtype=np.float32); maximum = np.full(d,-np.inf,dtype=np.float32)
        output = None if direct else backing.open('xb')
        try:
            for start in range(0, n, 4096):
                chunk = np.asarray(base[start:start + 4096], dtype='<f4')
                if not np.isfinite(chunk).all(): raise ValueError('Nonfinite base')
                if args.int8:
                    minimum=np.minimum(minimum,chunk.min(axis=0)); maximum=np.maximum(maximum,chunk.max(axis=0))
                norms = np.einsum('ij,ij->i', chunk, chunk)
                norm_min = min(norm_min, float(norms.min())); norm_max = max(norm_max, float(norms.max()))
                if output: output.write(chunk.tobytes())
        finally:
            if output: output.close()
        if not direct: offset = 0
        quantization = None
        if args.int8:
            scales = np.maximum((maximum-minimum) / np.float32(255), np.finfo(np.float32).tiny).astype('<f4')
            offsets=(minimum+np.float32(128)*scales).astype('<f4')
            offsets.tofile(args.output/'offsets.f32')
            scales.tofile(args.output / 'scales.f32')
            code_hash = hashlib.sha256(); squared_error = 0.0; max_error = 0.0
            with (args.output / 'base.i8').open('xb') as encoded:
                for start in range(0, n, 4096):
                    chunk = np.asarray(base[start:start + 4096], dtype=np.float32)
                    code = np.clip(np.floor((chunk-offsets) / scales + np.float32(.5)), -128, 127).astype('int8')
                    payload = code.tobytes(); encoded.write(payload); code_hash.update(payload)
                    error = chunk - (code.astype(np.float32) * scales+offsets)
                    squared_error += float(np.einsum('ij,ij->', error.astype(np.float64), error.astype(np.float64)))
                    max_error = max(max_error, float(np.abs(error).max()))
            quantization = dict(kind='affine per-dimension SQ8', codes_dtype='int8', codes_shape=[n,d],
                                codes_bytes=n*d, codes_sha256=code_hash.hexdigest(), scales_sha256=hashlib.sha256(scales.tobytes()).hexdigest(), offsets_sha256=hashlib.sha256(offsets.tobytes()).hexdigest(),
                                encoding='clip(floor((x-offset) / scale + 0.5), -128, 127)', decoding='float32(code) * scale + offset',
                                scale_range=[float(scales.min()),float(scales.max())],
                                mean_squared_error=squared_error/(n*d), maximum_absolute_error=max_error)
        attributes = {k: (v.item() if isinstance(v, np.generic) else v) for k,v in source.attrs.items()}
    descriptor = args.output / 'base.rawview'
    descriptor.write_text(f'geometry_raw_f32_v1 {n} {d} {offset} {json.dumps(str(backing), ensure_ascii=False)}\n')
    order = np.random.default_rng(args.seed).permutation(len(queries))
    calibration = order[:args.calibration]; test = order[args.calibration:args.calibration + args.test]
    hashes = {}
    truth = truth64[:, :10].astype('<i4')
    for prefix, ids in [('calibration', calibration), ('test', test)]:
        for suffix, values, dtype in [('queries.fvecs', queries[ids], '<f4'), ('truth.ivecs', truth[ids], '<i4')]:
            name = f'{prefix}_{suffix}'
            hashes[name] = write_records(args.output / name, values, dtype)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    manifest = dict(preparation_seconds=time.perf_counter()-started,preparation_peak_rss_bytes=peak,source=str(args.source.resolve()), source_sha256=checksum, base_shape=[n,d],
                    base_bytes=n*d*4, direct_hdf5_mapping=direct, payload_offset=offset,
                    source_attributes=attributes, quantization=quantization, base_squared_norm_range=[norm_min,norm_max],
                    query_squared_norm_range=[float(np.square(queries).sum(axis=1).min()),float(np.square(queries).sum(axis=1).max())],
                    metric='squared L2 on supplied normalized vectors' if attributes.get('distance')=='normalized' else 'squared L2', normalization='none added',
                    seed=args.seed, calibration_query_ids=calibration.tolist(), test_query_ids=test.tolist(),
                    ground_truth='supplied exact neighbors, first 10', sha256=hashes)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k:manifest[k] for k in ['base_shape','base_bytes','direct_hdf5_mapping','base_squared_norm_range']}, indent=2),flush=True)

if __name__ == '__main__': main()
