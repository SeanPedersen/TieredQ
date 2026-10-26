"""Uniform compressed-only search contracts and measured persistent serving storage."""
import ctypes as ct
import hashlib
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'build'))
import tieredq_bench


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Qdrant:
    codec = 'qdrant-turboquant-plus'
    lib = ct.CDLL(str(HERE/'native/target/release/libqdrant_bench.so'))
    lib.tq_build.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_uint32, ct.c_char_p]
    lib.tq_build.restype = ct.c_void_p
    lib.tq_load.argtypes = [ct.c_char_p]
    lib.tq_load.restype = ct.c_void_p
    lib.tq_stats.argtypes = [ct.c_void_p, ct.c_void_p]
    lib.tq_search.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]
    lib.tq_free.argtypes = [ct.c_void_p]
    lib.tq_validate.argtypes = [ct.c_void_p, ct.c_void_p]
    lib.tq_validate.restype = ct.c_bool

    def __init__(self, base, bits, path):
        self.path = Path(path)
        tick = time.perf_counter()
        pointer = self.lib.tq_build(base.ctypes.data, len(base), base.shape[1], bits, str(path).encode())
        self.build_s = time.perf_counter()-tick
        self.pointer = pointer
        self.encode_seconds = self._stats()[3]
        self.dim = base.shape[1]

    def _stats(self):
        values = np.empty(5, dtype=np.float64)
        self.lib.tq_stats(self.pointer, values.ctypes.data)
        return values.tolist()

    def reload(self):
        self.lib.tq_free(self.pointer)
        self.pointer = self.lib.tq_load(str(self.path).encode())

    def details(self):
        codes, model, shell, _, stride = self._stats()
        files = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in self.path.iterdir()}
        disk = sum(x['bytes'] for x in files.values())
        resident = int(codes+model+shell)
        return dict(disk_bytes=disk, resident_bytes=resident, serving_bytes=max(disk,resident),
                    codes_allocated_bytes=int(codes), rotation_calibration_bytes=int(model),
                    shell_bytes=int(shell), vector_bytes=int(stride), files=files)

    def search(self, queries, output, threads):
        assert queries.dtype == np.float32 and queries.flags.c_contiguous
        assert output.dtype == np.uint32 and output.shape == (len(queries), 10) and output.flags.c_contiguous
        assert threads in (1,8) and queries.shape[1] == self.dim
        self.lib.tq_search(self.pointer, queries.ctypes.data, len(queries), output.ctypes.data, threads)

    def close(self):
        if self.pointer:
            self.lib.tq_free(self.pointer)
            self.pointer = None


class TieredQ:
    codec = 'tieredq-fast'

    def __init__(self, base, budget, path, validate_reload=False):
        self.path, self.dim = Path(path), base.shape[1]
        tieredq_bench.set_threads(8)
        tick = time.perf_counter()
        self.index = tieredq_bench.Index(base, budget)
        self.build_s = time.perf_counter()-tick
        self.encode_seconds = self.index.encode_seconds
        self.index.save(str(path))
        if validate_reload:
            before = np.empty((65,10), dtype=np.uint32)
            self.index.search(base[:65], before, 1)
        self.reload()
        if validate_reload:
            after = np.empty((65,10), dtype=np.uint32)
            self.index.search(base[:65], after, 8)
            assert np.array_equal(before, after), 'Built/reloaded TieredQ differs'


    def reload(self):
        self.index = tieredq_bench.Index(str(self.path), self.dim)

    def details(self):
        details = self.index.details()
        details.update(disk_bytes=self.path.stat().st_size, file_sha256=sha(self.path))
        details['serving_bytes'] = max(details['disk_bytes'], details['resident_bytes'])
        return details

    def search(self, queries, output, threads):
        self.index.search(queries, output, threads)

    def close(self):
        self.index = None
