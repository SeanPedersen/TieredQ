"""Qdrant TurboQuant+ (production `quantization` crate) behind the codec_native.Index interface.

The Rust shim in codec_adapters/qdrant_turboquant_plus/native builds, saves, reloads and exhaustively scores TQ+ codes
(cosine, or L2 for SIFT) without a Qdrant server. An index is a directory holding
`metadata.json` and `codes.bin`. TQ+ calibration is seeded inside the patched crate and its
rotation seeds are fixed upstream, so the benchmark seed does not change the codec.
Parallelism is across queries: a single-query search runs on one thread.
"""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

HERE = Path(__file__).resolve().parent
DEFAULT_LIBRARY = HERE/"codec_adapters/qdrant_turboquant_plus/native/target/release/libqdrant_bench.so"
SUPPORTED_BITS = (1, 2, 4)
# Allocated codes, quantizer heap (rotation/calibration), Index struct, encode seconds, stride.
STAT_FIELDS = 5
CONFIG = "qdrant-turboquant-plus"


def _library():
    library = ct.CDLL(os.environ.get("QDRANT_TQ_LIB", str(DEFAULT_LIBRARY)))
    library.tq_build.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_uint32, ct.c_bool, ct.c_char_p]
    library.tq_build.restype = ct.c_void_p
    library.tq_load.argtypes = [ct.c_char_p]
    library.tq_load.restype = ct.c_void_p
    library.tq_stats.argtypes = [ct.c_void_p, ct.c_void_p]
    library.tq_search.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]
    library.tq_free.argtypes = [ct.c_void_p]
    return library


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).iterdir())


def directory_sha256(path):
    digest = hashlib.sha256()
    for p in sorted(Path(path).iterdir()):
        digest.update(p.name.encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


class QdrantIndex:
    library = None

    def __init__(self, pointer, directory, encode_seconds, workspace=None):
        self.pointer, self.directory, self.encode_seconds = pointer, Path(directory), encode_seconds
        self._workspace = workspace
        self.dim = None

    @classmethod
    def _lib(cls):
        if cls.library is None:
            cls.library = _library()
        return cls.library

    @classmethod
    def build(cls, vectors, bits, l2):
        import numpy as np
        if bits not in SUPPORTED_BITS:
            raise ValueError(f"TQ+ supports {SUPPORTED_BITS} bits, not {bits}")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        workspace = tempfile.TemporaryDirectory()
        directory = Path(workspace.name)/"index"
        pointer = cls._lib().tq_build(vectors.ctypes.data, len(vectors), vectors.shape[1], bits, l2,
                                      str(directory).encode())
        index = cls(pointer, directory, 0.0, workspace)
        index.encode_seconds = index._stats()[3]
        index.dim = vectors.shape[1]
        return index

    @classmethod
    def load(cls, path):
        pointer = cls._lib().tq_load(str(path).encode())
        return cls(pointer, path, 0.0)

    def _stats(self):
        import numpy as np
        values = np.empty(STAT_FIELDS, dtype=np.float64)
        self._lib().tq_stats(self.pointer, values.ctypes.data)
        return values.tolist()

    def save(self, path):
        """Copy the index directory to `path`; return its total bytes."""
        path = Path(path)
        if path.resolve() != self.directory.resolve():
            shutil.copytree(self.directory, path, dirs_exist_ok=True)
        return directory_bytes(path)

    def storage(self, path):
        return self.save(path)

    def resident_bytes(self):
        codes, model, shell, _, _ = self._stats()
        return int(codes + model + shell)

    def search(self, queries, output, threads):
        assert queries.flags.c_contiguous and output.flags.c_contiguous and output.shape == (len(queries), 10)
        self._lib().tq_search(self.pointer, queries.ctypes.data, len(queries), output.ctypes.data, threads)

    def __del__(self):
        if self.pointer and self.library is not None:
            self.library.tq_free(self.pointer)
            self.pointer = None
