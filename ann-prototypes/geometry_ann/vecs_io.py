"""Shared exclusive vecs writer for reproducible benchmark exports."""
import hashlib
from pathlib import Path
import numpy as np

IO_CHUNK_ROWS = 4096

def write_records(path: Path, data: np.ndarray, dtype: str) -> str:
    """Write little-endian vecs records without overwriting existing artifacts."""
    digest = hashlib.sha256()
    with path.open("xb") as output:
        for start in range(0, len(data), IO_CHUNK_ROWS):
            chunk = np.asarray(data[start : start + IO_CHUNK_ROWS], dtype=dtype)
            packed = np.empty((len(chunk), chunk.shape[1] + 1), dtype="<i4")
            packed[:, 0] = chunk.shape[1]
            packed[:, 1:] = chunk.view("<i4")
            payload = packed.tobytes()
            output.write(payload)
            digest.update(payload)
    return digest.hexdigest()


