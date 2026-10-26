"""Fit a metric-preserving L2 lift using base vectors as a query-distribution proxy."""
from dataclasses import dataclass
import io

import numpy as np

CALIBRATION_ROWS = 32768


@dataclass(frozen=True)
class L2Transform:
    scale: float
    base_map: np.ndarray
    query_map: np.ndarray

    @classmethod
    def fit(cls, base):
        scale = float(np.max(np.linalg.norm(base, axis=1))) or 1.0
        sample = (base[:CALIBRATION_ROWS] / scale).astype(np.float64)
        queries = np.column_stack([sample, np.ones(len(sample))])
        moment = queries.T @ queries / len(queries)
        precision = np.float32 if np.linalg.matrix_rank(moment) < len(moment) else np.float64
        ridge = np.finfo(precision).eps * np.trace(moment)
        identity = np.eye(len(moment))
        try:
            transform = np.linalg.cholesky(moment + identity * ridge)
        except np.linalg.LinAlgError:
            transform = np.linalg.cholesky(moment + identity * np.finfo(np.float32).eps * np.trace(moment))
        return cls(scale, transform.astype(np.float32), np.linalg.inv(transform).T.astype(np.float32))

    def base_vectors(self, base):
        vectors = base / self.scale
        lifted = np.column_stack([vectors, -np.sum(vectors * vectors, axis=1) / 2])
        return np.ascontiguousarray(lifted @ self.base_map, dtype=np.float32)

    def queries(self, queries):
        lifted = np.column_stack([queries / self.scale, np.ones(len(queries), dtype=np.float32)])
        return np.ascontiguousarray(lifted @ self.query_map, dtype=np.float32)

    @property
    def storage_bytes(self):
        serialized = io.BytesIO()
        np.savez(serialized, scale=self.scale, base_map=self.base_map, query_map=self.query_map)
        return max(serialized.tell(), self.base_map.nbytes + self.query_map.nbytes + np.dtype(np.float64).itemsize)
