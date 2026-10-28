"""Write top-10 IDs (8 threads, one batch) of a saved TieredQ index to an .npy file."""
import sys
from pathlib import Path
import numpy as np
import codec_native
from data import prepare
data, path, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
_, queries, _ = prepare(data, Path(path).stem.split("-")[0])
index = codec_native.Index.load(path, True)
ids = np.empty((len(queries), 10), dtype=np.uint32)
index.search(queries, ids, 8)
np.save(out, ids)
