"""Recall@10 and top-10 agreement of a candidate tree's IDs against the reference tree's IDs."""
import json, sys
from pathlib import Path
import numpy as np
from data import prepare
data, reference, candidate = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
for ref in sorted(reference.glob("*.npy")):
    a, b = np.load(ref), np.load(candidate / ref.name)
    _, _, truth = prepare(data, ref.stem.split("-")[0])
    recall = lambda ids: np.mean([len(set(r) & set(t[:10])) / 10 for r, t in zip(ids, truth)])
    print(json.dumps(dict(index=ref.stem, recall_reference=round(recall(a), 4), recall_candidate=round(recall(b), 4),
        identical_rows=int((a == b).all(axis=1).sum()), rows=len(a),
        set_overlap=round(float(np.mean([len(set(x) & set(y)) / 10 for x, y in zip(a, b)])), 5))))
