"""Create a pinned, auditable RaBitQ include overlay without changing upstream."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REVISION = "ced3763857e7e4e48553951bf66fabc8517bc4f0"

def prepare(root, full_score=False):
    source = root / "RaBitQ-Library"
    if not source.exists():
        subprocess.run(["git", "clone", "https://github.com/VectorDB-NTU/RaBitQ-Library.git", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", REVISION], check=True)
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise RuntimeError(f"RaBitQ revision mismatch: {actual}")
    overlay = root / ("rabitq-full-score" if full_score else "rabitq-native")
    shutil.copytree(source / "include", overlay / "include", dirs_exist_ok=True)
    patches = {
        "utils/rotator.hpp": ("std::mt19937 gen(rd());", 'std::mt19937 gen(std::stoul(std::getenv("BENCH_SEED")));'),
    }
    if full_score:
        patches["index/ivf/ivf.hpp"] = ("if (lower_dist < distk)", "if (true)")
    hashes = {}
    for name, (old, new) in patches.items():
        path = overlay / "include/rabitqlib" / name
        text = path.read_text()
        if text.count(old) != 1:
            raise RuntimeError(f"Upstream patch context changed: {name}")
        path.write_text(text.replace(old, new))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (overlay / "patches.json").write_text(json.dumps(hashes, indent=2))
    return overlay
