"""Prepare an isolated seeded/metric adapter without changing retained issue-15 artifacts."""
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parent
OLD = Path('/home/sean/experiments/TieredQ-gemini-turboquant-20261026-vDAm7R/TieredQ/benchmark/turboquant/native')
s = (ROOT/'qdrant/lib/quantization/src/quantile.rs').read_text()
old = 'seed_from_u64(0)'
assert s.count(old) == 1
s = s.replace(old, 'seed_from_u64(std::env::var("TQ_BENCH_SEED").unwrap_or_else(|_| "0".into()).parse().expect("benchmark seed"))')
(ROOT/'qdrant/lib/quantization/src/quantile.rs').write_text(s)
s = (OLD/'Cargo.toml').read_text().replace('../../../../qdrant', '../qdrant')
(ROOT/'native/Cargo.toml').write_text(s)
(ROOT/'native/Cargo.lock').write_bytes((OLD/'Cargo.lock').read_bytes())
patch = subprocess.run(['git','diff'],cwd=ROOT/'qdrant',capture_output=True,text=True,check=True)
(ROOT/'qdrant-overlay.patch').write_text(patch.stdout)
