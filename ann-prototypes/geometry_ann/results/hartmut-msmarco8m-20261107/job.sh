#!/bin/bash
# Full-corpus job; only task-owned paths are created. Source SHA256 is mandatory.
set -euo pipefail
ROOT=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
BIG=/mnt/storage/tieredq-issue44-msmarco8m-20261107
cd "$ROOT"
trap 'code=$?; printf "%s\n" "$code" >job.exitcode' EXIT
python3 - <<'PY'
import datetime,json
from pathlib import Path
Path('status.json').write_text(json.dumps(dict(stage='waiting_for_download',utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
PY
native="$BIG/xet-download/msmarco-qwen-1024-normalized.hdf5"
if [ -f "$BIG/msmarco-qwen-1024-normalized.hdf5" ]; then native="$BIG/msmarco-qwen-1024-normalized.hdf5"; fi
while [ ! -f "$native" ]; do
  if ! pgrep -f '/bin/hf download vector-index-bench/vibe msmarco-qwen-1024-normalized.hdf5' >/dev/null; then
    if [ -f "$native" ]; then break; fi
    resume_file=$(python3 "$ROOT/choose_resume.py" "$BIG")
    cd "$BIG"
    aria2c --continue=true --max-connection-per-server=4 --split=4 --min-split-size=16M \
      --file-allocation=none --console-log-level=warn --summary-interval=30 \
      --checksum=sha-256=efa21f8418e29137ad5be44171fd4657e1738caee86eb62645ad20605457cb2c \
      --dir=. --out="$resume_file" \
      https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/msmarco-qwen-1024-normalized.hdf5 \
      >"$ROOT/download-fallback.log" 2>&1
    mv "$resume_file" msmarco-qwen-1024-normalized.hdf5
    native="$BIG/msmarco-qwen-1024-normalized.hdf5"
    cd "$ROOT"
    break
  fi
  sleep 20
done
python3 - <<'PY'
import datetime,json
from pathlib import Path
Path('status.json').write_text(json.dumps(dict(stage='validating_and_preparing_dataset',utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
PY
uv run --python 3.12 source/prepare_hdf5_view.py "$native" "$BIG/export" \
  --calibration 250 --test 750 --int8 \
  --expected-sha256 efa21f8418e29137ad5be44171fd4657e1738caee86eb62645ad20605457cb2c >prepare.log 2>&1
python3 - <<'PY2'
import shutil
from pathlib import Path
source=Path('/mnt/storage/tieredq-issue44-msmarco8m-20261107/export/base.i8')
root=Path('/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107')
if shutil.disk_usage(root).free < source.stat().st_size+(2<<30):
    raise RuntimeError('Insufficient NVMe space for SQ8 dataset plus 2 GiB headroom')
PY2
rsync -a "$BIG/export/base.i8" "$BIG/export/scales.f32" "$BIG/export/offsets.f32" "$ROOT/"
python3 - <<'PY2'
import hashlib,json
from pathlib import Path
root=Path('/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107')
manifest=json.loads(Path('/mnt/storage/tieredq-issue44-msmarco8m-20261107/export/manifest.json').read_text())
h=hashlib.sha256()
with (root/'base.i8').open('rb') as f:
    while b:=f.read(8<<20): h.update(b)
if h.hexdigest()!=manifest['quantization']['codes_sha256']: raise RuntimeError('NVMe SQ8 copy failed SHA256 verification')
for name,key in [('scales.f32','scales_sha256'),('offsets.f32','offsets_sha256')]:
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=manifest['quantization'][key]: raise RuntimeError('NVMe SQ8 parameter copy failed verification')
PY2
ln -s "$BIG/export" data
{ hostname; date -u; c++ --version; lscpu; free -b; df -B1 /mnt/storage; sha256sum source/index.cpp source/scan_distance.cpp; } >environment.txt
python3 study.py >study.log 2>&1
