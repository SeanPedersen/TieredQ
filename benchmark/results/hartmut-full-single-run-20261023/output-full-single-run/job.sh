#!/usr/bin/env bash
set -euo pipefail
cd /home/sean/experiments/TieredQ-issue4-20261023/TieredQ
out=benchmark/output-full-single-run
trap 'code=$?; printf "%s\n" "$code" > "$out/exit-code"; date -Is; echo "JOB_EXIT_CODE=$code"' EXIT
printf 'Validation started: '; date -Is
cd benchmark
./validate.sh 2>&1 | tee output-full-single-run/validation.log
cd ..
git status --porcelain > "$out/git-status-before.txt"
git diff --binary > "$out/tracked-changes.patch"
git rev-parse HEAD > "$out/checkout-revision.txt"
python3 - <<'PY'
import pathlib,tarfile,hashlib,json
root=pathlib.Path('.')
files=[p for folder in ('src','include','benchmark/native') for p in (root/folder).rglob('*') if p.is_file()]
files += [p for p in (root/'benchmark').iterdir() if p.is_file() and p.suffix in ('.py','.sh','.toml','.lock','.json','.md')]
files += [root/'CMakeLists.txt',root/'SOURCE_REVISION',root/'README.md']
files=[p for p in files if p.is_file()]
out=root/'benchmark/output-full-single-run'
(out/'validated-source-sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},indent=2))
with tarfile.open(out/'as-run-sources.tar.gz','w:gz') as archive:
    for p in sorted(files): archive.add(p,arcname=str(p))
PY
printf 'Matrix started: '; date -Is
cd benchmark
taskset -c 0-7 ./run.sh --seeds 0 --output output-full-single-run
