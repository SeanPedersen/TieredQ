"""Wait for matrix completion, then validate adapters and run the serial control experiment."""
import os
from pathlib import Path
import subprocess
import time
root=Path(__file__).resolve().parent
started=time.monotonic()
while not (root/"exit-code").exists():
    if time.monotonic()-started>3600:
        raise TimeoutError("matrix completion marker missing after one hour")
    time.sleep(10)
assert (root/"exit-code").read_text().strip()=="0", "main matrix failed"
python=root/"source/benchmark/.venv/bin/python"
subprocess.run([str(python),"validate.py"],cwd=root/"source/benchmark",check=True,
    env=os.environ|dict(OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="8"))
result=subprocess.run([str(python),"-u",str(root/"run_ablation.py")])
(root/"ablation-exit-code").write_text(str(result.returncode)+"\n")
raise SystemExit(result.returncode)
