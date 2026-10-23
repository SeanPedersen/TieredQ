"""Reproduce compiler-policy, official-kernel parity and storage tests without datasets."""
import os
import subprocess
import sys
from native_build import build_native


def main():
    build = build_native()
    env = os.environ | {"PYTHONPATH":str(build.directory), "OPENBLAS_NUM_THREADS":"1", "BENCH_SEED":"0"}
    subprocess.run([sys.executable,"-m","unittest","test_native","test_reference","test_benchmark",
                    "test_compare","test_build_policy"],env=env,check=True)
    print("Native compiler options, direct-kernel rankings and flat storage accounting verified.",flush=True)

if __name__ == "__main__":
    main()
