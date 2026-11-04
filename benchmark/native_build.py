"""Build and audit the native adapters identically for validation and measurement.

Builds hold an exclusive lock on the deps directory, so parallel lanes calling build_native do not race on patching
the RaBitQ overlay or on the shared CMake build directory.
"""
from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from build_policy import audit_compile_commands
from setup_native import prepare

HERE = Path(__file__).resolve().parent
BUILD_THREADS = 8


@dataclass(frozen=True)
class NativeBuild:
    directory: Path
    overlay: Path
    compiler_commands: list[dict]


def build_native(full_score=False):
    deps = HERE/".deps"
    deps.mkdir(exist_ok=True)
    directory = HERE/("build-full-score" if full_score else "build-native")
    with (deps/"build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        overlay = prepare(deps, full_score=full_score)
        subprocess.run(["cmake", "-S", str(HERE/"native"), "-B", str(directory),
                        f"-DRABITQ_SOURCE={overlay}", f"-DBENCH_FULL_SCORE={'ON' if full_score else 'OFF'}",
                        f"-DPython_EXECUTABLE={sys.executable}", "-DCMAKE_BUILD_TYPE=Release"], check=True)
        subprocess.run(["cmake", "--build", str(directory), "-j", str(BUILD_THREADS)], check=True)
    commands = audit_compile_commands(json.loads((directory/"compile_commands.json").read_text()))
    return NativeBuild(directory,overlay,commands)
