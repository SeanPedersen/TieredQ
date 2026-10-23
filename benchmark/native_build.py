"""Build and audit the native adapters identically for validation and measurement."""
from dataclasses import dataclass
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
    overlay = prepare(deps, full_score=full_score)
    directory = HERE/("build-full-score" if full_score else "build-native")
    subprocess.run(["cmake", "-S", str(HERE/"native"), "-B", str(directory),
                    f"-DRABITQ_SOURCE={overlay}", f"-DBENCH_FULL_SCORE={'ON' if full_score else 'OFF'}",
                    f"-DPython_EXECUTABLE={sys.executable}", "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(directory), "-j", str(BUILD_THREADS)], check=True)
    commands = audit_compile_commands(json.loads((directory/"compile_commands.json").read_text()))
    return NativeBuild(directory,overlay,commands)
