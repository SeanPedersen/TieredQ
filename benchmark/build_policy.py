"""Audit effective compiler options for both codecs before accepting measurements."""
from pathlib import Path
import shlex

CODEC_SOURCES = {"module.cpp", "tieredq.cpp", "tieredq_index.cpp", "tieredq_index_io.cpp", "reference.cpp"}


def audit_compile_commands(commands):
    audited = []
    for entry in commands:
        name = Path(entry["file"]).name
        if name not in CODEC_SOURCES:
            continue
        arguments = entry.get("arguments") or shlex.split(entry["command"])
        optimization = [arg for arg in arguments if arg.startswith("-O")]
        if not optimization or optimization[-1] != "-O3":
            raise RuntimeError(f"{name}: effective optimization must be -O3, got {optimization}")
        for required in ("-march=native", "-ffp-contract=off"):
            if required not in arguments:
                raise RuntimeError(f"{name}: missing {required}")
        audited.append({"file": entry["file"], "arguments": arguments, "effective_optimization": optimization[-1]})
    names = {Path(entry["file"]).name for entry in audited}
    if not {"module.cpp", "tieredq.cpp"}.issubset(names):
        raise RuntimeError("Compile database must cover both RaBitQ's header kernels and TieredQ")
    return audited
