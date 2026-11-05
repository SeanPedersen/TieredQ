"""Build and test macOS and Hartmut manylinux wheels, with optional local publishing."""

import argparse
import os
import platform
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
CIBUILDWHEEL_VERSION = "4.3.5"
BUILD_PYTHON = "3.12"


def run(*command: str, **kwargs) -> subprocess.CompletedProcess:
    print(f"+ {shlex.join(map(str, command))}", flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def remote(host: str, *command: str, **kwargs) -> subprocess.CompletedProcess:
    return run("ssh", "-o", "BatchMode=yes", host, shlex.join(command), **kwargs)


def python_versions(config: dict) -> list[str]:
    identifiers = config["tool"]["cibuildwheel"]["build"].split()
    return [f"{item[2]}.{item[3 : item.index('-')]}" for item in identifiers]


def linux_wheels(host: str, source: Path, output: Path, versions: list[str]) -> None:
    workspace = remote(
        host,
        "mktemp",
        "-d",
        "/tmp/tieredq-release.XXXXXXXX",
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"Linux build workspace: {host}:{workspace}", flush=True)
    run("scp", str(source), f"{host}:{workspace}/")
    builds = " ".join(f"cp{version.replace('.', '')}-*" for version in versions)
    remote(
        host,
        "env",
        f"CIBW_BUILD={builds}",
        "uvx",
        "--python",
        BUILD_PYTHON,
        "--from",
        f"cibuildwheel=={CIBUILDWHEEL_VERSION}",
        "cibuildwheel",
        f"{workspace}/{source.name}",
        "--platform",
        "linux",
        "--archs",
        "x86_64",
        "--output-dir",
        f"{workspace}/wheels",
    )
    run("scp", f"{host}:{workspace}/wheels/*.whl", str(output))


def macos_wheels(
    source: Path, output: Path, workspace: Path, versions: list[str]
) -> None:
    build_env = {
        **os.environ,
        "MACOSX_DEPLOYMENT_TARGET": os.environ.get("MACOSX_DEPLOYMENT_TARGET", "14.0"),
        "CMAKE_BUILD_PARALLEL_LEVEL": "3",
    }
    test_env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}
    tests = workspace / "source"
    tests.mkdir()
    with tarfile.open(source) as archive:
        archive.extractall(tests, filter="data")
    test_suite = next(tests.iterdir()) / "tests" / "python"
    for version in versions:
        raw = workspace / f"raw-{version}"
        run(
            "uv",
            "build",
            str(source),
            "--wheel",
            "--python",
            version,
            "--out-dir",
            str(raw),
            "-Ccmake.define.BLA_VENDOR=Apple",
            "-Ccmake.define.TIEREDQ_URING=OFF",
            env=build_env,
        )
        wheel = next(raw.glob("*.whl"))
        run(
            "uvx",
            "--python",
            BUILD_PYTHON,
            "--from",
            "delocate",
            "delocate-wheel",
            "--require-archs",
            platform.machine(),
            "-w",
            str(output),
            str(wheel),
        )
        repaired = list(output.glob(f"*-cp{version.replace('.', '')}-*.whl"))
        repaired = [path for path in repaired if "macosx" in path.name]
        if len(repaired) != 1:
            raise RuntimeError(
                f"Expected one repaired macOS wheel for Python {version}"
            )
        venv = workspace / f"test-{version}"
        run("uv", "venv", "--python", version, str(venv))
        python = str(venv / "bin" / "python")
        run("uv", "pip", "install", "--python", python, str(repaired[0]), "pytest>=8")
        run(
            python,
            "-I",
            "-m",
            "pytest",
            str(test_suite),
            "-q",
            cwd=workspace,
            env=test_env,
        )


def validate_artifacts(output: Path, versions: list[str]) -> list[Path]:
    artifacts = sorted([*output.glob("*.whl"), *output.glob("*.tar.gz")])
    for version in versions:
        python_tag = f"cp{version.replace('.', '')}"
        for platform_tag in ("manylinux", "macosx"):
            matches = list(
                output.glob(f"*-{python_tag}-{python_tag}-{platform_tag}*.whl")
            )
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one {platform_tag} wheel for Python {version}"
                )
    if len(artifacts) != 1 + 2 * len(versions):
        raise RuntimeError(
            "Release does not contain the expected source archive and wheels"
        )
    run(
        "uvx",
        "--python",
        BUILD_PYTHON,
        "--from",
        "twine",
        "twine",
        "check",
        "--strict",
        *map(str, artifacts),
    )
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linux-host", default="sean@hartmut")
    parser.add_argument(
        "--publish", action="store_true", help="Upload validated artifacts to PyPI"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Rehearse upload without publishing (requires --publish)",
    )
    parser.add_argument(
        "--python",
        action="append",
        dest="versions",
        help="Restrict Python versions (repeatable)",
    )
    args = parser.parse_args()
    if args.dry_run and not args.publish:
        parser.error("--dry-run requires --publish")
    if platform.system() != "Darwin":
        parser.error(
            "Run on the release Mac; Linux builds run over SSH on --linux-host"
        )
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    supported = python_versions(config)
    versions = args.versions or supported
    if not versions or any(version not in supported for version in versions):
        parser.error(f"Supported Python versions: {', '.join(supported)}")
    versions = list(dict.fromkeys(versions))
    version = config["project"]["version"]
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=f"release-{version}-", dir=dist))
    output = workspace / "artifacts"
    output.mkdir()
    print(f"Release workspace: {workspace}", flush=True)
    run(
        "uv",
        "build",
        "--sdist",
        "--python",
        BUILD_PYTHON,
        "--out-dir",
        str(output),
        cwd=ROOT,
    )
    source = next(output.glob("*.tar.gz"))
    linux_wheels(args.linux_host, source, output, versions)
    macos_wheels(source, output, workspace, versions)
    artifacts = validate_artifacts(output, versions)
    print("Validated artifacts:", output, flush=True)
    publish_command = ["uv", "publish", *map(str, artifacts)]
    if args.publish:
        if args.dry_run:
            publish_command.append("--dry-run")
        run(*publish_command)
        return
    print(
        "Publish explicitly with:",
        shlex.join(publish_command),
        flush=True,
    )


if __name__ == "__main__":
    main()
