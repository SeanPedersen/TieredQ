# Python releases

Releases are built from the Mac and Hartmut, then uploaded locally using the
maintainer's PyPI credentials. There is no GitHub release workflow.

## Build and validate

The Mac needs uv, a C++20 compiler, CMake and Homebrew `libomp`. Hartmut needs
uv and a running Docker daemon accessible to the SSH user. The default SSH
host is `sean@hartmut`; override it with `--linux-host user@host`.

From the repository root:

```sh
uv run --no-project --python 3.12 scripts/release.py
```

The command creates a fresh `dist/release-<version>-<random>/` directory and:

1. Builds a source archive excluding benchmarks and local build artifacts.
2. Copies that archive to a unique `/tmp/tieredq-release.*` directory on Hartmut.
3. Uses cibuildwheel in manylinux Docker containers there to build Linux x86-64
   wheels, bundle dependencies with auditwheel, and test each installed wheel.
4. Copies the Linux wheels back, builds native macOS wheels from the same source
   archive, bundles dependencies with delocate, and tests each installed wheel
   in an isolated environment outside the repository.
5. Checks every distribution with `twine check --strict` and prints the exact
   local publish command. It does not upload anything.

The Python versions come from `tool.cibuildwheel.build` in `pyproject.toml`
(currently CPython 3.10–3.14). For a shorter validation run:

```sh
uv run --no-project --python 3.12 scripts/release.py --python 3.12
```

The final `artifacts/` directory holds only the source archive and repaired,
validated wheels. Intermediate wheels, test environments, and the remote
workspace are retained for inspection. Build and test output streams to the
terminal. A failed command stops the release; it never proceeds to upload.

Linux wheels require glibc 2.28 or newer. macOS wheels target the build Mac's
architecture (Apple Silicon on the current release machine). The build requests
macOS 14 as its baseline; delocate raises the wheel's minimum macOS version when
bundled libraries require a newer version. Inspect the final wheel tags before
publishing. A lower deployment target requires a compatible build of `libomp`;
setting `MACOSX_DEPLOYMENT_TARGET` explicitly also asks delocate to enforce it.
The current release Mac's Homebrew `libomp` produces macOS 26+ wheels.
Intel macOS, Linux ARM, Windows, musl Linux and free-threaded Python are outside
this initial release set. Other supported environments can build from source
with the native dependencies described in the README.

Optional Linux io_uring support is disabled in release wheels for a consistent
dependency set; source builds retain automatic detection. Bundled-library
license texts live in `licenses/`; review them alongside changes to native
dependencies or repair output.

## Publish locally

To build, validate and publish in one command from the repository root:

```sh
./publish-pypi.sh
```

Use `./publish-pypi.sh --dry-run` to build and validate everything and rehearse
the upload without publishing. The wrapper accepts the same `--linux-host` and
`--python` options as the Python build command. It stops before uploading if any
build, test or metadata check fails. Running the Python command directly still
only builds and validates unless `--publish` is supplied.

1. Set the same version in `pyproject.toml` and `CMakeLists.txt`, and refresh
   `uv.lock` with `uv lock` if the version changes.
2. Run the build command and inspect the final artifacts and their wheel tags.
3. Run the exact `uv publish ...` command printed after successful validation.
   uv uses the maintainer's local authentication configuration; for a token it
   supports `UV_PUBLISH_TOKEN`. Keep credentials out of source files and command
   arguments. For a rehearsal, add `--dry-run` to the printed command.
4. Confirm the upload and install the published version in a fresh environment:

   ```sh
   uv venv --python 3.12 /tmp/tieredq-release-check
   uv pip install --python /tmp/tieredq-release-check/bin/python --only-binary :all: tieredq==0.1.0
   /tmp/tieredq-release-check/bin/python -c 'from tieredq import TieredQIndex; print(TieredQIndex)'
   ```

PyPI versions cannot be overwritten. If an upload partially succeeds, inspect
its files before retrying. Use a new version for changed distributions.
