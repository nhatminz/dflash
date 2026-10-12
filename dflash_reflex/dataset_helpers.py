"""Build the pinned Megatron dataset helper with the policy worker's Python ABI."""

import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
import tempfile


def _check_import(artifact):
    # A separate process also detects a stale/broken native module without
    # importing it into the policy worker or creating a CUDA context.
    code = (
        "import importlib.util, sys; "
        "s = importlib.util.spec_from_file_location('helpers_cpp', sys.argv[1]); "
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "assert callable(m.build_exhaustive_blending_indices)"
    )
    return subprocess.run([sys.executable, "-c", code, str(artifact)],
                          capture_output=True, text=True)


def compile_megatron_helpers(directory=None):
    """Compile the original helpers.cpp, retaining the original Makefile flags.

    Override Makefile variables instead of relying on PATH's python3 and
    python3-config. The driver may have activated a different Python version
    even though Ray launches the policy actor under the correct interpreter.
    """
    if directory is None:
        from megatron.core.datasets import utils

        directory = Path(utils.__file__).parent
    directory = Path(directory).resolve()
    try:
        import pybind11
    except ImportError as error:
        raise RuntimeError(
            f"Dataset helper requires pybind11 in {sys.executable}; "
            "install the locked pybind11==3.0.4 into this mcore environment"
        ) from error
    include_dirs = dict.fromkeys([
        sysconfig.get_path("include"), sysconfig.get_path("platinclude"),
        pybind11.get_include(),
    ])
    if not any((Path(path) / "Python.h").is_file() for path in include_dirs if path):
        raise RuntimeError(f"Python.h missing for policy interpreter {sys.executable}")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError(f"No Python EXT_SUFFIX for {sys.executable}")
    flags = shlex.split(os.environ.get("CPPFLAGS", ""))
    flags += ["-I" + path for path in include_dirs if path]
    artifact = directory / ("helpers_cpp" + suffix)
    sources = [directory / "helpers.cpp", directory / "Makefile"]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    # Independent training jobs share the source tree. Serialize builds and
    # publish atomically so another actor never imports a half-written .so.
    with (directory / ".dflash_helpers_build.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if artifact.is_file() and artifact.stat().st_mtime_ns >= max(
            source.stat().st_mtime_ns for source in sources
        ) and _check_import(artifact).returncode == 0:
            print(f"Megatron dataset helper ready: {artifact.name} ({sys.executable})", flush=True)
            return artifact
        with tempfile.TemporaryDirectory(prefix=".dflash-helpers-", dir=directory) as temp:
            candidate = Path(temp) / artifact.name
            # OUT is relative and contains no spaces, including when the
            # project's absolute directory does. Makefile uses an unquoted $@.
            relative_output = str(candidate.relative_to(directory))
            command = ["make", "-C", str(directory), "CPPFLAGS=" + shlex.join(flags),
                       "LIBEXT=" + suffix, "OUT=" + relative_output]
            print(f"Building Megatron dataset helper with {sys.executable}: {artifact.name}", flush=True)
            result = subprocess.run(command)
            if result.returncode:
                raise RuntimeError(
                    f"Megatron dataset helper build failed (exit {result.returncode}) "
                    f"using {sys.executable}; see compiler diagnostic above"
                )
            probe = _check_import(candidate)
            if probe.returncode:
                raise RuntimeError(f"Dataset helper import failed: {probe.stderr}")
            os.replace(candidate, artifact)
    return artifact


def configure_megatron_dataset_builder():
    """Replace Bridge's imported build callback, leaving dataset algorithms intact."""
    from megatron.bridge.training import initialize

    original = initialize.compile_helpers
    if original is compile_megatron_helpers:
        return
    if (original.__module__, original.__name__) != (
        "megatron.core.datasets.utils", "compile_helpers"
    ):
        raise RuntimeError("Upstream Bridge dataset compile callback changed")
    initialize.compile_helpers = compile_megatron_helpers
