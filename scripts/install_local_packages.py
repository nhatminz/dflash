"""Install staged external artifacts; ordinary locked PyPI packages use pip index.

This is the hybrid (PyPI reachable, GitHub blocked) path, not an offline solver.
It preserves the upstream dependency selections via frozen per-backend exports.
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from local_build import (
    check_cuda_toolchain,
    check_deep_ep_toolchain,
    check_deep_gemm_link,
    check_transformer_engine_headers,
    cuda_build_environment,
    deep_ep_build_environment,
    deep_gemm_build_environment,
    pip_build_environment,
    run_logged,
)

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(plan):
    from fetch_sources import source_head, verify_submodules

    if not plan.get("ready"):
        raise RuntimeError(
            "Local artifacts are incomplete. Finish prepare_local_packages.py on the connected host first."
        )
    if sha256(ROOT / "third_party/nemo-rl/uv.lock") != plan["lock_sha256"]:
        raise RuntimeError("Staged artifacts do not match this NeMo uv.lock")
    for backend, items in plan["requirements"].items():
        for kind, lines in items.items():
            path = ROOT / "offline_packages" / f"{backend}.{kind}.txt"
            if not path.is_file() or path.read_text().splitlines() != lines:
                raise RuntimeError(f"Missing/modified staged requirements: {path}")
    for record in plan["wheels"] + [dict(plan["python"], hash=plan["python"]["sha256"])]:
        path = ROOT / record["path"]
        if not path.is_file() or sha256(path) != record["hash"].removeprefix("sha256:"):
            raise RuntimeError(f"Missing/corrupt staged artifact: {path}")
    for source in plan["sources"].values():
        path = ROOT / source["path"]
        if source_head(path) != source["commit"]:
            raise RuntimeError(f"Missing/wrong pinned local Git source: {path}")
        verify_submodules(path)


def source_order(line):
    names = [
        "nemo-lens",
        "transferqueue",
        "emerging-optimizers",
        "nvidia-resiliency-ext",
        "nvidia-modelopt",
        "torch-memory-saver",
        "fast-hadamard-transform",
        "causal-conv1d",
        "mamba-ssm",
        "deep-ep",
        "deep-gemm",
        "flash-mla",
        "transformer-engine",
        "Megatron-LM",
        "Megatron-Bridge",
    ]
    name = line.rstrip("/").rsplit("/", 1)[-1]
    return names.index(name) if name in names else len(names)


def installation_schedule(plan, *, backend="mcore", retry_source=None, resume_from_source=None):
    """Select a source boundary before mutating any environment.

    Resume trusts the user's explicit completed boundary: earlier backends
    and sources are not installed again. Later backends still receive their
    full registry/source setup, followed by the normal runtime qualification.
    """
    if retry_source and resume_from_source:
        raise RuntimeError("Choose either --retry-source or --resume-from-source")
    order = ("base", "mcore", "vllm")
    requested = retry_source or resume_from_source
    selected = None
    if requested:
        lines = sorted(plan["requirements"][backend]["sources"], key=source_order)
        selected = next(
            (i for i, line in enumerate(lines) if line.rstrip("/").rsplit("/", 1)[-1] == requested),
            None,
        )
        if selected is None:
            raise RuntimeError(f"No staged source {requested!r} in backend {backend}")
    result = []
    for current in order:
        if retry_source and current != backend:
            continue
        if resume_from_source and order.index(current) < order.index(backend):
            continue
        lines = sorted(plan["requirements"][current]["sources"], key=source_order)
        reuse_registry = bool(requested and current == backend)
        if reuse_registry:
            lines = lines[selected : selected + 1] if retry_source else lines[selected:]
        result.append((current, lines, reuse_registry))
    return result


def install_cuda_build_headers(python, env, destination, logs):
    """Install only hash-pinned development headers; bundled wheels need no index."""
    wheels = destination / "toolchain_wheels"
    command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--require-hashes",
        "-r",
        str(ROOT / "requirements-build-cuda13.txt"),
    ]
    if wheels.is_dir():
        command += ["--find-links", str(wheels)]
        if all(
            any(wheels.glob(pattern))
            for pattern in (
                "nvidia_cuda_profiler_api-13.0.85-*.whl",
                "nvidia_nvml_dev-13.0.87-*.whl",
            )
        ):
            command += ["--no-index"]
    run_logged(command, logs / "cuda-build-headers.log", cwd=ROOT, env=pip_build_environment(env))


def bootstrap_lines(lines):
    # Torch's CUDA libraries and the upstream build group must be present
    # before packages whose build metadata imports Torch are evaluated.
    names = {
        "torch",
        "torchvision",
        "triton",
        "numpy",
        "setuptools",
        "pip",
        "packaging",
        "einops",
        "psutil",
        "ninja",
        "pybind11",
        "hatchling",
        "filelock",
        "typing-extensions",
        "sympy",
        "mpmath",
        "networkx",
        "jinja2",
        "markupsafe",
        "fsspec",
    }
    result = []
    for line in lines:
        name = line.split("==", 1)[0].split("[", 1)[0].lower()
        if line.startswith("./"):
            name = Path(line).name.split("-", 1)[0].replace("_", "-")
        if name in names or name.startswith(("nvidia-", "cuda-")):
            result.append(line)
    return result


def source_build_options(line, plan, env, no_isolation):
    """Use upstream's isolation policy and locked versions for shallow Git trees."""
    name = line.rstrip("/").rsplit("/", 1)[-1]
    source_env = env.copy()
    if name == "nvidia-modelopt":
        source_env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_MODELOPT"] = plan["sources"][name][
            "version"
        ]
    elif name == "nvidia-resiliency-ext":
        source_env["POETRY_DYNAMIC_VERSIONING_BYPASS"] = plan["sources"][name]["version"]
    normalized = name.lower().replace("_", "-")
    flags = ["--no-build-isolation"] if normalized in no_isolation else []
    return flags + shlex.split(line), source_env


def compiler_environment(env):
    """Override Python's build-time compiler defaults with installed compilers.

    A standalone Python can name clang/clang++ in sysconfig even on a host
    that only provides GCC. Explicit, executable CC/CXX commands take priority.
    """
    result = env.copy()
    search_path = env.get("PATH", os.defpath)
    for variable in ("CC", "CXX"):
        if not env.get(variable):
            continue
        try:
            command = shlex.split(env[variable])
        except ValueError as error:
            raise RuntimeError(f"Invalid {variable}: {error}") from None
        if not command or not shutil.which(command[0], path=search_path):
            raise RuntimeError(
                f"{variable}={env[variable]!r} is not executable on this host. "
                "Set CC/CXX to installed C/C++ compilers, or unset both for automatic selection."
            )
    if not all(result.get(key) for key in ("CC", "CXX")):
        for c_name, cxx_name in (("gcc", "g++"), ("clang", "clang++"), ("cc", "c++")):
            c = result.get("CC") or shutil.which(c_name, path=search_path)
            cxx = result.get("CXX") or shutil.which(cxx_name, path=search_path)
            if c and cxx:
                result.update(CC=c, CXX=cxx)
                break
        else:
            raise RuntimeError(
                "No usable C/C++ compiler pair found (gcc/g++, clang/clang++, cc/c++). "
                "Expose an installed development toolchain on PATH or set CC and CXX. "
                "These are system compilers, not Python pip dependencies."
            )
    print(f"Local source compilers: CC={result['CC']}, CXX={result['CXX']}", flush=True)
    return result


def resiliency_cupti_environment(python, env):
    """Expose the target environment's pinned CUPTI to NRX's isolated build.

    NRX searches CUDA_PATH recursively for cupti.h and exactly libcupti.so.
    NVIDIA's CUDA-13 pip wheel ships libcupti.so.13 without that linker alias.
    Keep this override scoped to NRX; other packages still use CUDA_HOME/nvcc.
    """
    result = env.copy()
    if env.get("STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD", "0").lower() in {"1", "on", "yes", "true"}:
        return result
    probe = """import importlib.metadata as m, json
from pathlib import Path
try:
    dist = m.distribution("nvidia-cuda-cupti")
    files = [Path(dist.locate_file(f)).absolute() for f in (dist.files or [])]
    header = next(p for p in files if p.name == "cupti.h" and p.is_file())
    names = ("libcupti.so", "libcupti.so." + dist.version.split(".")[0])
    library = next(p for name in names for p in files if p.name == name and p.is_file())
    print(json.dumps(dict(header=str(header), library=str(library))))
except (m.PackageNotFoundError, StopIteration):
    raise SystemExit("The target environment's nvidia-cuda-cupti lacks its header or shared library")
"""
    process = subprocess.run([str(python), "-c", probe], env=env, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(
            f"Cannot prepare CUPTI for nvidia-resiliency-ext using {python}:\n"
            f"{process.stderr.strip()}\n"
            "Restore nvidia-cuda-cupti from this backend's locked registry requirements, "
            "then rerun setup. activate.sh is created only after package installation succeeds."
        )
    paths = json.loads(process.stdout.strip().splitlines()[-1])
    header, library = Path(paths["header"]), Path(paths["library"])
    alias = library.parent / "libcupti.so"
    if not alias.exists():
        if alias.is_symlink():
            raise RuntimeError(f"Broken existing CUPTI linker alias: {alias}")
        alias.symlink_to(library.name)
    elif alias.resolve() != library.resolve():
        raise RuntimeError(f"Existing CUPTI linker alias differs from the pinned library: {alias}")
    root = Path(os.path.commonpath((header.parent, library.parent)))
    result["CUDA_PATH"] = str(root)
    # CUPTI headers include CUDA driver/runtime headers; pip's runtime wheel
    # does not include every CRT header supplied by the development toolkit.
    nvcc = shutil.which("nvcc", path=env.get("PATH"))
    candidates = [env.get("CUDA_HOME"), env.get("CUDA_PATH")]
    if nvcc:
        candidates.append(str(Path(nvcc).resolve().parent.parent))
    candidates.append("/usr/local/cuda")
    for candidate in candidates:
        if candidate and (Path(candidate) / "include/crt/host_defines.h").is_file():
            include = shlex.quote(f"-I{Path(candidate) / 'include'}")
            result["CPPFLAGS"] = (result.get("CPPFLAGS", "") + " " + include).strip()
            break
    # NRX's Extension omits runtime_library_dirs. Retain the actual CUPTI
    # directory in its ELF RUNPATH so importing it works in a fresh terminal.
    rpath = shlex.quote(f"-Wl,-rpath,{library.parent}")
    result["LDFLAGS"] = (result.get("LDFLAGS", "") + " " + rpath).strip()
    print(f"nvidia-resiliency-ext CUPTI: {root}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-runtime-check", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument(
        "--retry-source", help="Rebuild only this staged source in an existing backend"
    )
    recovery.add_argument(
        "--resume-from-source",
        help="Continue setup at this source in --backend, preserving earlier completed installs",
    )
    parser.add_argument(
        "--backend",
        choices=("base", "mcore", "vllm"),
        default="mcore",
        help="Backend used by --retry-source or --resume-from-source (default: mcore)",
    )
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        raise RuntimeError(
            "Run this staging installer with Python >=3.12 (tar extraction filter required)"
        )
    destination = ROOT / "offline_packages"
    plan = json.loads((destination / "manifest.json").read_text())
    schedule = installation_schedule(
        plan,
        backend=args.backend,
        retry_source=args.retry_source,
        resume_from_source=args.resume_from_source,
    )
    verify(plan)
    env = dict(
        os.environ,
        UV_PYTHON_DOWNLOADS="never",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    # uv does not automatically consume pip's index environment variables.
    if not env.get("UV_INDEX_URL") and not env.get("UV_DEFAULT_INDEX") and env.get("PIP_INDEX_URL"):
        env["UV_INDEX_URL"] = env["PIP_INDEX_URL"]
    if env.get("PIP_EXTRA_INDEX_URL") and not env.get("UV_EXTRA_INDEX_URL"):
        env["UV_EXTRA_INDEX_URL"] = env["PIP_EXTRA_INDEX_URL"]
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/fetch_sources.py"), "--offline"], env=env, check=True
    )
    if args.check_only:
        print("All staged external wheels, Python, sources and recursive submodules verified.")
        return
    # Propagate into every backend and isolated PEP-517 build, before any install.
    env = compiler_environment(env)
    runtime = destination / "runtime"
    python = runtime / "python/bin/python3.13"
    if not python.is_file():
        runtime.mkdir(exist_ok=True)
        with tarfile.open(ROOT / plan["python"]["path"], "r:gz") as archive:
            archive.extractall(runtime, filter="data")
    subprocess.run(
        [str(python), "-c", "import sys; assert (3,13,14) <= sys.version_info[:3] < (3,14,0)"],
        check=True,
    )
    env.setdefault("NVTE_CUDA_ARCHS", "100")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
    # These setup.py files otherwise try GitHub release wheels during a build.
    env.update(
        MAMBA_FORCE_BUILD="TRUE",
        CAUSAL_CONV1D_FORCE_BUILD="TRUE",
        FAST_HADAMARD_TRANSFORM_FORCE_BUILD="TRUE",
        DG_FORCE_BUILD="1",
        TMS_CUDA_MAJOR="13",
        NVTE_FRAMEWORK="pytorch",
    )
    interpreters = {}
    if args.resume_from_source:
        # No automatic inference from version metadata: local Git packages
        # can share a version across commits. The boundary is explicitly supplied.
        for backend in ("base", "mcore", "vllm"):
            path = ROOT / (".venv-local" if backend == "base" else f".venv-local-{backend}")
            executable = path / "bin/python"
            if not executable.is_file():
                raise RuntimeError(f"Resume requires the earlier/selected environment: {path}")
            subprocess.run(
                [
                    str(executable),
                    "-c",
                    "import sys; assert (3,13,14) <= sys.version_info[:3] < (3,14,0)",
                ],
                check=True,
            )
            interpreters[backend] = str(executable)
            if backend == args.backend:
                break
    log_root = destination / "build_logs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    upstream = tomllib.loads((ROOT / "third_party/nemo-rl/pyproject.toml").read_text())
    no_isolation = {
        name.lower().replace("_", "-")
        for name in upstream["tool"]["uv"]["no-build-isolation-package"]
    }
    for backend, source_lines, reuse_registry in schedule:
        items = plan["requirements"][backend]
        path = ROOT / (".venv-local" if backend == "base" else f".venv-local-{backend}")
        executable = path / "bin/python"
        if executable.exists():
            subprocess.run(
                [
                    str(executable),
                    "-c",
                    "import sys; assert (3,13,14) <= sys.version_info[:3] < (3,14,0)",
                ],
                check=True,
            )
        else:
            if reuse_registry:
                raise RuntimeError(f"Retry requires an existing {path}; run full setup first")
            subprocess.run(["uv", "venv", str(path), "--python", str(python)], env=env, check=True)
        build_env = dict(env, VIRTUAL_ENV=str(path), PATH=f"{path}/bin:{env['PATH']}")
        command = ["uv", "pip", "install", "--python", str(executable), "--no-deps"]
        # Build tooling can come from the configured PyPI/internal index.
        if not reuse_registry:
            subprocess.run(
                command + ["wheel>=0.46.2", "cmake>=3.21"], cwd=ROOT, env=build_env, check=True
            )
            bootstrap = destination / f"{backend}.bootstrap.txt"
            bootstrap.write_text("\n".join(bootstrap_lines(items["registry"])) + "\n")
            subprocess.run(command + ["-r", str(bootstrap)], cwd=ROOT, env=build_env, check=True)
            subprocess.run(
                command + ["-r", str(destination / f"{backend}.registry.txt")],
                cwd=ROOT,
                env=build_env,
                check=True,
            )
        site_packages = Path(
            subprocess.check_output(
                [str(executable), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                text=True,
            ).strip()
        )
        cudnn = site_packages / "nvidia/cudnn"
        if cudnn.is_dir():
            build_env.setdefault("CUDNN_PATH", str(cudnn))
            build_env.setdefault("CUDNN_HOME", str(cudnn))
        logs = log_root / backend
        if backend in {"mcore", "vllm"}:
            install_cuda_build_headers(executable, build_env, destination, logs)
            build_env = cuda_build_environment(build_env, site_packages)
            check_cuda_toolchain(executable, build_env, logs)
        for line in source_lines:
            name = line.rstrip("/").rsplit("/", 1)[-1]
            options, source_env = source_build_options(line, plan, build_env, no_isolation)
            if name == "nvidia-resiliency-ext":
                source_env = resiliency_cupti_environment(executable, source_env)
            if name == "deep-ep":
                source_env = deep_ep_build_environment(source_env, site_packages)
                check_deep_ep_toolchain(executable, source_env, logs)
            if name == "deep-gemm":
                source_env = deep_gemm_build_environment(source_env, site_packages)
                check_deep_gemm_link(executable, source_env, logs)
            if name == "transformer-engine":
                check_transformer_engine_headers(source_env, ROOT / shlex.split(line)[-1], logs)
            # pip streams the complete backend output with --verbose. uv's
            # failure summary can truncate the first compiler error behind
            # thousands of successful ptxas messages from another Ninja job.
            source_command = [str(executable), "-m", "pip", "install", "--verbose", "--no-deps"]
            if name == "Megatron-Bridge":
                # The pinned NeMo root requires 3.13 while its Bridge gitlink
                # still declares <3.13. Keep the pin and make this one metadata
                # exception explicit; the final MCore import check remains required.
                print(
                    "Pinned Megatron-Bridge declares Python <3.13; overriding only this metadata bound."
                )
                options = ["--no-build-isolation", "--ignore-requires-python"] + shlex.split(line)
            run_logged(
                source_command + options,
                logs / f"{name}.log",
                cwd=ROOT,
                env=pip_build_environment(source_env),
            )
            if name == "deep-gemm":
                # Link success alone does not prove Torch and the extension
                # load together. Verify the installed package in a fresh actor.
                run_logged(
                    [
                        str(executable),
                        "-c",
                        "import deep_gemm; print('DeepGEMM import:', deep_gemm.__version__)",
                    ],
                    logs / "deep-gemm-import.log",
                    cwd=ROOT,
                    env=source_env,
                )
        if args.retry_source:
            print(f"Source retry succeeded; full setup still required. Logs: {logs}")
            return
        subprocess.run(
            command + ["--no-build-isolation", "-e", "./third_party/nemo-rl"],
            cwd=ROOT,
            env=build_env,
            check=True,
        )
        subprocess.run(
            command + ["-r", str(ROOT / "requirements.txt")], cwd=ROOT, env=build_env, check=True
        )
        interpreters[backend] = str(executable)
    subprocess.run(
        [interpreters["vllm"], str(ROOT / "scripts/patch_upstream.py"), "--vllm-installed"],
        env=env,
        check=True,
    )
    env.update(
        PYTHON_BIN=interpreters["base"],
        DFLASH_MCORE_PYTHON=interpreters["mcore"],
        DFLASH_VLLM_PYTHON=interpreters["vllm"],
        PYTHONPATH=f"{ROOT}:{ROOT}/third_party/nemo-rl"
        + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
    )
    # Compute paths when sourced so this activation file relocates with project.
    activation = """_dflash_local_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN="$_dflash_local_root/.venv-local/bin/python"
export DFLASH_MCORE_PYTHON="$_dflash_local_root/.venv-local-mcore/bin/python"
export DFLASH_VLLM_PYTHON="$_dflash_local_root/.venv-local-vllm/bin/python"
export UV_PYTHON_DOWNLOADS=never
export PYTHONPATH="$_dflash_local_root:$_dflash_local_root/third_party/nemo-rl${PYTHONPATH:+:$PYTHONPATH}"
unset _dflash_local_root
"""
    (destination / "activate.sh").write_text(activation)
    if not args.skip_runtime_check:
        subprocess.run(
            [
                interpreters["base"],
                str(ROOT / "scripts/validate_environment.py"),
                "--require-runtime",
            ],
            env=env,
            check=True,
        )
    print("Use these environments: source offline_packages/activate.sh")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"Local setup failed: {error}", file=sys.stderr)
        raise SystemExit(2) from None
