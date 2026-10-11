"""CUDA-13 source build environment, early compile probes and complete logs."""

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path


def pip_build_environment(env):
    """Keep the configured package index when pip drives the source backend."""
    result = env.copy()
    index = env.get("UV_DEFAULT_INDEX") or env.get("UV_INDEX_URL")
    if not env.get("PIP_INDEX_URL") and index:
        result["PIP_INDEX_URL"] = index
    if not env.get("PIP_EXTRA_INDEX_URL") and env.get("UV_EXTRA_INDEX_URL"):
        result["PIP_EXTRA_INDEX_URL"] = env["UV_EXTRA_INDEX_URL"]
    return result


def cuda_build_environment(env, site_packages):
    result = env.copy()
    nvcc = shutil.which("nvcc", path=env.get("PATH"))
    home = env.get("CUDA_HOME")
    if not home:
        home = str(Path(nvcc).resolve().parent.parent) if nvcc else "/usr/local/cuda"
    home = Path(home).absolute()
    nvcc = home / "bin/nvcc"
    if not shutil.which(str(nvcc)):
        raise RuntimeError(
            f"CUDA_HOME={home} has no executable bin/nvcc; CUDA 13.0 devel is required"
        )
    version = subprocess.check_output([str(nvcc), "--version"], env=env, text=True)
    if not re.search(r"release 13\.0(?:,|\s)", version):
        raise RuntimeError(
            f"Expected CUDA toolkit 13.0 to match locked Torch; {nvcc} reports:\n{version}"
        )
    result["CUDA_HOME"] = str(home)
    result["PATH"] = str(home / "bin") + os.pathsep + env.get("PATH", os.defpath)
    # Match the pinned NeMo Dockerfile. NVCC knows this directory, but the
    # host C++ compiler does not implicitly search CUDA 13's include/cccl.
    candidates = [home / "include/cccl", Path(site_packages) / "nvidia/cu13/include/cccl"]
    for cccl in candidates:
        if (cccl / "cuda/std/type_traits").is_file() and (cccl / "nv/target").is_file():
            break
    else:
        raise RuntimeError(
            "CUDA 13 CCCL headers cuda/std/type_traits and nv/target are missing. "
            f"Checked: {', '.join(map(str, candidates))}. "
            "Restore the development toolkit's include/cccl or the locked nvidia-cuda-cccl wheel."
        )
    # Torch's CUDAContextLight.h also includes cuSPARSE/cuBLAS/cuSOLVER.
    # CUDA 13 wheels merge these headers under nvidia/cu13/include; cuDNN
    # and older wheel layouts use separate nvidia/<component>/include roots.
    # Expose existing backend-local headers to BOTH the host compiler and
    # NVCC's device frontend. CPLUS_INCLUDE_PATH alone only covers the host.
    components = (
        "cu13",
        "cublas",
        "cusparse",
        "cusolver",
        "cudnn",
        "cusparselt",
        "cuda_runtime",
        "cuda_nvrtc",
        "cuda_cccl",
        "curand",
        "cufft",
        "nvtx",
        "nccl",
    )
    wheel_includes = [Path(site_packages) / "nvidia" / name / "include" for name in components]
    includes = [home / "include", cccl] + [p for p in wheel_includes if p.is_dir()]
    for variable in ("CPATH", "CPLUS_INCLUDE_PATH"):
        previous = [p for p in env.get(variable, "").split(os.pathsep) if p]
        result[variable] = os.pathsep.join(dict.fromkeys([*map(str, includes), *previous]))
    # Keep the development toolkit headers first: pip can contain a newer
    # CRT than the installed NVCC. Never replace CUDA_HOME with a wheel root.
    flags = ["-I" + str(p) for p in includes]
    own_flags = set(flags)
    flags += [f for f in shlex.split(env.get("NVCC_PREPEND_FLAGS", "")) if f not in own_flags]
    result["NVCC_PREPEND_FLAGS"] = shlex.join(flags)
    # These are build resource limits, independent of training batch sizes.
    for key, default in (
        ("MAX_JOBS", "2"),
        ("NVCC_THREADS", "1"),
        ("NVTE_BUILD_MAX_JOBS", result.get("MAX_JOBS", "2")),
        ("NVTE_BUILD_THREADS_PER_JOB", "1"),
        ("CMAKE_BUILD_PARALLEL_LEVEL", result.get("MAX_JOBS", "2")),
    ):
        result.setdefault(key, default)
    print(f"CUDA source build: {nvcc}; CCCL={cccl}; MAX_JOBS={result['MAX_JOBS']}", flush=True)
    print("CUDA header roots: " + os.pathsep.join(map(str, includes)), flush=True)
    return result


def first_build_error(log_path):
    """Extract the first failed command/diagnostic, before a long successful tail."""
    lines = log_path.read_text(errors="replace").splitlines()
    pattern = re.compile(r"fatal error:|\berror:|\bKilled\b|nvcc fatal|CMake Error", re.I)
    for index, line in enumerate(lines):
        if pattern.search(line):
            failed = next((i for i in range(index) if "FAILED:" in lines[i]), None)
            command = lines[failed : failed + 2] if failed is not None else []
            return "\n".join(command + lines[max(0, index - 2) : index + 10])[:6000]
    return "\n".join(lines[-20:])[:6000]


def run_logged(command, log_path, *, env, cwd):
    """Stream pip/compiler output directly, retaining the beginning on failure."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Build log: {log_path}", flush=True)
    with log_path.open("w") as log:
        log.write("Command: " + shlex.join(map(str, command)) + "\n")
        log.flush()
        with subprocess.Popen(
            command,
            env=env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        ) as process:
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            returncode = process.wait()
    if returncode:
        raise RuntimeError(
            f"Build exited {returncode}. Full log: {log_path}\n"
            f"First diagnostic:\n{first_build_error(log_path)}"
        )


def check_cuda_toolchain(python, env, log_dir):
    """Compile host Torch/CUDA headers and a small SM100 object before long builds.

    No GPU kernel is executed. The CUDA probe tests NVCC's host compiler
    compatibility; the C++ probe covers the header path used by Hadamard.
    """
    includes = subprocess.check_output(
        [
            str(python),
            "-c",
            "import json, sysconfig; from torch.utils.cpp_extension import include_paths; "
            "print(json.dumps(include_paths('cuda') + [sysconfig.get_path('include')]))",
        ],
        env=env,
        text=True,
    )
    include_dirs = json.loads(includes.strip().splitlines()[-1])
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="toolchain-", dir=log_dir) as temporary:
        temporary = Path(temporary)
        host = temporary / "headers.cpp"
        host.write_text(
            "#include <cuda_runtime.h>\n#include <cuda/std/type_traits>\n"
            "#include <ATen/cuda/CUDAContext.h>\n#include <torch/extension.h>\n"
        )
        run_logged(
            shlex.split(env["CXX"])
            + ["-std=c++20", "-fsyntax-only", str(host)]
            + ["-I" + p for p in include_dirs],
            log_dir / "toolchain-host.log",
            env=env,
            cwd=temporary,
        )
        device = temporary / "probe.cu"
        device.write_text(
            "#include <cuda_fp16.h>\n#include <cusparse.h>\n"
            "#include <cublas_v2.h>\n#include <cusolverDn.h>\n"
            "__global__ void probe(float* x) { x[threadIdx.x] = 0.0f; }\n"
        )
        # CUDA's -ccbin expects an executable, not shell flags or a launcher.
        # For custom CC commands NVCC keeps its own host compiler selection.
        cc = shlex.split(env["CC"])
        host_flags = ["-ccbin", cc[0]] if len(cc) == 1 else []
        run_logged(
            [str(Path(env["CUDA_HOME"]) / "bin/nvcc"), "-std=c++20", "-arch=sm_100"]
            + host_flags
            + ["-c", str(device), "-o", str(temporary / "probe.o")],
            log_dir / "toolchain-nvcc.log",
            env=env,
            cwd=temporary,
        )
