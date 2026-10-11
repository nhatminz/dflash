"""CUDA header visibility and complete diagnostics across source-build failures."""

import importlib
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("local_build")


def toolkit_fixture(tmp_path, *, version="13.0", cccl=True):
    toolkit = tmp_path / "cuda"
    (toolkit / "bin").mkdir(parents=True)
    nvcc = toolkit / "bin/nvcc"
    nvcc.write_text(f"#!/bin/sh\necho 'Cuda compilation tools, release {version}, V13.0.88'\n")
    nvcc.chmod(0o755)
    if cccl:
        includes = toolkit / "include/cccl"
        (includes / "cuda/std").mkdir(parents=True)
        (includes / "nv").mkdir()
        (includes / "cuda/std/type_traits").write_text("#include <nv/target>\n")
        (includes / "nv/target").write_text("#define CUDA_HEADER_FIXTURE 13\n")
    return toolkit


def test_source_pip_uses_configured_index_without_overriding_explicit_pip(module):
    env = {
        "UV_DEFAULT_INDEX": "https://internal.example/simple",
        "UV_EXTRA_INDEX_URL": "https://extra.example/simple",
    }
    result = module.pip_build_environment(env)
    assert result["PIP_INDEX_URL"] == env["UV_DEFAULT_INDEX"]
    assert result["PIP_EXTRA_INDEX_URL"] == env["UV_EXTRA_INDEX_URL"]
    assert "PIP_INDEX_URL" not in env
    env["PIP_INDEX_URL"] = "https://explicit.example/simple"
    assert module.pip_build_environment(env)["PIP_INDEX_URL"] == env["PIP_INDEX_URL"]


def test_cccl_host_compile_regression_and_build_limits(tmp_path, module):
    compiler = shutil.which("g++") or shutil.which("c++")
    assert compiler, "A C++ compiler is required for the header-path regression"
    toolkit = toolkit_fixture(tmp_path)
    env = dict(os.environ, CUDA_HOME=str(toolkit))
    env.pop("CPLUS_INCLUDE_PATH", None)
    for key in ("MAX_JOBS", "NVCC_THREADS", "NVTE_BUILD_MAX_JOBS"):
        env.pop(key, None)
    command = [compiler, "-x", "c++", "-std=c++20", "-fsyntax-only", "-"]
    source = "#include <cuda/std/type_traits>\nstatic_assert(CUDA_HEADER_FIXTURE == 13);\n"
    failed = subprocess.run(command, input=source, text=True, capture_output=True, env=env)
    assert failed.returncode and "cuda/std/type_traits" in failed.stderr
    result = module.cuda_build_environment(env, tmp_path / "site")
    subprocess.run(command, input=source, text=True, check=True, env=result)
    assert "CPLUS_INCLUDE_PATH" not in env
    assert result["MAX_JOBS"] == "2" and result["NVCC_THREADS"] == "1"
    env.update(MAX_JOBS="7", NVCC_THREADS="3", CPLUS_INCLUDE_PATH="/custom/include")
    custom = module.cuda_build_environment(env, tmp_path / "site")
    assert custom["MAX_JOBS"] == "7" and custom["NVCC_THREADS"] == "3"
    assert custom["CPLUS_INCLUDE_PATH"].endswith(os.pathsep + "/custom/include")


def test_cccl_locked_pip_layout_fallback(tmp_path, module):
    toolkit = toolkit_fixture(tmp_path, cccl=False)
    site = tmp_path / "site"
    cccl = site / "nvidia/cu13/include/cccl"
    (cccl / "cuda/std").mkdir(parents=True)
    (cccl / "nv").mkdir()
    (cccl / "cuda/std/type_traits").touch()
    (cccl / "nv/target").touch()
    env = dict(os.environ, CUDA_HOME=str(toolkit))
    assert str(cccl) in module.cuda_build_environment(env, site)["CPLUS_INCLUDE_PATH"].split(
        os.pathsep
    )


def test_torch_math_headers_from_backend_wheels_reach_host_and_nvcc(tmp_path, module):
    compiler = shutil.which("g++") or shutil.which("c++")
    assert compiler
    toolkit = toolkit_fixture(tmp_path)
    site = tmp_path / "mcore-site"
    include = site / "nvidia/cu13/include"
    include.mkdir(parents=True)
    (include / "cusparse.h").write_text("#include <cublas_v2.h>\n#include <cusolverDn.h>\n")
    (include / "cublas_v2.h").write_text("#define CUBLAS_HEADER_FIXTURE 13\n")
    (include / "cusolverDn.h").write_text("#define CUSOLVER_HEADER_FIXTURE 13\n")
    cudnn = site / "nvidia/cudnn/include"
    cudnn.mkdir(parents=True)
    (cudnn / "cudnn.h").write_text("#define CUDNN_HEADER_FIXTURE 9\n")
    env = dict(
        os.environ, CUDA_HOME=str(toolkit), NVCC_PREPEND_FLAGS="-Xcompiler -Wall -Xcompiler -O2"
    )
    env.pop("CPLUS_INCLUDE_PATH", None)
    env.pop("CPATH", None)
    source = (
        "#include <cusparse.h>\n#include <cudnn.h>\n"
        "static_assert(CUBLAS_HEADER_FIXTURE == 13 && CUSOLVER_HEADER_FIXTURE == 13 "
        "&& CUDNN_HEADER_FIXTURE == 9);\n"
    )
    command = [compiler, "-x", "c++", "-std=c++20", "-fsyntax-only", "-"]
    failed = subprocess.run(command, input=source, env=env, text=True, capture_output=True)
    assert failed.returncode and "cusparse.h" in failed.stderr
    result = module.cuda_build_environment(env, site)
    subprocess.run(command, input=source, env=result, text=True, check=True)
    flags = shlex.split(result["NVCC_PREPEND_FLAGS"])
    assert flags[0] == "-I" + str(toolkit / "include")
    assert "-I" + str(include) in flags and "-I" + str(cudnn) in flags
    assert flags[-4:] == ["-Xcompiler", "-Wall", "-Xcompiler", "-O2"]
    assert "CPATH" not in env and "CPLUS_INCLUDE_PATH" not in env
    # A repeated call retains argument pairs and does not duplicate generated flags.
    assert (
        module.cuda_build_environment(result, site)["NVCC_PREPEND_FLAGS"]
        == result["NVCC_PREPEND_FLAGS"]
    )
    other = tmp_path / "vllm-site"
    isolated = module.cuda_build_environment(env, other)
    assert str(site) not in isolated["CPATH"]


@pytest.mark.parametrize("version, cccl, error", [("12.8", True, "13.0"), ("13.0", False, "CCCL")])
def test_toolkit_incompatible_or_incomplete_fails_before_build(
    tmp_path, module, version, cccl, error
):
    toolkit = toolkit_fixture(tmp_path, version=version, cccl=cccl)
    with pytest.raises(RuntimeError, match=error):
        module.cuda_build_environment(dict(os.environ, CUDA_HOME=str(toolkit)), tmp_path / "site")


def test_complete_build_log_keeps_first_error_before_ptxas_tail(tmp_path, module, capsys):
    log = tmp_path / "failed.log"
    # Include a long preprocessor include stack after FAILED, and a much longer
    # successful NVCC tail: neither may hide the actual host-compiler error.
    script = (
        "import sys\n"
        "print('FAILED: fast_hadamard_transform.o', flush=True)\n"
        "print('g++ -I/cuda/include -c fast_hadamard_transform.cpp', flush=True)\n"
        "print('Included from header.h\\n' * 40, flush=True)\n"
        "print('fatal error: nv/target: No such file or directory', file=sys.stderr, flush=True)\n"
        "print('ptxas info: successful kernel\\n' * 3000, flush=True)\n"
        "sys.exit(7)\n"
    )
    source = tmp_path / "failed_build.py"
    source.write_text(script)
    with pytest.raises(RuntimeError, match="nv/target") as error:
        module.run_logged([sys.executable, str(source)], log, env=os.environ.copy(), cwd=tmp_path)
    assert "Build exited 7" in str(error.value)
    assert "g++ -I/cuda/include" in str(error.value)
    assert log.read_text().count("ptxas info: successful kernel") == 3000
    assert "fatal error: nv/target" in capsys.readouterr().out


def test_logged_pip_build_reports_real_cpp_compiler_failure(tmp_path, module):
    source = tmp_path / "package"
    source.mkdir()
    (source / "setup.py").write_text(
        "from setuptools import setup, Extension\n"
        "setup(name='dflash-build-log-probe', version='0.0.0', "
        "ext_modules=[Extension('probe', ['probe.cpp'], language='c++')])\n"
    )
    (source / "probe.cpp").write_text('#include "dflash_expected_missing_header.h"\n')
    log = tmp_path / "pip-build.log"
    with pytest.raises(RuntimeError, match="dflash_expected_missing_header.h"):
        module.run_logged(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--verbose",
                "--no-deps",
                "--no-index",
                "--no-build-isolation",
                "--wheel-dir",
                str(tmp_path / "wheels"),
                str(source),
            ],
            log,
            env=os.environ.copy(),
            cwd=tmp_path,
        )
    assert "fatal error:" in log.read_text()


def test_toolchain_probe_stops_before_nvcc_if_host_headers_fail(tmp_path, module, monkeypatch):
    # The probe must use backend Torch/Python includes and the selected CXX,
    # retain its error, and never run the long CUDA build after this failure.
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: '["/backend/include"]')
    calls = []

    def fail(command, path, **kwargs):
        calls.append(command)
        assert "-I/backend/include" in command
        raise RuntimeError("host compiler failed")

    monkeypatch.setattr(module, "run_logged", fail)
    with pytest.raises(RuntimeError, match="host compiler failed"):
        module.check_cuda_toolchain(
            sys.executable,
            {"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++", "CUDA_HOME": "/cuda"},
            tmp_path,
        )
    assert len(calls) == 1 and calls[0][0] == "/usr/bin/g++"


def deep_ep_fixture(tmp_path):
    site = tmp_path / "site"
    include = site / "nvidia/cu13/include"
    include.mkdir(parents=True)
    (include / "cuda_profiler_api.h").write_text("extern int cudaProfilerStart();\n")
    (include / "nvtx3").mkdir()
    (include / "nvtx3/nvToolsExt.h").write_text("extern int nvtxRangePushA(const char*);\n")
    library = site / "nvidia/cu13/lib/libnvtx3interop.so.1"
    library.parent.mkdir()
    subprocess.run(
        [
            shutil.which("gcc"),
            "-shared",
            "-fPIC",
            "-x",
            "c",
            "-",
            "-Wl,-soname,libnvtx3interop.so.1",
            "-o",
            str(library),
        ],
        input="int nvtxRangePushA(const char* label) { return 13; }\n",
        text=True,
        check=True,
    )
    return site, include, library


def test_deep_ep_nvtx_host_link_load_and_relocation(tmp_path, module):
    site, _, library = deep_ep_fixture(tmp_path)
    toolkit = toolkit_fixture(tmp_path)
    env = dict(os.environ, CUDA_HOME=str(toolkit), LDFLAGS="-Wl,--no-undefined")
    env.pop("CPATH", None)
    env.pop("CPLUS_INCLUDE_PATH", None)
    env.pop("LIBRARY_PATH", None)
    fixed = module.deep_ep_build_environment(env, site)
    assert module.deep_ep_build_environment(fixed, site) == fixed
    alias = library.with_name("libnvtx3interop.so")
    assert alias.is_symlink() and os.readlink(alias) == library.name
    assert "LIBRARY_PATH" not in env
    output = site / "probe.so"
    # Distutils applies LDFLAGS to the host link. Exercise actual -l resolution,
    # a required external symbol, fresh-process loading, and the relative RUNPATH.
    command = [shutil.which("gcc"), "-shared", "-fPIC", "-x", "c", "-"]
    command += shlex.split(fixed["LDFLAGS"]) + ["-lnvtx3interop", "-o", str(output)]
    subprocess.run(
        command,
        input="#include <nvtx3/nvToolsExt.h>\n"
        'int probe(void) { return nvtxRangePushA("probe"); }\n',
        env=fixed,
        text=True,
        check=True,
    )
    moved = tmp_path / "relocated-site"
    site.rename(moved)
    load_env = os.environ.copy()
    load_env.pop("LD_LIBRARY_PATH", None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import ctypes,sys; assert ctypes.CDLL(sys.argv[1]).probe() == 13",
            str(moved / "probe.so"),
        ],
        env=load_env,
        check=True,
    )


def test_deep_ep_refuses_conflicting_linker_alias(tmp_path, module):
    site, _, library = deep_ep_fixture(tmp_path)
    toolkit = toolkit_fixture(tmp_path)
    library.with_name("libnvtx3interop.so").write_text("unrelated library")
    with pytest.raises(RuntimeError, match="Conflicting NVTX"):
        module.deep_ep_build_environment(dict(os.environ, CUDA_HOME=str(toolkit)), site)


@pytest.mark.parametrize("missing", ["cuda_profiler_api.h", "nvtx3/nvToolsExt.h", "library"])
def test_deep_ep_missing_development_artifacts_fail_early(tmp_path, module, missing):
    site, include, library = deep_ep_fixture(tmp_path)
    toolkit = toolkit_fixture(tmp_path)
    (library if missing == "library" else include / missing).unlink()
    with pytest.raises(RuntimeError, match="Missing locked|development header missing"):
        module.deep_ep_build_environment(dict(os.environ, CUDA_HOME=str(toolkit)), site)


def test_deep_ep_probe_compiler_failure_prevents_load(tmp_path, module, monkeypatch):
    calls = []

    def fail(command, path, **kwargs):
        calls.append(command)
        source = next(Path(p) for p in command if str(p).endswith("probe.cu"))
        assert "cuda_profiler_api.h" in source.read_text()
        assert "-lcuda" in command and "-lnvtx3interop" in command
        raise RuntimeError("profiler compile failed")

    monkeypatch.setattr(module, "run_logged", fail)
    with pytest.raises(RuntimeError, match="profiler compile failed"):
        module.check_deep_ep_toolchain(
            sys.executable, {"CUDA_HOME": "/cuda", "CC": "/usr/bin/gcc"}, tmp_path
        )
    assert len(calls) == 1
