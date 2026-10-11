"""External artifact selection and offline fetch failure/recovery contracts."""

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_plan_stages_target_wheels_and_every_non_pypi_requirement(monkeypatch):
    module = load_script("prepare_local_packages", monkeypatch)
    plan = module.build_plan(ROOT, ROOT / "offline_packages")
    assert len(plan["wheels"]) == 7
    assert len(plan["sources"]) == 13
    for wheel in plan["wheels"]:
        assert "aarch64" not in wheel["path"] and "x86_64" in wheel["path"]
        assert "cp313" in wheel["path"] or "abi3" in wheel["path"]
        assert wheel["hash"].startswith("sha256:")
    assert {w["package"] for w in plan["wheels"]} == {
        "torch",
        "torchvision",
        "triton",
        "flash-attn",
        "flashinfer-jit-cache",
        "vllm",
    }
    assert {w["package"] for w in plan["wheels"] if "cu130" in w["path"]} >= {
        "torch",
        "torchvision",
    }
    for backend in ("base", "mcore", "vllm"):
        items = plan["requirements"][backend]
        assert all(
            "://" not in line and "git+" not in line for group in items.values() for line in group
        )
        assert all(line.startswith(("./", "-e ./")) for line in items["sources"])
    assert not plan["ready"]


def test_download_rejects_hash_mismatch_before_publishing(tmp_path, monkeypatch):
    module = load_script("prepare_local_packages", monkeypatch)
    source = tmp_path / "source.bin"
    source.write_bytes(b"unexpected contents")
    target = tmp_path / "wheel.whl"
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        module.download(source.as_uri(), target, "sha256:" + "0" * 64)
    assert not target.exists()
    module.download(source.as_uri(), target, module.digest(source))
    assert target.read_bytes() == source.read_bytes()


def test_offline_empty_git_does_not_fetch_or_modify_sources(tmp_path, monkeypatch):
    module = load_script("fetch_sources", monkeypatch)
    repo = tmp_path / "third_party/nemo-rl"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    (tmp_path / "versions.lock").write_text(
        json.dumps({"nemo-rl": {"url": "https://invalid.example/repo", "sha": "a" * 40}})
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["fetch_sources.py", "--offline"])
    calls = []
    run = subprocess.run

    def record(cmd, **kwargs):
        calls.append(cmd)
        return run(cmd, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", record)
    with pytest.raises(RuntimeError, match="empty .git"):
        module.main()
    assert all("fetch" not in command and "checkout" not in command for command in calls)
    assert list(repo.iterdir()) == [repo / ".git"]


def test_online_fetch_recovers_only_empty_repository(tmp_path, monkeypatch):
    module = load_script("fetch_sources", monkeypatch)
    remote = tmp_path / "remote"
    subprocess.run(["git", "init", str(remote)], check=True, capture_output=True)
    (remote / "file").write_text("pinned source")
    subprocess.run(["git", "-C", str(remote), "add", "file"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(remote),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(remote), "rev-parse", "HEAD"], text=True
    ).strip()
    project = tmp_path / "project"
    names = ("nemo-rl", "vllm", "original-dflash")
    project.mkdir()
    (project / "versions.lock").write_text(
        json.dumps({name: {"url": remote.as_uri(), "sha": commit} for name in names})
    )
    (project / "scripts").mkdir()
    (project / "scripts/patch_upstream.py").write_text("# no vendor patch in this fixture\n")
    empty = project / "third_party/nemo-rl"
    empty.mkdir(parents=True)
    subprocess.run(["git", "init", str(empty)], check=True, capture_output=True)
    monkeypatch.setattr(module, "ROOT", project)
    monkeypatch.setattr(sys, "argv", ["fetch_sources.py"])
    monkeypatch.delenv("DFLASH_OFFLINE", raising=False)
    monkeypatch.delenv("UV_OFFLINE", raising=False)
    module.main()
    assert all(module.source_head(project / "third_party" / name) == commit for name in names)
    # A pinned working tree containing local changes is left intact.
    (empty / "file").write_text("keep local patch")
    module.main()
    assert (empty / "file").read_text() == "keep local patch"


def test_incomplete_local_bundle_fails_before_any_install(monkeypatch):
    module = load_script("install_local_packages", monkeypatch)
    with pytest.raises(RuntimeError, match="incomplete"):
        module.verify({"ready": False})


def test_shallow_sources_keep_locked_versions_and_upstream_build_isolation(monkeypatch):
    module = load_script("install_local_packages", monkeypatch)
    plan = {
        "sources": {
            "nvidia-modelopt": {"version": "0.44.0.dev0"},
            "nvidia-resiliency-ext": {"version": "0.6.0"},
        }
    }
    env = {"PATH": "/fixture"}
    options, source_env = module.source_build_options(
        "./offline_packages/sources/nvidia-modelopt", plan, env, {"deep-gemm"}
    )
    assert "--no-build-isolation" not in options
    assert source_env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_MODELOPT"] == "0.44.0.dev0"
    _, source_env = module.source_build_options(
        "./offline_packages/sources/nvidia-resiliency-ext", plan, env, {"deep-gemm"}
    )
    assert source_env["POETRY_DYNAMIC_VERSIONING_BYPASS"] == "0.6.0"
    options, _ = module.source_build_options(
        "./offline_packages/sources/deep-gemm", plan, env, {"deep-gemm"}
    )
    assert options[0] == "--no-build-isolation"
    assert env == {"PATH": "/fixture"}


def test_compiler_selection_on_gcc_only_host(tmp_path, monkeypatch):
    module = load_script("install_local_packages", monkeypatch)
    for name in ("gcc", "g++"):
        executable = tmp_path / name
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    env = {"PATH": str(tmp_path), "CFLAGS": "-O2"}
    result = module.compiler_environment(env)
    assert result["CC"] == str(tmp_path / "gcc")
    assert result["CXX"] == str(tmp_path / "g++")
    assert result["CFLAGS"] == "-O2"
    assert env == {"PATH": str(tmp_path), "CFLAGS": "-O2"}


def test_compiler_selection_preserves_explicit_commands_and_flags(tmp_path, monkeypatch):
    module = load_script("install_local_packages", monkeypatch)
    compiler = tmp_path / "custom compiler"
    compiler.write_text("#!/bin/sh\nexit 0\n")
    compiler.chmod(0o755)
    env = {"PATH": "", "CC": shlex.quote(str(compiler)), "CXX": shlex.quote(str(compiler)) + " -O2"}
    assert module.compiler_environment(env) == env


@pytest.mark.parametrize("explicit", [None, "missing-clang++", "'unterminated"])
def test_compiler_selection_missing_or_invalid_fails_early(tmp_path, monkeypatch, explicit):
    module = load_script("install_local_packages", monkeypatch)
    env = {"PATH": str(tmp_path)}
    if explicit is not None:
        env["CXX"] = explicit
    with pytest.raises(RuntimeError, match="compiler|CXX"):
        module.compiler_environment(env)


def test_cpp_extension_build_overrides_unavailable_python_compiler(tmp_path, monkeypatch):
    module = load_script("install_local_packages", monkeypatch)
    env = dict(os.environ)
    env.pop("CC", None)
    env.pop("CXX", None)
    env = module.compiler_environment(env)
    (tmp_path / "probe.cpp").write_text(
        "#include <Python.h>\n#include <vector>\n"
        "static PyObject* value(PyObject*, PyObject*) { std::vector<int> v{13}; "
        "return PyLong_FromLong(v.at(0)); }\n"
        'static PyMethodDef methods[] = {{"value", value, METH_NOARGS, nullptr}, '
        "{nullptr, nullptr, 0, nullptr}};\n"
        'static PyModuleDef mod = {PyModuleDef_HEAD_INIT, "compiler_probe", '
        "nullptr, -1, methods};\n"
        "PyMODINIT_FUNC PyInit_compiler_probe() { return PyModule_Create(&mod); }\n"
    )
    (tmp_path / "setup.py").write_text(
        "from setuptools import setup, Extension\n"
        "from setuptools._distutils import sysconfig\n"
        "sysconfig.get_config_vars().update(CC='unavailable-clang', "
        "CXX='unavailable-clang++', LDSHARED='unavailable-clang -shared', "
        "LDCXXSHARED='unavailable-clang++ -shared')\n"
        "setup(name='compiler-probe', ext_modules=[Extension('compiler_probe', "
        "['probe.cpp'], language='c++', extra_compile_args=['-std=c++17'])])\n"
    )
    # The broken sysconfig defaults really fail without the installer override.
    broken_env = env.copy()
    broken_env.pop("CC")
    broken_env.pop("CXX")
    failed = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=tmp_path,
        env=broken_env,
        capture_output=True,
        text=True,
    )
    assert failed.returncode and "unavailable-clang" in failed.stderr
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"], cwd=tmp_path, env=env, check=True
    )
    subprocess.run(
        [sys.executable, "-c", "import compiler_probe; assert compiler_probe.value() == 13"],
        cwd=tmp_path,
        env=env,
        check=True,
    )


def cupti_fixture(tmp_path):
    site = tmp_path / "site"
    cuda = site / "nvidia/cu13"
    (cuda / "include").mkdir(parents=True)
    (cuda / "lib").mkdir()
    (cuda / "include/cupti.h").write_text("#include <fixture_extra.h>\nint cupti_fixture(void);\n")
    toolkit = tmp_path / "toolkit"
    (toolkit / "include/crt").mkdir(parents=True)
    (toolkit / "include/crt/host_defines.h").write_text("/* CUDA CRT fixture */\n")
    (toolkit / "include/fixture_extra.h").write_text("/* CUDA development header fixture */\n")
    metadata = site / "nvidia_cuda_cupti-13.0.85.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: nvidia-cuda-cupti\nVersion: 13.0.85\n")
    (metadata / "RECORD").write_text(
        "nvidia/cu13/include/cupti.h,,\nnvidia/cu13/lib/libcupti.so.13,,\n"
    )
    env = dict(os.environ, PYTHONPATH=str(site), CUDA_PATH="/missing/cuda", CUDA_HOME=str(toolkit))
    env.pop("STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD", None)
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("LDFLAGS", None)
    env.pop("CPPFLAGS", None)
    return cuda, env


def test_cupti_pip_layout_builds_and_loads_with_versioned_library(tmp_path, monkeypatch):
    module = load_script("install_local_packages", monkeypatch)
    compiler = shutil.which("cc")
    assert compiler, "A C compiler is required to test the CUPTI link/load regression"
    cuda, env = cupti_fixture(tmp_path)
    subprocess.run(
        [
            compiler,
            "-x",
            "c",
            "-shared",
            "-fPIC",
            "-Wl,-soname,libcupti.so.13",
            "-o",
            str(cuda / "lib/libcupti.so.13"),
            "-",
        ],
        input="int cupti_fixture(void) { return 13; }",
        text=True,
        check=True,
    )
    result = module.resiliency_cupti_environment(sys.executable, env)
    assert result["CUDA_PATH"] == str(cuda)
    assert result["CUDA_HOME"] == str(tmp_path / "toolkit")
    assert env["CUDA_PATH"] == "/missing/cuda"
    alias = cuda / "lib/libcupti.so"
    assert alias.is_symlink() and alias.readlink() == Path("libcupti.so.13")
    # Match upstream's os.walk search (which does not follow directory symlinks).
    assert any("cupti.h" in files for _, _, files in os.walk(result["CUDA_PATH"]))
    assert any("libcupti.so" in files for _, _, files in os.walk(result["CUDA_PATH"]))
    library = tmp_path / "probe.so"
    subprocess.run(
        [
            compiler,
            "-x",
            "c",
            "-shared",
            "-fPIC",
            "-o",
            str(library),
            "-",
            "-I" + str(cuda / "include"),
            "-L" + str(cuda / "lib"),
            "-lcupti",
        ]
        + shlex.split(result["CPPFLAGS"])
        + shlex.split(result["LDFLAGS"]),
        input='#include "cupti.h"\nint probe(void) { return cupti_fixture(); }',
        text=True,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import ctypes,sys; assert ctypes.CDLL(sys.argv[1]).probe() == 13",
            str(library),
        ],
        env=env,
        check=True,
    )
    # A rerun reuses the valid alias and leaves caller flags intact.
    env["LDFLAGS"] = "-Wl,--as-needed"
    assert module.resiliency_cupti_environment(sys.executable, env)["LDFLAGS"].startswith(
        "-Wl,--as-needed "
    )


def test_cupti_missing_files_fail_with_actionable_error(tmp_path, monkeypatch):
    module = load_script("install_local_packages", monkeypatch)
    _, env = cupti_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="Restore nvidia-cuda-cupti"):
        module.resiliency_cupti_environment(sys.executable, env)


def test_cupti_explicit_upstream_skip_is_preserved(monkeypatch):
    module = load_script("install_local_packages", monkeypatch)
    env = {"STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD": "true"}
    assert module.resiliency_cupti_environment("/not/an/interpreter", env) == env


def recovery_plan():
    return {
        "requirements": {
            "base": {"sources": ["./sources/nemo-lens"], "registry": ["base==1"]},
            "mcore": {
                "sources": [
                    "-e ./sources/Megatron-Bridge",
                    "./sources/transformer-engine",
                    "./sources/fast-hadamard-transform",
                    "-e ./sources/Megatron-LM",
                ],
                "registry": ["mcore==1"],
            },
            "vllm": {
                "sources": ["./sources/deep-ep", "./sources/deep-gemm"],
                "registry": ["vllm==1"],
            },
        }
    }


def test_resume_schedule_retains_later_sources_and_backends(monkeypatch):
    module = load_script("install_local_packages", monkeypatch)
    schedule = module.installation_schedule(
        recovery_plan(), resume_from_source="transformer-engine"
    )
    assert [item[0] for item in schedule] == ["mcore", "vllm"]
    assert [Path(line.split()[-1]).name for line in schedule[0][1]] == [
        "transformer-engine",
        "Megatron-LM",
        "Megatron-Bridge",
    ]
    assert schedule[0][2] is True and schedule[1][2] is False
    retry = module.installation_schedule(recovery_plan(), retry_source="transformer-engine")
    assert retry == [("mcore", ["./sources/transformer-engine"], True)]
    full = module.installation_schedule(recovery_plan())
    assert [item[0] for item in full] == ["base", "mcore", "vllm"]
    assert all(not item[2] for item in full)


def test_resume_bad_boundary_fails_before_work(monkeypatch):
    module = load_script("install_local_packages", monkeypatch)
    with pytest.raises(RuntimeError, match="No staged source"):
        module.installation_schedule(
            recovery_plan(), backend="vllm", resume_from_source="transformer-engine"
        )
    with pytest.raises(RuntimeError, match="Choose either"):
        module.installation_schedule(
            recovery_plan(), retry_source="deep-ep", resume_from_source="deep-ep"
        )


@pytest.mark.parametrize("complete_bundle", [False, True])
def test_cuda_header_supplement_uses_hashes_and_local_bundle(
    tmp_path, monkeypatch, complete_bundle
):
    module = load_script("install_local_packages", monkeypatch)
    wheels = tmp_path / "toolchain_wheels"
    wheels.mkdir()
    (wheels / "nvidia_cuda_profiler_api-13.0.85-py3-none-any.whl").touch()
    if complete_bundle:
        (wheels / "nvidia_nvml_dev-13.0.87-py3-none-any.whl").touch()
    calls = []
    monkeypatch.setattr(module, "run_logged", lambda command, *a, **kw: calls.append((command, kw)))
    module.install_cuda_build_headers(
        "/backend/bin/python",
        {"UV_INDEX_URL": "https://internal.example/simple"},
        tmp_path,
        tmp_path / "logs",
    )
    command, kwargs = calls[0]
    assert command[:4] == ["/backend/bin/python", "-m", "pip", "install"]
    assert "--require-hashes" in command and "--no-deps" in command
    assert "--find-links" in command
    assert ("--no-index" in command) == complete_bundle
    assert kwargs["env"]["PIP_INDEX_URL"] == "https://internal.example/simple"


def test_resume_main_does_not_repeat_earlier_installs_and_keeps_runtime_check(
    tmp_path, monkeypatch
):
    module = load_script("install_local_packages", monkeypatch)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    plan = recovery_plan()
    destination = tmp_path / "offline_packages"
    (destination / "runtime/python/bin").mkdir(parents=True)
    (destination / "runtime/python/bin/python3.13").touch()
    (destination / "manifest.json").write_text(json.dumps(plan))
    project = tmp_path / "third_party/nemo-rl"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[tool.uv]\nno-build-isolation-package = []\n")
    for name in (".venv-local", ".venv-local-mcore", ".venv-local-vllm"):
        (tmp_path / name / "bin").mkdir(parents=True)
        (tmp_path / name / "bin/python").touch()
    calls = []
    monkeypatch.setattr(module, "verify", lambda plan: None)
    monkeypatch.setattr(module, "compiler_environment", lambda env: dict(env, CC="gcc", CXX="g++"))
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kw: calls.append(list(command)))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: str(tmp_path / "site"))
    monkeypatch.setattr(module, "run_logged", lambda command, *a, **kw: calls.append(list(command)))
    monkeypatch.setattr(module, "cuda_build_environment", lambda env, site: env)
    monkeypatch.setattr(module, "deep_ep_build_environment", lambda env, site: env)
    monkeypatch.setattr(module, "check_cuda_toolchain", lambda *a: None)
    monkeypatch.setattr(module, "check_deep_ep_toolchain", lambda *a: None)
    monkeypatch.setattr(module, "check_transformer_engine_headers", lambda *a: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "install_local_packages.py",
            "--backend",
            "mcore",
            "--resume-from-source",
            "transformer-engine",
        ],
    )
    module.main()
    source_calls = [c for c in calls if "--verbose" in c]
    assert [Path(c[-1]).name for c in source_calls] == [
        "transformer-engine",
        "Megatron-LM",
        "Megatron-Bridge",
        "deep-ep",
        "deep-gemm",
    ]
    registry = [
        c
        for c in calls
        if c[:3] == ["uv", "pip", "install"]
        and any(p.endswith((".registry.txt", ".bootstrap.txt")) or p == "wheel>=0.46.2" for p in c)
    ]
    assert registry and all(".venv-local-vllm/bin/python" in c[4] for c in registry)
    assert any("--require-runtime" in c for c in calls)
    assert any(str(tmp_path / "scripts/patch_upstream.py") in c for c in calls)
    activation = (destination / "activate.sh").read_text()
    assert "DFLASH_MCORE_PYTHON" in activation and "DFLASH_VLLM_PYTHON" in activation
