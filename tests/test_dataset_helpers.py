"""Reproduce wrong-PATH Python builds using the actual pinned C++ source."""

import ast
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from types import ModuleType

import numpy as np
import pytest

from dflash_reflex.dataset_helpers import (
    compile_megatron_helpers,
    configure_megatron_dataset_builder,
)

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / (
    "third_party/nemo-rl/3rdparty/Megatron-Bridge-workspace/"
    "Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets"
)


def test_actual_cpp_build_ignores_wrong_path_and_reuses_cache(tmp_path, monkeypatch):
    directory = tmp_path / "project with spaces" / "datasets"
    directory.mkdir(parents=True)
    for name in ("Makefile", "helpers.cpp"):
        shutil.copyfile(DATASETS / name, directory / name)
    bad_bin = tmp_path / "old-venv-bin"
    bad_bin.mkdir()
    marker = tmp_path / "wrong_python_used"
    for name, output in (("python3", "exit 1"),
                         ("python3-config", "echo .cpython-310-x86_64-linux-gnu.so")):
        script = bad_bin / name
        script.write_text(f"#!/bin/sh\ntouch '{marker}'\n{output}\n")
        script.chmod(0o755)
    monkeypatch.setenv("PATH", str(bad_bin) + os.pathsep + os.environ["PATH"])
    before = subprocess.run(["make", "-C", str(directory)], capture_output=True, text=True)
    assert before.returncode != 0
    assert marker.is_file()
    assert "pybind11/pybind11.h" in before.stderr
    assert "cpython-310" in before.stdout
    marker.unlink()

    artifact = compile_megatron_helpers(directory)
    assert artifact.name == "helpers_cpp" + sysconfig.get_config_var("EXT_SUFFIX")
    assert not marker.exists(), "Makefile must never invoke PATH's Python/config"
    spec = importlib.util.spec_from_file_location("helpers_cpp", artifact)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset_ids = np.empty(3, dtype=np.int16)
    sample_ids = np.empty(3, dtype=np.int64)
    module.build_exhaustive_blending_indices(
        dataset_ids, sample_ids, np.array([2, 1], dtype=np.int64), 2
    )
    assert sorted(zip(dataset_ids.tolist(), sample_ids.tolist())) == [(0, 0), (0, 1), (1, 0)]
    stamp = artifact.stat().st_mtime_ns
    assert compile_megatron_helpers(directory) == artifact
    assert artifact.stat().st_mtime_ns == stamp
    assert not marker.exists()

    # Failed recompilation must leave the previously working artifact intact.
    source = directory / "helpers.cpp"
    source.write_text("#error intentional_test_build_failure\n")
    os.utime(source, ns=(stamp + 1_000_000_000, stamp + 1_000_000_000))
    with pytest.raises(RuntimeError, match="build failed"):
        compile_megatron_helpers(directory)
    assert artifact.stat().st_mtime_ns == stamp
    assert not list(directory.glob(".dflash-helpers-*"))


def _fake_bridge(monkeypatch):
    modules = {}
    for name in ("megatron", "megatron.bridge", "megatron.bridge.training",
                 "megatron.bridge.training.initialize"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    initialize = modules["megatron.bridge.training.initialize"]

    def original():
        raise AssertionError("Wrong build callback called")

    original.__name__ = "compile_helpers"
    original.__module__ = "megatron.core.datasets.utils"
    initialize.compile_helpers = original
    return initialize


def test_bridge_imported_alias_is_replaced_idempotently(monkeypatch):
    initialize = _fake_bridge(monkeypatch)
    configure_megatron_dataset_builder()
    configure_megatron_dataset_builder()
    assert initialize.compile_helpers is compile_megatron_helpers


def test_bridge_callback_drift_fails_clearly(monkeypatch):
    initialize = _fake_bridge(monkeypatch)
    initialize.compile_helpers = lambda: None
    with pytest.raises(RuntimeError, match="callback changed"):
        configure_megatron_dataset_builder()


def test_actual_nemo_setup_installs_callback_before_initialization(monkeypatch):
    initialize = _fake_bridge(monkeypatch)
    source = ROOT / "third_party/nemo-rl/nemo_rl/models/megatron/setup.py"
    tree = ast.parse(source.read_text())
    setup = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "setup_model_and_optimizer")
    start = next(i for i, n in enumerate(setup.body)
                 if isinstance(n, ast.ImportFrom)
                 and n.module == "dflash_reflex.dataset_helpers")
    stop = next(i for i, n in enumerate(setup.body)
                if isinstance(n, ast.Expr) and ast.unparse(n).startswith("initialize_megatron("))
    observations = []
    namespace = {
        "initialize_megatron": lambda **kwargs: observations.append(initialize.compile_helpers),
        "megatron_cfg": object(), "get_embedding_ranks": None, "get_position_embedding_ranks": None,
    }
    exec(compile(ast.Module(body=setup.body[start:stop + 1], type_ignores=[]),
                 "actual_nemo_dataset_startup", "exec"), namespace)
    assert observations == [compile_megatron_helpers]
