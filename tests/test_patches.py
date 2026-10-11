"""Reproduce every vendor hook from pristine pinned Git objects, twice."""

import ast
import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vendor_patches_reproducible_and_idempotent(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "source_patcher", ROOT / "scripts/patch_upstream.py"
    )
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    for name, patch, package in (
        ("nemo-rl", patcher.patch_nemo, "nemo_rl"),
        ("vllm", patcher.patch_vllm, "vllm"),
    ):
        repo = ROOT / "third_party" / name
        paths = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "--name-only"], text=True
        ).splitlines()
        assert paths, "expected explicit source integration patches"
        for relative in paths:
            target = tmp_path / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{relative}"])
            )
        patch(tmp_path / name / package)
        patch(tmp_path / name / package)
        for relative in paths:
            target = tmp_path / name / relative
            assert target.read_bytes() == (repo / relative).read_bytes()
            ast.parse(target.read_text())


def test_previous_installed_vllm_hooks_migrate_in_place(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "source_patcher_migration", ROOT / "scripts/patch_upstream.py"
    )
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    repo = ROOT / "third_party/vllm"
    paths = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--name-only"], text=True
    ).splitlines()
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            subprocess.check_output(["git", "-C", str(repo), "show", "HEAD:" + relative])
        )
    subprocess.run(
        ["git", "apply", str(ROOT / "tests/reference/vllm_legacy.patch")], cwd=tmp_path, check=True
    )
    patcher.patch_vllm(tmp_path / "vllm")
    patcher.patch_vllm(tmp_path / "vllm")
    for relative in paths:
        assert (tmp_path / relative).read_bytes() == (repo / relative).read_bytes()
