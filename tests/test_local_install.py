"""External artifact selection and offline fetch failure/recovery contracts."""

import importlib.util
import json
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
