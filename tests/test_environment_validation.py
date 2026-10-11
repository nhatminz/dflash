"""Locked binary identity, metadata mismatch and final runtime reporting."""

import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERSION = "2.8.1+cu13torch2.10cxx11abitrue"
FILENAME = "flash_attn-2.8.1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
URL = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/" + FILENAME


def load_script(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    module = load_script("runtime_artifacts", monkeypatch)
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    site = tmp_path / "site"
    info = "flash_attn-2.8.1.dist-info"
    payload = {
        "flash_attn/__init__.py": b'__version__ = "2.8.1"\n',
        "flash_attn_2_cuda.cpython-313-x86_64-linux-gnu.so": b"locked binary fixture",
        info + "/METADATA": b"Metadata-Version: 2.2\nName: flash_attn\nVersion: 2.8.1\n",
        info + "/WHEEL": b"Wheel-Version: 1.0\nTag: cp313-cp313-linux_x86_64\n",
    }
    record = io.StringIO()
    writer = csv.writer(record)
    for name, data in payload.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((info + "/RECORD", "", ""))
    payload[info + "/RECORD"] = record.getvalue().encode()
    wheel = tmp_path / "offline_packages/wheels" / FILENAME
    wheel.parent.mkdir(parents=True)
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
            path = site / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    digest = module.file_sha256(wheel)
    lock = tmp_path / "third_party/nemo-rl/uv.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        f'[[package]]\nname="flash-attn"\nversion="{VERSION}"\n'
        f'wheels=[{{url="{URL}",hash="sha256:{digest}"}}]\n'
    )
    dist = importlib.metadata.PathDistribution(site / info)
    return module, tmp_path, site, wheel, dist


def test_upstream_base_metadata_requires_exact_locked_payload(artifact):
    module, root, _, _, dist = artifact
    assert dist.version == "2.8.1" and dist.version != VERSION
    result = module.qualify_flash_attn(root, VERSION, dist)
    assert result["verification"] == "locked-wheel-record"
    assert result["files_verified"] == 4
    assert result["locked_version"] == VERSION


def test_mcore_inspector_runs_artifact_check_inside_backend_process(artifact, monkeypatch):
    """Exercise the actual subprocess protocol with CPU-only module fixtures."""
    _, root, site, _, _ = artifact
    module = load_script("validate_environment", monkeypatch)
    monkeypatch.setattr(module, "ROOT", root)
    lock, _ = actor_state()
    (root / "versions.lock").write_text(json.dumps(lock))
    (root / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/runtime_artifacts.py", root / "scripts/runtime_artifacts.py")
    for package in ("torch", "triton", "transformers", "ray", "transformer-engine"):
        info = site / (package.replace("-", "_") + "-fixture.dist-info")
        info.mkdir()
        (info / "METADATA").write_text(f"Name: {package}\nVersion: {lock['runtime'][package]}\n")
    modules = {
        "torch.py": """from types import SimpleNamespace
version = SimpleNamespace(cuda='13.0')
cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda: 'fixture',
                      get_device_capability=lambda: (10, 0), is_bf16_supported=lambda: True)
""",
        "megatron/core.py": "",
        "megatron/bridge.py": "",
        "transformer_engine/pytorch.py": "",
        "nemo_rl/models/megatron/draft/dflash.py": "class DFlashDraftModel: pass\n",
        "nemo_rl/algorithms/loss/loss_functions.py": "class BlockDraftLossFn: pass\n",
    }
    for name, content in modules.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        for parent in path.parents:
            if parent == site:
                break
            (parent / "__init__.py").touch()
    monkeypatch.setenv("PYTHONPATH", str(site))
    result = module.inspect_python(sys.executable, "mcore")
    assert result["flash_attn"] == "2.8.1"
    assert result["flash_attn_artifact"]["files_verified"] == 4
    assert result["dflash_model"] and result["bf16"]


@pytest.mark.parametrize("modified", ["wheel", "extension", "metadata", "missing-extension"])
def test_wrong_binary_or_metadata_still_rejected(artifact, modified):
    module, root, site, wheel, dist = artifact
    extension = site / "flash_attn_2_cuda.cpython-313-x86_64-linux-gnu.so"
    if modified == "wheel":
        with wheel.open("ab") as stream:
            stream.write(b"changed")
    elif modified == "extension":
        extension.write_bytes(b"wrong binary fixture!")
    elif modified == "missing-extension":
        extension.unlink()
    else:
        (site / "flash_attn-2.8.1.dist-info/METADATA").write_text(
            "Name: flash_attn\nVersion: 2.8.2\n"
        )
    with pytest.raises(RuntimeError, match="SHA256 mismatch|differs|metadata="):
        module.qualify_flash_attn(root, VERSION, dist)


@pytest.mark.parametrize("provenance", ["correct", "missing", "wrong-url", "wrong-hash"])
def test_direct_uv_install_requires_locked_url_and_sha256(artifact, provenance):
    module, root, site, wheel, dist = artifact
    digest = module.file_sha256(wheel)
    wheel.unlink()
    if provenance != "missing":
        (site / "flash_attn-2.8.1.dist-info/direct_url.json").write_text(
            json.dumps(
                dict(
                    url=URL if provenance != "wrong-url" else URL.replace("cu13", "cu12"),
                    archive_info=dict(
                        hashes=dict(sha256=digest if provenance != "wrong-hash" else "0" * 64)
                    ),
                )
            )
        )
    if provenance == "correct":
        assert module.qualify_flash_attn(root, VERSION, dist)["verification"] == (
            "locked-direct-url-sha256"
        )
    else:
        with pytest.raises(RuntimeError, match="Cannot prove"):
            module.qualify_flash_attn(root, VERSION, dist)


def actor_state():
    lock = json.loads((ROOT / "versions.lock").read_text())
    state = {name: lock["runtime"][name] for name in ("torch", "triton", "transformers", "ray")}
    state.update(
        python="3.13.16",
        cuda="13.0",
        bf16=True,
        flash_attn="2.8.1",
        transformer_engine=lock["runtime"]["transformer-engine"],
        flash_attn_artifact=dict(locked_version=VERSION, verification="locked-wheel-record"),
        vllm=lock["vllm_version"],
        reflex_patch=True,
    )
    return lock, state


def test_metadata_fix_preserves_runtime_requirements(monkeypatch):
    module = load_script("validate_environment", monkeypatch)
    lock, state = actor_state()
    assert module.runtime_errors(state, "mcore", lock) == []
    state.pop("flash_attn_artifact")
    state.update(torch="2.10.0+cu130", cuda="12.8", bf16=False, transformer_engine="2.14.1")
    errors = module.runtime_errors(state, "mcore", lock)
    assert len(errors) == 4
    assert any("metadata=2.8.1" in error for error in errors)
    state.update(vllm="0.25.0", reflex_patch=False)
    assert any(
        "Reflex patch=False" in error for error in module.runtime_errors(state, "vllm", lock)
    )


def test_runtime_failure_collects_all_actors_and_saves_report(tmp_path, monkeypatch):
    module = load_script("validate_environment", monkeypatch)
    lock, state = actor_state()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "versions.lock").write_text(json.dumps(lock))
    (tmp_path / "SOURCE_MANIFEST.json").write_text('{"SpecNaacl_files": {}}')
    heads = iter(lock[name]["sha"] for name in ("nemo-rl", "vllm", "original-dflash"))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: next(heads))
    calls = []

    def inspect(python, backend):
        calls.append(backend)
        if backend == "mcore":
            return dict(state, transformer_engine="wrong")
        if backend == "vllm":
            raise RuntimeError("vllm import failed")
        return state.copy()

    monkeypatch.setattr(module, "inspect_python", inspect)
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["validate", "--require-runtime", "--output", str(output)])
    with pytest.raises(RuntimeError, match="transformer_engine=wrong"):
        module.main()
    report = json.loads(output.read_text())
    assert calls == ["base", "mcore", "vllm"]
    assert not report["runtime_valid"] and len(report["runtime_errors"]) == 2
    assert report["environments"]["mcore"]["flash_attn"] == "2.8.1"


def test_finalize_only_never_installs_or_requires_source_bundle(tmp_path, monkeypatch):
    module = load_script("install_local_packages", monkeypatch)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "offline_packages").mkdir()
    for name in (".venv-local", ".venv-local-mcore", ".venv-local-vllm"):
        (tmp_path / name / "bin").mkdir(parents=True)
        (tmp_path / name / "bin/python").touch()
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kw: calls.append(command))
    monkeypatch.setattr(sys, "argv", ["installer", "--finalize-only"])
    module.main()
    assert len(calls) == 2
    assert "--vllm-installed" in calls[0]
    assert "--require-runtime" in calls[1] and "--output" in calls[1]
    assert (tmp_path / "offline_packages/activate.sh").is_file()
    assert not (tmp_path / "offline_packages/manifest.json").exists()


def test_finalize_only_propagates_failed_qualification(tmp_path, monkeypatch):
    module = load_script("install_local_packages", monkeypatch)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "offline_packages").mkdir()
    for name in (".venv-local", ".venv-local-mcore", ".venv-local-vllm"):
        (tmp_path / name / "bin").mkdir(parents=True)
        (tmp_path / name / "bin/python").touch()

    def run(command, **kwargs):
        if "--require-runtime" in command:
            raise module.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["installer", "--finalize-only"])
    with pytest.raises(module.subprocess.CalledProcessError):
        module.main()
