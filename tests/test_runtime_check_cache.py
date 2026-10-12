import multiprocessing
import subprocess
import sys
import time

import pytest

from scripts import runtime_checks as cache


def test_cache_only_success_auto_hit_full_and_invalidation(tmp_path):
    revision = ["old"]
    calls = []

    def identify():
        return (revision[0], dict(version=revision[0]))

    def validate():
        return calls.append("deep")

    cache.qualify("auto", tmp_path, identify=identify, validate=validate)
    cache.qualify("auto", tmp_path, identify=identify, validate=validate)
    assert calls == ["deep"]
    revision[0] = "new"
    cache.qualify("auto", tmp_path, identify=identify, validate=validate)
    cache.qualify("full", tmp_path, identify=identify, validate=validate)
    assert calls == ["deep"] * 3
    cache.qualify(
        "skip",
        tmp_path,
        identify=lambda: pytest.fail("skip fingerprint"),
        validate=lambda: pytest.fail("skip imports"),
    )


def test_failed_and_racing_environment_not_cached(tmp_path):
    def fail():
        raise subprocess.CalledProcessError(17, "validator")

    with pytest.raises(subprocess.CalledProcessError):
        cache.qualify("auto", tmp_path, identify=lambda: ("a", {}), validate=fail)
    assert not (tmp_path / "validation.json").exists()
    versions = iter([("a", {}), ("b", {})])
    with pytest.raises(RuntimeError, match="changed during validation"):
        cache.qualify("auto", tmp_path, identify=lambda: next(versions), validate=lambda: None)
    assert not (tmp_path / "validation.json").exists()


def worker(directory):
    def validate():
        with (directory / "calls").open("a") as stream:
            stream.write("deep\n")
        time.sleep(0.1)

    cache.qualify("auto", directory, identify=lambda: ("same", {}), validate=validate)


def test_simultaneous_launches_validate_once(tmp_path):
    context = multiprocessing.get_context("spawn")
    jobs = [context.Process(target=worker, args=(tmp_path,)) for _ in range(8)]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join(5)
        assert job.exitcode == 0
    assert (tmp_path / "calls").read_text().splitlines() == ["deep"]


def test_fingerprint_detects_locks_commits_metadata_binary_and_installed_hooks(
    tmp_path, monkeypatch
):
    site = tmp_path / "site"
    site.mkdir()
    for relative in ("versions.lock", "scripts/runtime_checks.py", "patches/vllm.patch"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("initial")
    for relative in (
        "torch-fixture.dist-info/METADATA",
        "torch/lib/test.so",
        "vllm/v1/spec_decode/llm_base_proposer.py",
    ):
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("initial")
    revision = ["head"]

    def probe(command, **kwargs):
        if command[0] == "git":
            return revision[0] if "rev-parse" in command else b""
        import json

        return json.dumps(["fixture", str(tmp_path), [str(site)]])

    monkeypatch.setattr(cache.subprocess, "check_output", probe)
    environments = dict(base=sys.executable, mcore=sys.executable, vllm=sys.executable)

    def key():
        return cache.fingerprint(tmp_path, environments)[0]

    previous = key()
    assert key() == previous
    for path in (
        tmp_path / "versions.lock",
        tmp_path / "patches/vllm.patch",
        site / "torch-fixture.dist-info/METADATA",
        site / "torch/lib/test.so",
        site / "vllm/v1/spec_decode/llm_base_proposer.py",
    ):
        path.write_text("changed")
        assert key() != previous
        previous = key()
    revision[0] = "other-head"
    assert key() != previous
