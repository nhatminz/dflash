"""Launch-time quick/full policy, without importing or installing GPU stacks."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def launch_checks(tmp_path, *, kind="train", mode=None, missing=None, failure=None):
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"MODEL", "MODEL_ENV", "COMMON_ENV", "DFLASH_STARTUP_CHECKS", "RUNTIME_CHECK"}
    }
    record = tmp_path / "calls"
    interpreters = {}
    for backend in ("base", "mcore", "vllm"):
        path = tmp_path / f"{backend}-python"
        if backend != missing:
            path.write_text(
                "#!/usr/bin/env bash\n"
                'printf "%s %s\\n" "${0##*/}" "$*" >> "$CHECK_CALLS"\n'
                'if [[ "${0##*/}" == "$CHECK_FAILURE" ]]; then exit 17; fi\n'
            )
            path.chmod(0o755)
        interpreters[backend] = str(path)
    environment.update(
        MODEL_KEY="qwen25_3b",
        PYTHON_BIN=interpreters["base"],
        DFLASH_MCORE_PYTHON=interpreters["mcore"],
        DFLASH_VLLM_PYTHON=interpreters["vllm"],
        CHECK_CALLS=str(record),
        CHECK_FAILURE=f"{failure}-python" if failure else "",
        CHECK_KIND=kind,
    )
    if mode is not None:
        environment["DFLASH_STARTUP_CHECKS"] = mode
    result = subprocess.run(
        ["bash", "-c", 'source scripts/launch/common.sh; startup_checks "$CHECK_KIND"'],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    return result, record.read_text().splitlines() if record.exists() else []


@pytest.mark.parametrize("kind", ["train", "pretrain"])
def test_default_invokes_auto_cache_policy(tmp_path, kind):
    result, calls = launch_checks(tmp_path, kind=kind)
    assert result.returncode == 0
    assert len(calls) == 1
    assert calls[0].endswith("scripts/runtime_checks.py --mode auto")


@pytest.mark.parametrize("kind", ["train", "pretrain"])
def test_legacy_quick_skips_heavy_probes(tmp_path, kind):
    result, calls = launch_checks(tmp_path, kind=kind, mode="quick")
    assert result.returncode == 0
    assert "Runtime check: skip" in result.stdout
    assert calls == []


def test_full_calls_cached_validator_in_force_mode(tmp_path):
    result, calls = launch_checks(tmp_path, mode="full")
    assert result.returncode == 0
    assert len(calls) == 1
    assert calls[0].endswith("scripts/runtime_checks.py --mode full")


@pytest.mark.parametrize(
    "kind,missing", [("train", "mcore"), ("train", "vllm"), ("pretrain", "mcore")]
)
def test_still_rejects_missing_actor_python(tmp_path, kind, missing):
    result, calls = launch_checks(tmp_path, kind=kind, missing=missing)
    assert result.returncode == 2
    assert "Missing" in result.stderr
    assert calls == []


def test_skip_pretrain_does_not_require_vllm(tmp_path):
    result, _ = launch_checks(tmp_path, kind="pretrain", missing="vllm", mode="quick")
    assert result.returncode == 0


def test_full_failure_keeps_original_status(tmp_path):
    result, calls = launch_checks(tmp_path, mode="full", failure="base")
    assert result.returncode == 17
    assert len(calls) == 1


def test_invalid_mode_fails(tmp_path):
    result, calls = launch_checks(tmp_path, mode="ful")
    assert result.returncode == 2
    assert "must be quick or full" in result.stderr
    assert calls == []
