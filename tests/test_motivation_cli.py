import os
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.verification_feedback.collector import EVENT_TYPES, parquet
from analysis.verification_feedback.events import reconstruct
from analysis.verification_feedback.io import atomic_json, digest, file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]


def analysis_fixture(root):
    plan = dict(eval_rollouts=4, baseline="synthetic CPU test fixture; not experimental data")
    atomic_json(root / "evaluation_manifest.json", plan)
    for step in (250, 600):
        directory = root / f"collection/step_{step}"
        atomic_json(
            directory / "manifest.json",
            dict(identity=f"fixture-{step}", evaluation_manifest_sha256=digest(plan)),
        )
        for rollout in range(4):
            req = {
                str(i): dict(
                    response_id=i,
                    sample_index=i,
                    prompt_id=f"p{rollout}",
                    question_sha256=f"p{rollout}",
                    stop_token_ids=[9],
                )
                for i in range(2)
            }
            out = {
                str(i): dict(token_ids=[2, 2, 9], finish_reason="stop", stop_reason=None)
                for i in range(2)
            }
            packets = [
                dict(
                    request_ids=["0", "1"],
                    lengths=[1, 1],
                    draft_ids=[1, 1],
                    outputs=[[2, -1], [2, -1]],
                    discard=[False, False],
                )
                for _ in range(2)
            ]
            packets += [
                dict(
                    request_ids=["0", "1"],
                    lengths=[0, 0],
                    draft_ids=[],
                    outputs=[[9], [9]],
                    discard=[False, False],
                )
            ]
            events, rounds = reconstruct(packets, req, out, step, rollout)
            folder = directory / f"rollout_{rollout:04d}"
            parquet(folder / "events.parquet", events, EVENT_TYPES)
            parquet(folder / "rounds.parquet", rounds)
            atomic_json(folder / "responses.json", out)
            atomic_json(folder / "requests.json", req)
            atomic_json(folder / "audit.json", dict(parity={"status": "passed"}))
            files = {
                name: file_hash(folder / name)
                for name in (
                    "events.parquet",
                    "rounds.parquet",
                    "responses.json",
                    "requests.json",
                    "audit.json",
                )
            }
            atomic_json(
                folder / "complete.json",
                dict(
                    identity=f"fixture-{step}",
                    checkpoint_step=step,
                    rollout_id=rollout,
                    frozen_parameter_parity=True,
                    files_sha256=files,
                    prompts=1,
                    responses=2,
                    rounds=len(rounds),
                ),
            )


def test_offline_cli_generates_all_figures_without_importing_model_stacks(tmp_path):
    analysis_fixture(tmp_path)
    code = """
import importlib.abc, sys, runpy
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'vllm', 'nemo_rl', 'transformers'}:
            raise AssertionError('Offline analysis imported model stack: '+fullname)
sys.meta_path.insert(0, Block())
sys.argv = [sys.argv[1], '--mode', 'analyze', '--output-dir', sys.argv[2], '--bootstrap-samples', '40']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "scripts/run_dflash_motivation.py"), str(tmp_path)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    for name in (
        "fig1a_dflash.pdf",
        "fig1b_dflash.pdf",
        "fig1c_dflash.pdf",
        "fig1_dflash_combined.pdf",
    ):
        assert (tmp_path / name).read_bytes().startswith(b"%PDF")
    assert (tmp_path / "fig1_dflash_combined.png").read_bytes().startswith(b"\x89PNG")
    result = read_json(tmp_path / "statistics_step_250.json")
    assert result["fig1c"]["within_response_prior_miss_share"]["estimate"] == 0.5
    assert result["ordinary_events"] == 16 and result["misses"] == 16
    assert result["rollout_clusters"] == 4
    before = file_hash(tmp_path / "statistics_step_250.json")
    from analysis.verification_feedback.analysis import analyze

    analyze(tmp_path, [250, 600], 40, 2026)
    assert file_hash(tmp_path / "statistics_step_250.json") == before


def test_missing_replay_audit_is_not_reported_as_passed(tmp_path):
    analysis_fixture(tmp_path)
    for p in tmp_path.glob("collection/step_250/rollout_*/audit.json"):
        atomic_json(p, dict(parity={"status": "not_replayed"}))
        marker = p.parent / "complete.json"
        commit = read_json(marker)
        commit["files_sha256"]["audit.json"] = file_hash(p)
        atomic_json(marker, commit)
    from analysis.verification_feedback.analysis import analyze

    with pytest.raises(ValueError, match="no successful genuine"):
        analyze(tmp_path, [250], 40)


def test_cli_refuses_to_write_anywhere_inside_original_run(tmp_path):
    run = tmp_path / "original"
    run.mkdir()
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_dflash_motivation.py"),
            "--mode",
            "check",
            "--run-dir",
            str(run),
            "--output-dir",
            str(run / "analysis"),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "outside the original" in process.stderr
    assert not list(run.iterdir())


def test_manifest_can_add_a_real_stage_but_cannot_replace_an_old_stage(tmp_path):
    from scripts.run_dflash_motivation import save_checkpoint_manifest

    path = tmp_path / "checkpoint_manifest.json"
    save_checkpoint_manifest(
        path, dict(checkpoints=[dict(checkpoint_step=250, source_identity="a")])
    )
    save_checkpoint_manifest(
        path, dict(checkpoints=[dict(checkpoint_step=600, source_identity="b")])
    )
    assert [v["checkpoint_step"] for v in read_json(path)["checkpoints"]] == [250, 600]
    with pytest.raises(ValueError, match="identity changed"):
        save_checkpoint_manifest(
            path, dict(checkpoints=[dict(checkpoint_step=250, source_identity="wrong")])
        )


def test_help_is_stdlib_only_and_has_no_train_mode():
    result = subprocess.run(
        ["python3", str(ROOT / "scripts/run_dflash_motivation.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "{check,collect,analyze,all}" in result.stdout
    assert "--smoke" in result.stdout and "--resume" in result.stdout
    assert "--train-steps" not in result.stdout
