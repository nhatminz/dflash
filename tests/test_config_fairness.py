import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_matched_profile_fails_before_loading_models(tmp_path):
    p = subprocess.run(
        ["bash", str(ROOT / "train_qwen3_4b.sh")],
        capture_output=True,
        text=True,
        env=dict(
            os.environ,
            DRY_RUN="true",
            FAIRNESS_PROFILE="specnaacl_matched",
            OUTPUT_ROOT=str(tmp_path),
        ),
    )
    assert p.returncode != 0
    for word in ("LoRA", "four-rollout", "constant-reward"):
        assert word in p.stderr
    assert not list(tmp_path.iterdir())


def test_comparison_is_stdlib_and_covers_real_loss_and_optimizer(tmp_path):
    path = tmp_path / "matrix.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "scripts/compare_specnaacl_config.py"),
            "--profile",
            "native_dflash",
            "--output",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(path.read_text())
    rows = {r["parameter"]: r for r in report["entries"]}
    assert rows["BETA"]["effective_runtime_value"] == 0.04
    assert rows["EPSILON"]["effective_runtime_value"] == 0.1
    assert rows["TARGET_LORA_R"]["status"] == "UNSUPPORTED"
    assert rows["EFFECTIVE_TARGET_BATCH"]["effective_runtime_value"] == 64
    assert not report["strict_cross_framework_equivalence"]
    assert rows["reward_std"]["status"] == "DIFFERENT_BY_DESIGN"
    proc = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "scripts/compare_specnaacl_config.py"),
            "--model",
            "qwen3_8b",
            "--profile",
            "native_dflash",
            "--output",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(path.read_text())
    assert not report["SpecNaacl_model_config_present"]
    rows = {r["parameter"]: r for r in report["entries"]}
    assert rows["MODEL"]["status"] == "UNVERIFIED"
    assert rows["EFFECTIVE_TARGET_BATCH"]["status"] == "UNVERIFIED"


def test_original_native_loss_contract_is_available_for_legacy_resume():
    p = subprocess.run(
        ["bash", str(ROOT / "train_qwen3_4b.sh")],
        capture_output=True,
        text=True,
        env=dict(
            os.environ,
            DRY_RUN="true",
            FAIRNESS_PROFILE="native_dflash",
            GRPO_OBJECTIVE="original_nemo",
            BETA="0",
            EPSILON=".2",
        ),
    )
    assert p.returncode == 0, p.stderr
    config = json.loads(p.stdout[p.stdout.index("{\n") :])
    assert config["loss_fn"] == dict(
        reference_policy_kl_penalty=0.0,
        ratio_clip_min=0.2,
        ratio_clip_max=0.2,
        force_on_policy_ratio=False,
    )


def test_auto_resume_selects_the_requested_method_and_dataset(tmp_path):
    runs = tmp_path / "train/qwen3_4b"
    wanted, other = runs / "reflex_simplelr", runs / "baseline_dapo"
    wanted.mkdir(parents=True)
    other.mkdir()
    (runs / "active_run").symlink_to(other, target_is_directory=True)
    (runs / "active_run_dflash_reflex_simplelr").symlink_to(wanted, target_is_directory=True)
    p = subprocess.run(
        ["bash", str(ROOT / "train_qwen3_4b.sh")],
        capture_output=True,
        text=True,
        env=dict(
            os.environ,
            DRY_RUN="true",
            FAIRNESS_PROFILE="native_dflash",
            OUTPUT_ROOT=str(tmp_path),
            RESUME="auto",
            METHOD="dflash_reflex",
            DATASET="simplelr",
        ),
    )
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines()[0] == "Run directory: " + str(wanted)
