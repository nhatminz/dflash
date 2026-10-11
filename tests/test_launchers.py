import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "model",
    ["qwen25_1p5b", "qwen25_3b", "qwen25_7b", "qwen25_14b", "qwen3_1p7b", "qwen3_4b", "qwen3_8b"],
)
@pytest.mark.parametrize("method", ["dflash", "dflash_reflex"])
def test_dry_run_without_models_or_cuda(model, method, tmp_path):
    wrapper = ROOT / f"train_{model}{'_dflash' if method == 'dflash' else ''}.sh"
    process = subprocess.run(
        ["bash", str(wrapper)],
        capture_output=True,
        text=True,
        check=True,
        env=dict(
            os.environ,
            FAIRNESS_PROFILE="native_dflash",
            DRY_RUN="true",
            OUTPUT_ROOT=str(tmp_path),
            DATASET="dapo",
            OPD_TOPK="7",
            BATCH_SIZE="2",
            RESPONSES_PER_PROMPT="4",
            TARGET_LR="2e-6",
        ),
    )
    assert "Command:" in process.stdout
    config = json.loads(process.stdout[process.stdout.index("{\n") :])
    assert config["data"]["train"]["dataset"] == "dapo"
    assert config["policy"]["train_global_batch_size"] == 8
    assert config["policy"]["megatron_cfg"]["optimizer"]["lr"] == 2e-6
    assert not list(tmp_path.iterdir())


def test_unsupported_topology_fails_before_gpu():
    process = subprocess.run(
        ["bash", str(ROOT / "train_qwen3_4b.sh")],
        capture_output=True,
        text=True,
        env=dict(os.environ, DRY_RUN="true", NPROC_PER_NODE="2"),
        check=False,
    )
    assert process.returncode != 0
    assert "one colocated" in process.stderr


def test_resume_rejects_changed_method_or_reflex_parameters(tmp_path):
    import torch
    from safetensors.torch import save_file
    from test_dflash import original_model

    from dflash_reflex.model_contract import fingerprint

    model = original_model()
    target, draft, run = (tmp_path / name for name in ("target", "draft", "run"))
    target.mkdir()
    config = model.config.to_dict()
    config["num_hidden_layers"] = 4
    (target / "config.json").write_text(json.dumps(config))
    (target / "tokenizer.json").write_text("{}")
    (target / "tokenizer_config.json").write_text("{}")
    save_file({"lm_head.weight": torch.randn(48, 16)}, str(target / "model.safetensors"))
    model.save_pretrained(draft)
    (draft / "dflash_manifest.json").write_text(
        json.dumps(dict(target_fingerprint=fingerprint(target)))
    )
    torch.save(dict(A=torch.randn(16, 8), version=0), draft / "opd_projector.pt")
    data = tmp_path / "data.parquet"
    data.write_bytes(b"path validation fixture; no dataset loaded")
    defaults = subprocess.check_output(
        [
            "bash",
            "-c",
            'source "$1"\nsource "$2"\nenv -0',
            "defaults",
            str(ROOT / "configs/qwen3_4b/b200.env"),
            str(ROOT / "configs/_shared/b200_common.env"),
        ],
        env=dict(os.environ, FAIRNESS_PROFILE="native_dflash", METHOD="dflash_reflex"),
    )
    environment = dict(item.decode().split("=", 1) for item in defaults.split(b"\0") if item)
    environment.update(
        MODEL=str(target),
        DRAFT_CHECKPOINT=str(draft),
        DATASET_PATH=str(data),
        DFLASH_RUN_DIR=str(run),
        DFLASH_BLOCK_SIZE="4",
        DFLASH_MASK_TOKEN_ID="47",
        DFLASH_RESUMING="0",
        PYTHONPATH=str(ROOT),
    )
    command = [sys.executable, str(ROOT / "scripts/prepare_run.py"), "train"]
    subprocess.run(command, env=environment, check=True)
    checkpoint = run / "checkpoints/resume/step_1"
    checkpoint.mkdir()
    (checkpoint / "training_info.json").write_text("{}")
    environment["DFLASH_RESUMING"] = "1"
    subprocess.run(command, env=environment, check=True)
    for changed in (dict(METHOD="dflash"), dict(OPD_FAST_LR="0.03")):
        result = subprocess.run(
            command, env=dict(environment, **changed), capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "resume method, Reflex parameters" in result.stderr
