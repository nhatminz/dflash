"""Qwen2.5 targets retain their biased attention; DFlash stays bias-free."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config

from dflash_reflex.block_training import block_attention_mask, block_loss_reference, slot_weights
from dflash_reflex.model_contract import (
    draft_config,
    fingerprint,
    validate_checkpoint,
    validate_mask_token,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"qwen25_1p5b": (16, 2), "qwen25_3b": (8, 4), "qwen25_7b": (8, 4), "qwen25_14b": (4, 8)}


def test_six_pretrain_launchers_can_resolve_concurrently_without_shared_run_paths(tmp_path):
    keys = [*MODELS, "qwen3_1p7b", "qwen3_4b"]
    environment = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "MODEL",
            "MODEL_KEY",
            "MODEL_ENV",
            "RUN_DIR",
            "RUN_NAME",
            "RESUME",
            "PRETRAIN_BATCH_SIZE",
            "PRETRAIN_ACCUMULATION_STEPS",
        }
    }
    processes = [
        subprocess.Popen(
            ["bash", str(ROOT / f"pretrain_{key}.sh")],
            env=dict(
                environment,
                CUDA_VISIBLE_DEVICES=str(index),
                NPROC_PER_NODE="1",
                DRY_RUN="true",
                OUTPUT_ROOT=str(tmp_path),
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for index, key in enumerate(keys)
    ]
    paths = []
    for key, process in zip(keys, processes, strict=True):
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        path = Path(stdout.splitlines()[0].removeprefix("Run directory: "))
        assert path.parent == tmp_path / "pretrain" / key
        assert "--nproc_per_node=1" in stdout
        command = shlex.split(stdout.split("Command:", 1)[1])
        assert command[command.index("--batch-size") + 1] == "8"
        assert command[command.index("--accumulation") + 1] == "4"
        paths.append(path)
    assert len(set(paths)) == 6
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("model", MODELS)
def test_real_qwen25_geometry_and_native_configuration(model, tmp_path):
    target = json.loads((ROOT / f"third_party/reference_{model}_config.json").read_text())
    draft = draft_config(target)
    config = Qwen3Config.from_dict(draft)
    assert config.model_type == "qwen3"
    assert config.head_dim == target["hidden_size"] // target["num_attention_heads"]
    assert config.hidden_size == target["hidden_size"]
    assert config.vocab_size == target["vocab_size"]
    assert config.num_key_value_heads == target["num_key_value_heads"]
    assert not config.attention_bias and not config.sliding_window
    assert config.layer_types == ["full_attention"] * 3
    assert config.rope_theta == target["rope_theta"]
    assert draft["dflash_config"]["mask_token_id"] < target["vocab_size"]
    # No override: exercise actual defaults, including per-model batch sizes.
    environment = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "MODEL",
            "MODEL_KEY",
            "FAIRNESS_PROFILE",
            "BATCH_SIZE",
            "ACCUMULATION_STEPS",
            "RESPONSES_PER_PROMPT",
            "RUN_DIR",
            "RUN_NAME",
            "RESUME",
            "PRETRAIN_BATCH_SIZE",
            "PRETRAIN_ACCUMULATION_STEPS",
        }
    }
    environment.update(DRY_RUN="true", OUTPUT_ROOT=str(tmp_path))
    result = subprocess.run(
        ["bash", str(ROOT / f"pretrain_{model}.sh")],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "/workspace/storage-shared/models/Qwen2.5-" in result.stdout
    draft_args = shlex.split(result.stdout.split("Command:", 1)[1])
    spec = subprocess.run(
        ["bash", str(ROOT.parent / "SpecNaacl" / f"pretrain_{model}.sh")],
        env=dict(environment, PYTHON_BIN=sys.executable),
        text=True,
        capture_output=True,
        check=True,
    )
    spec_args = shlex.split(spec.stdout.split("Command:", 1)[1])
    spec_batch = int(spec_args[spec_args.index("--batch_size") + 1])
    spec_accum = int(spec_args[spec_args.index("--accumulation_steps") + 1])
    draft_batch = int(draft_args[draft_args.index("--batch-size") + 1])
    draft_accum = int(draft_args[draft_args.index("--accumulation") + 1])
    assert draft_batch == 8
    assert draft_accum == 4
    # The user explicitly chose batch8/accum4. Report the effective-batch
    # difference from the actual SpecNaacl CLI instead of claiming parity.
    ratio = {"qwen25_1p5b": 2, "qwen25_3b": 4, "qwen25_7b": 8, "qwen25_14b": 16}[model]
    assert draft_batch * draft_accum == spec_batch * spec_accum * ratio
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "scripts/compare_specnaacl_config.py"),
            "--model",
            model,
            "--output",
            str(tmp_path / "matrix.json"),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    matrix = json.loads(result.stdout)
    assert matrix["profile"] == "native_dflash" and matrix["supported"]
    rows = {row["parameter"]: row for row in matrix["entries"]}
    batch, accum = MODELS[model]
    assert rows["MODEL"]["status"] == "MATCH"
    assert rows["BATCH_SIZE"]["effective_runtime_value"] == batch
    assert rows["BATCH_SIZE"]["status"] == "MATCH"
    assert rows["ACCUMULATION_STEPS"]["SpecNaacl_value"] == accum
    assert matrix["resolved_DFlash"]["policy"]["train_global_batch_size"] == batch * 8
    assert not list(tmp_path.glob("pretrain/*"))


def test_actual_biased_qwen2_target_with_headless_dflash_training_and_export(tmp_path):
    from test_dflash import original_model

    torch.manual_seed(71)
    target = (
        Qwen2ForCausalLM(
            Qwen2Config(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                vocab_size=48,
                bos_token_id=1,
                eos_token_id=2,
                rope_theta=10000,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    assert target.model.layers[0].self_attn.q_proj.bias is not None
    target_config = target.config.to_dict()
    # Serialize the actual target family, not a fabricated Qwen3 target.
    target_path = tmp_path / "target"
    target_path.mkdir()
    (target_path / "config.json").write_text(json.dumps(target_config))
    (target_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"x": 3}}}))
    (target_path / "tokenizer_config.json").write_text("{}")
    validate_mask_token(target_path, 47)
    cfg = draft_config(target_config, block_size=4, num_layers=2, mask_token_id=47)
    draft = type(original_model())(Qwen3Config.from_dict(cfg))
    assert draft.layers[0].self_attn.q_proj.bias is None
    captured = {}
    handles = []
    for layer in cfg["dflash_config"]["target_layer_ids"]:

        def capture(module, args, output, key=layer):
            captured[key] = output[0] if isinstance(output, tuple) else output

        handles.append(target.model.layers[layer].register_forward_hook(capture))
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9]])
    with torch.no_grad():
        teacher = target(ids, use_cache=False).logits
        taps = torch.cat(
            [captured[layer] for layer in cfg["dflash_config"]["target_layer_ids"]], -1
        )
        anchors = torch.tensor([[1, 3]])
        positions = anchors[..., None] + torch.arange(4)
        query_ids = torch.full((1, 8), 47)
        query_ids[:, ::4] = ids.gather(1, anchors)
        embeds = target.get_input_embeddings()(query_ids)
    for handle in handles:
        handle.remove()
    hidden = draft(
        position_ids=torch.cat((torch.arange(7)[None], positions.flatten(1)), 1),
        noise_embedding=embeds,
        target_hidden=taps,
        attention_mask=block_attention_mask(anchors, 7, 4),
    )
    student = target.get_output_embeddings()(hidden).view(1, 2, 4, 48)[:, :, 1:]
    loss = block_loss_reference(student, teacher, anchors, torch.ones(1, 2, 3), slot_weights(4))
    optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-3)
    previous = draft.fc.weight.detach().clone()
    loss.backward()
    assert torch.isfinite(loss) and draft.fc.weight.grad.norm() > 0
    assert all(parameter.grad is None for parameter in target.parameters())
    optimizer.step()
    assert not torch.equal(previous, draft.fc.weight)
    checkpoint = tmp_path / "draft"
    draft.save_pretrained(checkpoint)
    (checkpoint / "dflash_manifest.json").write_text(
        json.dumps(
            {
                "target_fingerprint": fingerprint(target_path),
            }
        )
    )
    exported = validate_checkpoint(target_path, checkpoint, 4)
    assert exported["model_type"] == "qwen3"
    restored = type(draft).from_pretrained(checkpoint)
    for before, after in zip(draft.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(before, after)
    # The same geometry cannot bypass target/tokenizer identity on resume.
    target_config["model_type"] = "qwen3"
    (target_path / "config.json").write_text(json.dumps(target_config))
    with pytest.raises(ValueError, match="different target/tokenizer"):
        validate_checkpoint(target_path, checkpoint, 4)


def test_mask_validation_rejects_tokenizer_and_special_token_collisions(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 48, "eos_token_id": [2, 3]}))
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"x": 4}}}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"added_tokens_decoder": {"47": {}}})
    )
    for token, message in [(47, "unused"), (4, "unused"), (2, "EOS"), (48, "outside")]:
        with pytest.raises(ValueError, match=message):
            validate_mask_token(tmp_path, token)
