import json
from pathlib import Path

import pytest
import torch
import yaml
from torch.distributed import checkpoint as dcp

from analysis.verification_feedback.checkpoints import (
    inspect_checkpoint,
    iteration_path,
    save_tensor_stream,
    validate_export,
)
from analysis.verification_feedback.config import Config
from analysis.verification_feedback.io import atomic_json, file_hash
from dflash_reflex.model_contract import fingerprint


def fixture_run(tmp_path, steps=(250, 600), prefix=""):
    run = tmp_path / "source_run"
    target, draft = tmp_path / "base", tmp_path / "draft_architecture"
    target.mkdir()
    draft.mkdir()
    cfg = dict(
        model_type="qwen2",
        hidden_size=8,
        vocab_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=4,
        eos_token_id=9,
    )
    atomic_json(target / "config.json", cfg)
    atomic_json(target / "tokenizer.json", {"model": {"vocab": {"a": 0}}})
    atomic_json(target / "tokenizer_config.json", {})
    atomic_json(
        draft / "config.json",
        dict(
            cfg,
            model_type="qwen3",
            dflash_config=dict(block_size=3, mask_token_id=15, target_layer_ids=[0, 1, 2]),
        ),
    )
    atomic_json(draft / "dflash_manifest.json", dict(target_fingerprint=fingerprint(target)))
    recipe = {
        "policy": dict(
            model_name=str(target),
            tokenizer={"name": str(target)},
            draft=dict(model_name=str(draft), gamma=2, mask_token_id=15),
        )
    }
    for step in steps:
        folder = run / f"checkpoints/resume/step_{step}"
        weights = folder / "policy/weights/iter_0000000"
        weights.mkdir(parents=True)
        (weights.parent / "latest_checkpointed_iteration.txt").write_text("0")
        state = {
            "model.embedding.word_embeddings.weight": torch.full((16, 8), float(step)),
            "model.draft_model.fc.weight": torch.full((8, 24), float(step) + 0.5),
            "model.draft_model.hidden_norm.weight": torch.ones(24),
            "model.draft_model.decoder.layers.0.self_attention.weight": torch.ones(8, 8),
            "model.draft_model.decoder.final_layernorm.weight": torch.ones(8),
        }
        state = {prefix + k.removeprefix("model."): v for k, v in state.items()}
        dcp.save(state, checkpoint_id=str(weights))
        atomic_json(
            weights / "metadata.json", dict(sharded_backend="torch_dist", sharded_backend_version=1)
        )
        (weights / "run_config.yaml").write_text(yaml.safe_dump(dict(model=cfg)))
        torch.save(dict(iteration=0), weights / "common.pt")
        atomic_json(folder / "training_info.json", dict(total_steps=step, current_epoch=0))
        (folder / "config.yaml").write_text(yaml.safe_dump(recipe))
    return run, {"recipe": recipe}


def test_inspect_and_actual_dcp_loading_step_specific_pairs(tmp_path):
    run, record = fixture_run(tmp_path)
    for step in (250, 600):
        inspected = inspect_checkpoint(run, step, record)
        assert inspected["target"]["source"] == inspected["draft"]["source"]
        assert inspected["training_info"]["total_steps"] == step
        state = {
            "embedding.word_embeddings.weight": torch.zeros(16, 8),
            "draft_model.fc.weight": torch.zeros(8, 24),
        }
        dcp.load(state, checkpoint_id=inspected["iteration_path"])
        assert torch.all(state["embedding.word_embeddings.weight"] == step)
        assert torch.all(state["draft_model.fc.weight"] == step + 0.5)


def test_missing_and_wrong_step_are_precise_errors(tmp_path):
    run, record = fixture_run(tmp_path, (250,))
    with pytest.raises(FileNotFoundError, match="no fallback"):
        inspect_checkpoint(run, 600, record)
    atomic_json(run / "checkpoints/resume/step_250/training_info.json", dict(total_steps=251))
    with pytest.raises(ValueError, match="actual total_steps=251"):
        inspect_checkpoint(run, 250, record)


def test_truncated_real_storage_shard_fails_before_model_init(tmp_path):
    run, record = fixture_run(tmp_path, (250,))
    shard = next((run / "checkpoints/resume/step_250").rglob("*.distcp"))
    shard.write_bytes(b"broken")
    with pytest.raises(ValueError, match="truncated"):
        inspect_checkpoint(run, 250, record)


def test_missing_draft_weights_never_substitutes_pretrained(tmp_path):
    run, record = fixture_run(tmp_path, (250,))
    weights = iteration_path(run / "checkpoints/resume/step_250/policy/weights")
    dcp.save({"embedding.word_embeddings.weight": torch.ones(16, 8)}, checkpoint_id=str(weights))
    with pytest.raises(ValueError, match="both step-specific"):
        inspect_checkpoint(run, 250, record)


def test_source_is_read_only_and_hash_cache_reuses_verified_identity(tmp_path):
    run, record = fixture_run(tmp_path, (250,))
    before = {str(p): file_hash(p) for p in run.rglob("*") if p.is_file()}
    cache = tmp_path / "analysis/hash.json"
    a = inspect_checkpoint(run, 250, record, cache)
    b = inspect_checkpoint(run, 250, record, cache)
    assert a == b
    assert before == {str(p): file_hash(p) for p in run.rglob("*") if p.is_file()}


def test_bridge_tracker_and_native_unprefixed_tensor_keys(tmp_path):
    run, record = fixture_run(tmp_path, (250,))
    root = run / "checkpoints/resume/step_250/policy/weights"
    torch.save(dict(step=0), root / "latest_train_state.pt")
    (root / "latest_checkpointed_iteration.txt").write_text("999")
    inspected = inspect_checkpoint(run, 250, record)
    assert inspected["target"]["tensor_keys"] == ["embedding.word_embeddings.weight"]
    assert inspected["megatron_iteration_directory"] == "iter_0000000"
    torch.save(dict(step=1), root / "latest_train_state.pt")
    with pytest.raises(FileNotFoundError, match="Bridge tracker"):
        inspect_checkpoint(run, 250, record)


def test_common_metadata_is_required_before_checkpoint_qualification(tmp_path):
    run, record = fixture_run(tmp_path, (250,))
    next(run.rglob("common.pt")).unlink()
    with pytest.raises(ValueError, match="common state"):
        inspect_checkpoint(run, 250, record)


def test_export_cache_may_not_mix_steps(tmp_path):
    run, record = fixture_run(tmp_path)
    source = inspect_checkpoint(run, 250, record)
    dest = tmp_path / "analysis/export"
    dest.mkdir(parents=True)
    atomic_json(
        dest / "export_manifest.json",
        dict(
            source_identity=source["source_identity"],
            target_source=source["iteration_path"],
            draft_source="step_600",
            files_sha256={},
        ),
    )
    with pytest.raises(ValueError, match="mixed"):
        validate_export(dest, source)
    atomic_json(dest / "export_manifest.json", dict(source_identity="other", files_sha256={}))
    with pytest.raises(ValueError, match="another checkpoint"):
        validate_export(dest, source)


def test_frozen_export_does_not_modify_parameter_values(tmp_path):
    model = torch.nn.Linear(4, 3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    hashes = save_tensor_stream(tmp_path / "export", model.named_parameters())
    assert set(hashes) == set(before)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    from safetensors.torch import load_file

    index = json.loads((tmp_path / "export/model.safetensors.index.json").read_text())
    for key, filename in index["weight_map"].items():
        assert torch.equal(before[key], load_file(str(tmp_path / "export" / filename))[key])


def test_outputs_may_not_be_inside_original_run(tmp_path):
    c = Config(run_dir=tmp_path / "run", output_dir=tmp_path / "run/analysis")
    with pytest.raises(ValueError, match="outside"):
        c.validate()


def test_export_loader_has_no_training_or_projector_path():
    import ast

    from analysis.verification_feedback import checkpoints

    tree = ast.parse(Path(checkpoints.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(
        isinstance(n.func, ast.Attribute) and n.func.attr in {"step", "backward", "save_checkpoint"}
        for n in calls
    )
    loaders = [
        n
        for n in calls
        if isinstance(n.func, ast.Attribute) and n.func.attr == "setup_model_and_optimizer"
    ]
    assert len(loaders) == 1
    assert any(k.arg == "load_optimizer" and k.value.value is False for k in loaders[0].keywords)


def test_frozen_export_orchestration_loads_exact_pair_and_never_builds_optimizer(
    tmp_path, monkeypatch
):
    """CPU contract test; fake NeMo API around real Torch DCP weights."""
    import sys
    import types
    from contextlib import contextmanager

    from analysis.verification_feedback import checkpoints as module

    run, record = fixture_run(tmp_path, (250,))
    policy = record["recipe"]["policy"]
    policy["megatron_cfg"] = {"distributed_data_parallel_config": {}}
    inspected = inspect_checkpoint(run, 250, record)
    models = torch.nn.Module()
    models.embedding = torch.nn.Embedding(16, 8)
    models.draft_model = torch.nn.Linear(24, 8, bias=False)
    cfg = types.SimpleNamespace(
        checkpoint=types.SimpleNamespace(), model=types.SimpleNamespace(), validate=lambda: None
    )
    setup = types.ModuleType("nemo_rl.models.megatron.setup")
    setup.setup_distributed = lambda policy: None
    setup.validate_and_set_config = lambda *a: types.SimpleNamespace(megatron_cfg=cfg)
    called = []

    def load(policy, config, *, load_optimizer, pre_load_checkpoint_hook):
        assert load_optimizer is False
        called.append("frozen loader")
        pre_load_checkpoint_hook(None, [models])
        state = {
            "embedding.word_embeddings.weight": models.embedding.weight.detach(),
            "draft_model.fc.weight": models.draft_model.weight.detach(),
        }
        dcp.load(state, checkpoint_id=inspected["iteration_path"])
        return types.SimpleNamespace(
            model=models, draft_model=models.draft_model, optimizer=None, scheduler=None
        )

    setup.setup_model_and_optimizer = load

    class Bridge:
        @classmethod
        def from_hf_pretrained(cls, *a, **k):
            return cls()

        def export_hf_weights(self, models, **kwargs):
            return [("model.embed_tokens.weight", models[0].embedding.weight)]

    @contextmanager
    def detached(models):
        yield

    modules = {
        "megatron": {},
        "megatron.bridge": {"AutoBridge": Bridge},
        "nemo_rl": {},
        "nemo_rl.models": {},
        "nemo_rl.models.megatron": {"setup": setup},
        "nemo_rl.models.megatron.setup": setup,
        "nemo_rl.models.megatron.draft": {
            "export_draft_weights_to_hf": lambda model: [("fc.weight", model.weight)]
        },
        "nemo_rl.models.megatron.draft.utils": {
            "draft_model_detached": detached,
            "load_hf_weights_to_block_draft": lambda *a: (_ for _ in ()).throw(
                AssertionError("Pretrained weights read")
            ),
        },
    }
    for name, attributes in modules.items():
        fake = types.ModuleType(name) if isinstance(attributes, dict) else attributes
        if isinstance(attributes, dict):
            fake.__dict__.update(attributes)
            fake.__path__ = []
        monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setattr(
        module,
        "require_coverage",
        lambda models, path: {"required_tensors": 2} if path == inspected["iteration_path"] else {},
    )
    cache = tmp_path / "analysis/export_step_250"
    module.export_checkpoint(record, inspected, cache)
    assert called == ["frozen loader"]
    assert torch.all(models.embedding.weight == 250)
    assert torch.all(models.draft_model.weight == 250.5)
    assert not any(p.requires_grad for p in models.parameters())
    from analysis.verification_feedback.io import read_json

    manifest = read_json(cache / "export_manifest.json")
    assert manifest["target_source"] == manifest["draft_source"] == inspected["iteration_path"]
    assert manifest["optimizer_created"] is False
    assert manifest["projector_created"] is False
