"""Inspect real NeMo torch_dist checkpoints and export a frozen paired snapshot.

Inspection reads DCP metadata, storage ranges and training_info without a model.
Export uses the pinned NeMo/Megatron loader with load_optimizer=False. It never
calls GRPO setup/train, creates a projector, or writes to a source checkpoint.
"""

import copy
import os
import re
import shutil
from pathlib import Path

from .config import ROOT, at
from .io import atomic_json, digest, file_hash, files_snapshot, read_json, verify_files


def iteration_path(weights):
    weights = Path(weights)
    if (weights / ".metadata").is_file():
        return weights
    bridge_tracker = weights / "latest_train_state.pt"
    if bridge_tracker.is_file():
        import torch

        value = torch.load(bridge_tracker, map_location="cpu", weights_only=True)["step"]
        path = weights / f"iter_{int(value):07d}"
        if not (path / ".metadata").is_file():
            raise FileNotFoundError(f"Bridge tracker points to an incomplete checkpoint: {path}")
        return path
    tracker = weights / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        value = tracker.read_text().strip()
        if not value.isdigit():
            raise ValueError(f"Unsupported/non-numeric checkpoint tracker: {tracker}")
        path = weights / f"iter_{int(value):07d}"
        if not (path / ".metadata").is_file():
            raise FileNotFoundError(f"Tracker points to an incomplete checkpoint: {path}")
        return path
    candidates = sorted(p.parent for p in weights.glob("iter_*/.metadata"))
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot identify one exact Megatron iteration under {weights}: {candidates}"
        )
    return candidates[0]


def inspect_checkpoint(run_dir, step, recipe_record, hash_cache=None):
    from torch.distributed.checkpoint import FileSystemReader

    path = Path(run_dir).resolve() / f"checkpoints/resume/step_{step}"
    if not path.is_dir():
        raise FileNotFoundError(
            f"Required real checkpoint missing: {path}; no fallback checkpoint is allowed"
        )
    snapshot = files_snapshot(path)
    info = read_json(path / "training_info.json")
    if info.get("total_steps") != step:
        raise ValueError(
            f"{path}: requested step {step}, actual total_steps={info.get('total_steps')}"
        )
    weights = path / "policy/weights"
    iteration = iteration_path(weights)
    for name in ("metadata.json", "run_config.yaml"):
        if not (iteration / name).is_file():
            raise FileNotFoundError(f"Incomplete native Megatron checkpoint: {iteration / name}")
    backend = read_json(iteration / "metadata.json")
    if backend.get("sharded_backend") != "torch_dist":
        raise ValueError(f"Unsupported native checkpoint backend: {backend}")
    metadata = FileSystemReader(str(iteration)).read_metadata()
    if not (iteration / "common.pt").is_file() and not any(
        key.startswith("common_state/") or key == "common_state"
        for key in metadata.state_dict_metadata
    ):
        raise ValueError(f"Native checkpoint lacks common state: {iteration}")
    keys = sorted(k for k, value in metadata.state_dict_metadata.items() if hasattr(value, "size"))
    draft_keys = [k for k in keys if "draft_model." in k]
    # Native MCore flattens ShardedTensor.key, not {'model': ...} containers.
    target_keys = [
        k
        for k in keys
        if re.match(r"^(?:model\d*\.)?(?:module\.)?(?:embedding|decoder|output_layer)\.", k)
        and "draft_model." not in k
    ]
    if not target_keys or not draft_keys:
        raise ValueError(
            f"{path}: target tensors={len(target_keys)}, draft tensors={len(draft_keys)}; "
            "both step-specific weight sets are required"
        )
    for fragment in (
        "fc.weight",
        "hidden_norm.weight",
        "decoder.layers.",
        "final_layernorm.weight",
    ):
        if not any(fragment in k for k in draft_keys):
            raise ValueError(f"{path}: trained DFlash checkpoint lacks {fragment}")
    ranges = 0
    for storage in metadata.storage_data.values():
        relative = Path(storage.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe DCP storage path")
        shard = iteration / relative
        if not shard.is_file() or shard.stat().st_size < storage.offset + storage.length:
            raise ValueError(f"Missing or truncated DCP shard: {shard}")
        ranges += 1
    if not ranges:
        raise ValueError(f"Checkpoint has no DCP storage ranges: {iteration}")
    from .config import read_config

    checkpoint_config = read_config(path / "config.yaml")
    for name in ("policy.model_name", "policy.draft.gamma", "policy.draft.mask_token_id"):
        if at(checkpoint_config, name) != at(recipe_record["recipe"], name):
            raise ValueError(f"Checkpoint configuration disagrees with run: {name}")
    policy = recipe_record["recipe"]["policy"]
    target_path = Path(policy["model_name"])
    draft_path = Path(policy["draft"]["model_name"])
    target, draft = read_json(target_path / "config.json"), read_json(draft_path / "config.json")
    if target.get("model_type") != "qwen2":
        raise ValueError("This Qwen2.5 experiment requires the recorded Qwen2 target family")
    for name in ("hidden_size", "vocab_size", "num_attention_heads", "num_key_value_heads"):
        if target.get(name) != draft.get(name):
            raise ValueError(f"Target/draft architecture mismatch: {name}")
    d = draft["dflash_config"]
    if (
        draft.get("model_type") != "qwen3"
        or draft.get("attention_bias")
        or draft.get("mlp_bias")
        or d.get("causal")
        or d.get("use_swa")
        or draft.get("sliding_window")
    ):
        raise ValueError("Unsupported recorded DFlash block attention architecture")
    target_hd = target.get("head_dim") or target["hidden_size"] // target["num_attention_heads"]
    draft_hd = draft.get("head_dim") or draft["hidden_size"] // draft["num_attention_heads"]
    if (
        target_hd != draft_hd
        or not d.get("target_layer_ids")
        or any(not 0 <= layer < target["num_hidden_layers"] for layer in d["target_layer_ids"])
    ):
        raise ValueError("Incompatible DFlash head dimension or target auxiliary layers")
    if (
        d["block_size"] != policy["draft"]["gamma"] + 1
        or d["mask_token_id"] != policy["draft"]["mask_token_id"]
    ):
        raise ValueError("Recorded block size/MASK disagree with draft architecture")
    from dflash_reflex.model_contract import fingerprint, validate_mask_token

    validate_mask_token(target_path, d["mask_token_id"])
    provenance = read_json(draft_path / "dflash_manifest.json")
    if recipe_record.get("original_target_fingerprint") and recipe_record[
        "original_target_fingerprint"
    ] != fingerprint(target_path):
        raise ValueError("Original run metadata target/tokenizer identity has changed")
    if provenance["target_fingerprint"] != fingerprint(target_path):
        raise ValueError(
            "Original draft architecture/tokenizer provenance differs from the recorded target"
        )
    tokenizer_path = Path(policy["tokenizer"]["name"])
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if file_hash(tokenizer_path / name) != file_hash(target_path / name):
            raise ValueError(f"Recorded tokenizer differs from target: {name}")
    cached = read_json(hash_cache) if hash_cache and Path(hash_cache).is_file() else {}
    if cached.get("snapshot") == snapshot and cached.get("checkpoint_path") == str(path):
        hashes = cached["files_sha256"]
    else:
        hashes = {
            name: file_hash(path / name)
            for name in snapshot
            if name.startswith("policy/weights/") or name in {"training_info.json", "config.yaml"}
        }
    if snapshot != files_snapshot(path):
        raise RuntimeError(f"Checkpoint changed while being inspected: {path}")
    if hash_cache:
        atomic_json(
            hash_cache, dict(snapshot=snapshot, checkpoint_path=str(path), files_sha256=hashes)
        )
    shape_map = {k: list(metadata.state_dict_metadata[k].size) for k in keys}
    return dict(
        checkpoint_step=step,
        checkpoint_path=str(path),
        weights_path=str(weights),
        iteration_path=str(iteration),
        training_info=info,
        target=dict(source=str(iteration), tensor_keys=target_keys),
        draft=dict(source=str(iteration), tensor_keys=draft_keys),
        tensor_shapes=shape_map,
        files_sha256=hashes,
        snapshot=snapshot,
        model_identity=dict(
            target_fingerprint=fingerprint(target_path),
            target_config_sha256=file_hash(target_path / "config.json"),
            draft_config_sha256=file_hash(draft_path / "config.json"),
            block_size=d["block_size"],
            mask_token_id=d["mask_token_id"],
        ),
        source_identity=digest(dict(path=str(path), step=step, hashes=hashes)),
        format="NeMo/Megatron torch_dist",
        megatron_iteration_directory=iteration.name,
        optimizer_restored=False,
    )


def require_coverage(models, iteration):
    """Reject missing tensors BEFORE the loader can keep initialized values."""
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

    available = dist_checkpointing.load_sharded_metadata(str(iteration))
    required = {}

    def walk(value):
        if isinstance(value, ShardedTensor):
            required[value.key] = list(value.global_shape)
        elif isinstance(value, ShardedTensorFactory):
            walk(value.build())
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    for model in models:
        walk(model.sharded_state_dict())
    if not required or not any("draft_model." in name for name in required):
        raise RuntimeError("Frozen loader failed to attach the trained DFlash architecture")
    missing = [
        name
        for name, shape in required.items()
        if name not in available or list(available[name].global_shape) != shape
    ]
    if missing:
        raise ValueError(
            f"Checkpoint does not cover every constructed target/draft tensor: {missing[:20]}"
        )
    return dict(required_tensors=len(required), required_shapes=required)


def tensor_hash(value):
    import hashlib

    import torch

    h = hashlib.sha256()
    flat = value.detach().reshape(-1)
    for start in range(0, flat.numel(), 4 * 1024 * 1024):
        h.update(
            flat[start : start + 4 * 1024 * 1024]
            .contiguous()
            .view(torch.uint8)
            .cpu()
            .numpy()
            .tobytes()
        )
    return h.hexdigest()


def save_tensor_stream(destination, weights):
    """Bounded shards, no second full model copy and no overwrite of source."""
    from safetensors.torch import save_file

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    pending, size, count, index, parameters = {}, 0, 0, {}, {}

    def flush():
        nonlocal pending, size, count
        if not pending:
            return
        count += 1
        filename = f"model-{count:05d}.safetensors"
        save_file(pending, str(destination / filename), metadata={"format": "pt"})
        index.update({key: filename for key in pending})
        pending, size = {}, 0

    for name, tensor in weights:
        if name in parameters:
            raise ValueError(f"Duplicate exported tensor: {name}")
        tensor = tensor.detach().cpu().contiguous().clone()
        nbytes = tensor.numel() * tensor.element_size()
        if size + nbytes > 256 * 1024 * 1024:
            flush()
        parameters[name] = dict(
            sha256=tensor_hash(tensor), shape=list(tensor.shape), dtype=str(tensor.dtype)
        )
        pending[name], size = tensor, size + nbytes
    flush()
    if not parameters:
        raise ValueError(f"No exported weights for {destination}")
    atomic_json(
        destination / "model.safetensors.index.json",
        dict(
            metadata={
                "total_size": sum((destination / f).stat().st_size for f in set(index.values()))
            },
            weight_map=index,
        ),
    )
    return parameters


def validate_export(cache, inspected):
    cache = Path(cache)
    marker = cache / "export_manifest.json"
    if not marker.is_file():
        return False
    saved = read_json(marker)
    if saved["source_identity"] != inspected["source_identity"]:
        raise ValueError(f"Export cache belongs to another checkpoint: {cache}")
    if (
        saved["target_source"] != inspected["iteration_path"]
        or saved["draft_source"] != inspected["iteration_path"]
    ):
        raise ValueError("Export cache mixed target and draft checkpoint provenance")
    verify_files(cache, saved["files_sha256"])
    return saved


def export_checkpoint(record, inspected, cache):
    """Single-rank GPU worker: exact restore, freeze, export, then exit."""
    cache = Path(cache)
    if validate_export(cache, inspected):
        return
    if cache.exists():
        raise FileExistsError(f"Incomplete export cache: {cache}; use a separate analysis cache")
    from unittest.mock import patch

    import torch
    from megatron.bridge import AutoBridge
    from nemo_rl.models.megatron import setup as setup_module
    from nemo_rl.models.megatron.draft import export_draft_weights_to_hf
    from nemo_rl.models.megatron.draft.utils import draft_model_detached

    policy = copy.deepcopy(record["recipe"]["policy"])
    if policy.get("megatron_cfg", {}).get("peft", {}).get("enabled"):
        raise ValueError("PEFT export is unsupported for this native full-weight experiment")
    iteration = inspected["iteration_path"]
    if not (Path(iteration) / "run_config.yaml").is_file():
        raise FileNotFoundError(
            f"Megatron architecture metadata missing: {iteration}/run_config.yaml"
        )
    policy["pretrained_checkpoint"] = dict(format="megatron_bridge", path=iteration)
    policy["megatron_cfg"]["checkpoint"] = dict(async_save=False)
    policy["megatron_cfg"]["activation_checkpointing"] = False
    policy["megatron_cfg"]["gradient_accumulation_fusion"] = False
    ddp = policy["megatron_cfg"]["distributed_data_parallel_config"]
    ddp.update(overlap_param_gather=False, overlap_grad_reduce=False)
    draft_source = Path(policy["draft"]["model_name"])
    draft_hf = read_json(draft_source / "config.json")
    target_source = Path(policy["model_name"])
    from dflash_reflex.model_contract import validate_mask_token

    validate_mask_token(target_source, policy["draft"]["mask_token_id"])
    if draft_hf["dflash_config"]["mask_token_id"] != policy["draft"]["mask_token_id"]:
        raise ValueError("Recorded draft architecture MASK disagrees with source run")
    coverage = {}

    def strict_hook(state, models):
        coverage.update(require_coverage(models, iteration))

    temporary = cache.with_name(cache.name + f".building-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        setup_module.setup_distributed(policy)
        runtime = setup_module.validate_and_set_config(
            policy, 0, str(target_source), iteration, iteration, None
        )
        runtime.megatron_cfg.validate()
        runtime.megatron_cfg.checkpoint.load_optim = False
        runtime.megatron_cfg.checkpoint.load_rng = False
        runtime.megatron_cfg.checkpoint.async_save = False
        runtime.megatron_cfg.checkpoint.save = None
        runtime.megatron_cfg.model.hidden_dropout = 0.0
        runtime.megatron_cfg.model.attention_dropout = 0.0
        # Read the original draft CONFIG, never its pretrained WEIGHTS. Every
        # attached tensor must be covered by this exact resume checkpoint.
        with patch(
            "nemo_rl.models.megatron.draft.utils.load_hf_weights_to_block_draft",
            return_value=([], []),
        ):
            loaded = setup_module.setup_model_and_optimizer(
                policy,
                runtime.megatron_cfg,
                load_optimizer=False,
                pre_load_checkpoint_hook=strict_hook,
            )
        if (
            loaded.optimizer is not None
            or loaded.scheduler is not None
            or loaded.draft_model is None
            or not coverage.get("required_tensors")
        ):
            raise RuntimeError(
                "Frozen restore unexpectedly constructed an optimizer or lacks the draft"
            )
        model, draft = loaded.model, loaded.draft_model
        model.eval().requires_grad_(False)
        draft.eval().requires_grad_(False)
        before = {name: tensor_hash(value) for name, value in model.named_parameters()}
        bridge = AutoBridge.from_hf_pretrained(str(target_source), trust_remote_code=True)
        with draft_model_detached([model]):
            target_parameters = save_tensor_stream(
                temporary / "target", bridge.export_hf_weights([model], show_progress=True)
            )
        draft_parameters = save_tensor_stream(
            temporary / "draft", export_draft_weights_to_hf(draft)
        )
        after = {name: tensor_hash(value) for name, value in model.named_parameters()}
        if before != after:
            raise RuntimeError("Model parameters changed during frozen export")
        if files_snapshot(inspected["checkpoint_path"]) != inspected["snapshot"]:
            raise RuntimeError("Original checkpoint changed during frozen export")
        for source in target_source.iterdir():
            if (
                source.is_file()
                and source.suffix in {".json", ".model", ".tiktoken"}
                and not source.name.endswith("index.json")
            ):
                shutil.copy2(source, temporary / "target" / source.name)
        # DFlash intentionally shares the exact restored target's embedding
        # and LM head. Its own exported tensors contain only the trained trunk.
        atomic_json(temporary / "draft/config.json", draft_hf)
        manifest = dict(
            checkpoint_step=inspected["checkpoint_step"],
            source_identity=inspected["source_identity"],
            target_source=iteration,
            draft_source=iteration,
            required_coverage=coverage,
            target_parameters=target_parameters,
            draft_parameters=draft_parameters,
            restored_parameters=before,
            frozen_export_parameter_parity=True,
            optimizer_created=False,
            projector_created=False,
            draft_architecture_source=str(draft_source / "config.json"),
            draft_architecture_sha256=file_hash(draft_source / "config.json"),
            versions=read_json(ROOT / "versions.lock"),
        )
        manifest["files_sha256"] = {
            str(p.relative_to(temporary)): file_hash(p) for p in temporary.rglob("*") if p.is_file()
        }
        atomic_json(temporary / "export_manifest.json", manifest)
        os.replace(temporary, cache)
    finally:
        if torch.distributed.is_initialized():
            setup_module.destroy_parallel_state()
            torch.distributed.destroy_process_group()
