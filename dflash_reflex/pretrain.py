"""ShareGPT pretraining with NVIDIA PR #3712's actual model and loss."""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from .adapter import initialize_projector
from .checkpointing import atomic_link, atomic_save, restore_rng, rng_state
from .data import ShareGPTCollator, load_sharegpt
from .model_contract import draft_config, fingerprint, validate_checkpoint, validate_mask_token


def make_model(target_config, *, block_size, layers, mask_id, warm_start=None):
    from megatron.core.transformer import TransformerConfig
    from nemo_rl.models.megatron.draft.dflash import DFlashDraftModel
    from nemo_rl.models.megatron.draft.utils import load_hf_weights_to_block_draft

    hf = draft_config(
        target_config, block_size=block_size, num_layers=layers, mask_token_id=mask_id
    )
    if warm_start:
        hf = json.loads((Path(warm_start) / "config.json").read_text())
    config = TransformerConfig(
        num_layers=hf["num_hidden_layers"],
        hidden_size=hf["hidden_size"],
        num_attention_heads=hf["num_attention_heads"],
        num_query_groups=hf["num_key_value_heads"],
        kv_channels=hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"]),
        ffn_hidden_size=hf["intermediate_size"],
        normalization="RMSNorm",
        layernorm_epsilon=hf["rms_norm_eps"],
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        qk_layernorm=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
        apply_rope_fusion=False,
        use_cpu_initialization=False,
    )
    config.rotary_base = hf.get(
        "rope_theta", (hf.get("rope_parameters") or {}).get("rope_theta", 1000000)
    )
    config.vocab_size = config.draft_vocab_size = hf["vocab_size"]
    model = DFlashDraftModel(
        config,
        gamma=block_size - 1,
        mask_token_id=hf["dflash_config"]["mask_token_id"],
        num_aux_hidden_states=len(hf["dflash_config"]["target_layer_ids"]),
        target_hidden_size=target_config["hidden_size"],
    ).cuda()
    if warm_start:
        missing, extra = load_hf_weights_to_block_draft(model, warm_start)
        if [k for k in missing if "_extra_state" not in k] or extra:
            raise ValueError(f"warm-start DFlash keys mismatch: {missing}, {extra}")
    return model, hf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model-key", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--anchors", type=int, default=16)
    p.add_argument("--mask-id", type=int, default=151669)
    p.add_argument("--loss-chunk-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="")
    p.add_argument("--warm-start", default="")
    a = p.parse_args()
    if min(a.batch_size, a.accumulation, a.epochs, a.max_length, a.save_interval, a.anchors) < 1:
        p.error("positive training settings required")
    from megatron.core import parallel_state
    from nemo_rl.algorithms.loss.loss_functions import (
        BlockDraftLossFn,
        resolve_block_draft_slot_weights,
    )
    from nemo_rl.algorithms.loss.utils import compute_block_draft_slot_valid_counts
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.megatron.draft.dflash import sample_block_anchors
    from nemo_rl.models.megatron.draft.utils import export_block_draft_weights_to_hf
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path, data_path = Path(a.model), Path(a.data)
    fingerprint(model_path)  # Fails before any model/CUDA allocation.
    draft_config(
        json.loads((model_path / "config.json").read_text()),
        block_size=a.block_size,
        num_layers=a.layers,
        mask_token_id=a.mask_id,
    )
    validate_mask_token(model_path, a.mask_id)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if a.warm_start:
        validate_checkpoint(model_path, a.warm_start, a.block_size)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    # Supported Qwen tokenizers use unused padded embedding row 151669, not
    # <|fim_pad|> (151662). Validate before allocating target weights on CUDA.
    if a.mask_id in {tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id}:
        raise ValueError("MASK embedding must be distinct from EOS/BOS/PAD")
    if a.mask_id in tokenizer.get_vocab().values():
        raise ValueError("DFlash MASK must be an unused target embedding row")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    from megatron.core.tensor_parallel.random import (
        get_cuda_rng_tracker,
        model_parallel_cuda_manual_seed,
    )

    model_parallel_cuda_manual_seed(a.seed)
    target = (
        AutoModelForCausalLM.from_pretrained(
            a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        )
        .cuda()
        .eval()
    )
    target.requires_grad_(False)
    model, hf = make_model(
        target.config.to_dict(),
        block_size=a.block_size,
        layers=a.layers,
        mask_id=a.mask_id,
        warm_start=a.warm_start or None,
    )
    # BF16 forward + FP32 master weights and Adam moments (no BF16 Adam states).
    parameters = list(model.parameters())
    master = [torch.nn.Parameter(param.detach().float().clone()) for param in parameters]
    optimizer = torch.optim.AdamW(master, lr=a.lr, weight_decay=0.0)
    collator = ShareGPTCollator(tokenizer, a.max_length)
    rows = load_sharegpt(data_path, collator, a.max_samples)
    sampler = DistributedSampler(rows, num_replicas=world, rank=rank, seed=a.seed, shuffle=True)
    loader = DataLoader(
        rows,
        batch_size=a.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=a.workers,
        generator=torch.Generator().manual_seed(a.seed),
        persistent_workers=a.workers > 0,
    )
    steps_per_epoch = math.ceil(len(loader) / a.accumulation)
    total_steps = a.epochs * steps_per_epoch
    if rank == 0:
        print(
            f"DFlash pretrain: samples={len(rows)}, batch_size_per_gpu={a.batch_size}, "
            f"world_size={world}, accumulation={a.accumulation}, "
            f"effective_global_batch={a.batch_size * a.accumulation * world}, "
            f"batches_per_epoch={len(loader)}, optimizer_steps_per_epoch={steps_per_epoch}",
            flush=True,
        )
    warmup = min(500, max(1, int(0.05 * total_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min(1.0, (step + 1) / warmup)
            * 0.5
            * (1 + math.cos(math.pi * min(max(step - warmup, 0) / max(total_steps - warmup, 1), 1)))
        ),
    )
    loss_fn = BlockDraftLossFn(
        vocab_parallel_group=parallel_state.get_tensor_model_parallel_group(),
        slot_weights=resolve_block_draft_slot_weights("exp", a.block_size - 1),
        seq_chunk_size=a.loss_chunk_size,
    )
    root = Path(a.run_dir)
    logs, checkpoints = root / "logs", root / "checkpoints"
    logs.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    step = epoch_start = batch_start = tokens_processed = 0
    previous_elapsed = 0.0
    contract = dict(
        model=fingerprint(a.model),
        world=world,
        epochs=a.epochs,
        accumulation=a.accumulation,
        batch_size=a.batch_size,
        seed=a.seed,
        samples=len(rows),
        block_size=a.block_size,
        max_length=a.max_length,
        anchors=a.anchors,
        mask_id=a.mask_id,
        lr=a.lr,
        loss_chunk_size=a.loss_chunk_size,
        draft_config=hf,
        data=dict(
            path=str(data_path.resolve()),
            size=data_path.stat().st_size,
            mtime_ns=data_path.stat().st_mtime_ns,
        ),
    )
    if a.resume:
        path = checkpoints / "resume/latest.pt" if a.resume == "auto" else Path(a.resume)
        if path.is_dir():
            path = path / "checkpoints/resume/latest.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["contract"] != contract:
            raise ValueError("pretraining resume configuration differs")
        model.load_state_dict(state["draft"])
        for param, value in zip(master, state["master"], strict=True):
            param.data.copy_(value)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, epoch_start, batch_start, tokens_processed, previous_elapsed = (
            state[k] for k in ("step", "epoch", "next_batch", "tokens", "elapsed")
        )
        summary_path = root / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            if all(summary.get(k) == state.get(k) for k in ("step", "epoch", "next_batch")):
                previous_elapsed = max(previous_elapsed, summary["total_pretrain_time_s"])
        restore_rng(state["ranks"][rank]["rng"])
        get_cuda_rng_tracker().set_states(state["ranks"][rank]["cuda_tracker"])
        if rank == 0:
            from .metrics import truncate_jsonl

            truncate_jsonl(logs / "metrics.jsonl", "step", step)
    tapped = {}
    handles = []
    for layer_id in hf["dflash_config"]["target_layer_ids"]:

        def capture(module, args, output, key=layer_id):
            tapped[key] = output[0] if isinstance(output, tuple) else output

        handles.append(target.model.layers[layer_id].register_forward_hook(capture))
    projector = initialize_projector(
        target.get_output_embeddings().weight, rank=int(os.environ.get("OPD_RANK", "8"))
    )
    started = time.perf_counter()
    if a.max_steps and step >= a.max_steps:
        dist.destroy_process_group()
        return

    def save(epoch, next_batch):
        ranks = [None] * world
        dist.all_gather_object(
            ranks, dict(rng=rng_state(), cuda_tracker=get_cuda_rng_tracker().get_states())
        )
        if rank:
            return
        export_dir = checkpoints / f"draft/step{step}"
        export_dir.mkdir(parents=True, exist_ok=True)
        weights = {
            key: value.detach().cpu().contiguous()
            for key, value in export_block_draft_weights_to_hf(model)
        }
        save_file(weights, str(export_dir / "model.safetensors"))
        (export_dir / "config.json").write_text(json.dumps(hf, indent=2) + "\n")
        (export_dir / "dflash_manifest.json").write_text(
            json.dumps(
                dict(
                    target_fingerprint=fingerprint(a.model),
                    target_path=str(model_path.resolve()),
                    block_size=a.block_size,
                    model_key=a.model_key,
                    source="NVIDIA-NeMo/RL#3712",
                    step=step,
                    pretrain_elapsed_s=previous_elapsed + time.perf_counter() - started,
                ),
                indent=2,
            )
            + "\n"
        )
        atomic_save(dict(A=projector.cpu(), version=0), export_dir / "opd_projector.pt")
        atomic_save(
            dict(
                draft=model.state_dict(),
                master=[p.detach().cpu() for p in master],
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                ranks=ranks,
                step=step,
                epoch=epoch,
                next_batch=next_batch,
                tokens=tokens_processed,
                elapsed=previous_elapsed + time.perf_counter() - started,
                contract=contract,
            ),
            checkpoints / "resume/latest.pt",
        )
        atomic_link(export_dir, root.parent / "latest_checkpoint")
        atomic_link(root, root.parent / "latest_run")
        (root / "summary.json").write_text(
            json.dumps(
                dict(
                    step=step,
                    epoch=epoch,
                    next_batch=next_batch,
                    total_pretrain_time_s=previous_elapsed + time.perf_counter() - started,
                    tokens_processed=tokens_processed,
                    latest_checkpoint=str(export_dir),
                    target=str(model_path),
                ),
                indent=2,
            )
            + "\n"
        )

    for epoch in range(epoch_start, a.epochs):
        sampler.set_epoch(epoch)
        bar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"DFlash pretrain epoch {epoch + 1}/{a.epochs}",
            disable=rank != 0,
            dynamic_ncols=True,
            unit="batch",
        )
        window_start = time.perf_counter()
        window_loss = 0.0
        for i, batch in bar:
            if epoch == epoch_start and i < batch_start:
                continue
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            ids, mask = batch["input_ids"], batch["loss_mask"]
            with torch.no_grad():
                output = target.model(
                    input_ids=ids, attention_mask=batch["attention_mask"], use_cache=False
                )
                teacher = target.get_output_embeddings()(output.last_hidden_state)
                taps = (
                    torch.cat([tapped[key] for key in hf["dflash_config"]["target_layer_ids"]], -1)
                    .transpose(0, 1)
                    .contiguous()
                )
                embeds = target.get_input_embeddings()(ids).transpose(0, 1).contiguous()
                anchors, anchor_valid = sample_block_anchors(
                    token_mask=mask,
                    sample_mask=mask.new_ones(mask.shape[0]),
                    input_ids=ids,
                    num_anchors=a.anchors,
                    generation_only=True,
                )
                counts = compute_block_draft_slot_valid_counts(
                    mask,
                    mask.new_ones(mask.shape[0]),
                    anchors,
                    anchor_valid,
                    gamma=a.block_size - 1,
                )
                dist.all_reduce(counts)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student = model(
                    taps=taps,
                    input_embeds=embeds,
                    anchors=anchors,
                    anchor_valid=anchor_valid,
                    lm_head_weight=target.get_output_embeddings().weight.detach(),
                    mask_embedding=target.get_input_embeddings().weight[a.mask_id].detach(),
                )[:, :, 1:]
                data = BatchedDataDict(
                    dict(
                        draft_anchor_positions=anchors,
                        draft_anchor_valid=anchor_valid,
                        token_mask=mask,
                        sample_mask=mask.new_ones(mask.shape[0]),
                    )
                )
                loss, _ = loss_fn(
                    teacher, student, data, counts.new_tensor(mask.shape[0]), counts.sum(), counts
                )
            # Each rank's numerator/global denominator is summed, not averaged.
            window_size = min(a.accumulation, len(loader) - (i // a.accumulation) * a.accumulation)
            (loss / window_size).backward()
            # Accumulate in FP32 rather than repeatedly adding into BF16 .grad.
            with torch.no_grad():
                for param, fp32 in zip(parameters, master, strict=True):
                    if param.grad is not None:
                        if fp32.grad is None:
                            fp32.grad = param.grad.float()
                        else:
                            fp32.grad.add_(param.grad)
                        param.grad = None
            window_loss += float(loss.detach()) / window_size
            token_count = batch["attention_mask"].sum()
            dist.all_reduce(token_count)
            tokens_processed += int(token_count)
            del student, teacher, taps, embeds, output
            tapped.clear()
            if (i + 1) % a.accumulation and i + 1 != len(loader):
                continue
            with torch.no_grad():
                for param, fp32 in zip(parameters, master, strict=True):
                    if fp32.grad is None:
                        fp32.grad = torch.zeros_like(fp32)
                    dist.all_reduce(fp32.grad)
                grad_norm = torch.nn.utils.clip_grad_norm_(master, 1.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("nonfinite DFlash gradient")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                for param, fp32 in zip(parameters, master, strict=True):
                    param.copy_(fp32)
                    param.grad = None
            step += 1
            now = time.perf_counter()
            if rank == 0:
                row = dict(
                    epoch=epoch + 1,
                    step=step,
                    loss=window_loss,
                    lr=optimizer.param_groups[0]["lr"],
                    tokens=tokens_processed,
                    step_time_s=now - window_start,
                    elapsed_time_s=previous_elapsed + now - started,
                )
                with (logs / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                bar.set_postfix(
                    step=step,
                    loss=f"{window_loss:.4f}",
                    lr=f"{row['lr']:.2e}",
                    tokens=tokens_processed,
                    step_s=f"{row['step_time_s']:.2f}",
                    elapsed_s=f"{row['elapsed_time_s']:.1f}",
                )
            window_start, window_loss = now, 0.0
            if step % a.save_interval == 0:
                save(epoch, i + 1)
            if a.max_steps and step >= a.max_steps:
                save(epoch, i + 1)
                bar.close()
                dist.destroy_process_group()
                return
        batch_start = 0
        bar.close()
    save(a.epochs, 0)
    for handle in handles:
        handle.remove()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
