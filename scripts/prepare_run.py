"""Resolve paths/configuration without importing Torch or touching GPUs."""

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def env(name, cast=str):
    return cast(os.environ[name])


def build_grpo_config(*, enforce_profile=True):
    batch, responses, accum = (
        env("BATCH_SIZE", int),
        env("RESPONSES_PER_PROMPT", int),
        env("ACCUMULATION_STEPS", int),
    )
    if batch < 1 or responses < 2 or accum < 1:
        raise ValueError("positive batch/accumulation and >=2 responses required")
    if (
        min(
            env("TRAIN_MICRO_BATCH_SIZE", int),
            env("MAX_PROMPT_LENGTH", int),
            env("GEN_MAX_LENGTH", int),
            env("MAX_GRPO_STEPS", int),
            env("TRAIN_EPOCHS", int),
            env("SAVE_INTERVAL", int),
            env("DFLASH_ANCHORS_PER_SEQ", int),
            env("DFLASH_LOSS_CHUNK_SIZE", int),
            env("REWARD_WORKERS", int),
        )
        < 1
    ):
        raise ValueError("positive lengths, steps, chunk sizes and worker counts are required")
    if env("TEMPERATURE", float) < 0 or not 0 < env("TOP_P", float) <= 1:
        raise ValueError("TEMPERATURE >= 0 and 0 < TOP_P <= 1 are required")
    if env("MAX_TRAIN_SAMPLES", int) < 0 or not 0 < env("VLLM_MEMORY_UTILIZATION", float) < 1:
        raise ValueError("invalid sample limit or vLLM memory fraction")
    if env("DRAFT_ACCUMULATION_STEPS", int) != 1:
        raise ValueError(
            "NVIDIA's joint optimizer requires DRAFT_ACCUMULATION_STEPS=1; independent draft cadence is unsupported"
        )
    if env("NPROC_PER_NODE", int) != 1:
        raise ValueError(
            "this shared-B integration currently requires one colocated trainer/inference GPU"
        )
    if env("METHOD") not in {"dflash", "dflash_reflex"}:
        raise ValueError("METHOD must be dflash or dflash_reflex")
    if env("DATASET") not in {"simplelr", "gsm8k", "dapo"}:
        raise ValueError("invalid DATASET")
    from dflash_reflex.fairness import validate_profile

    profile = env("FAIRNESS_PROFILE")
    if (
        env("METHOD") == "dflash"
        and env("METRICS_MODE") == "production"
        and os.environ.get("OPD_PROFILE", "0").lower() in {"1", "true"}
    ):
        raise ValueError("baseline profiling requires METRICS_MODE=instrumented")
    if env("METRICS_MODE") not in {"production", "instrumented"}:
        raise ValueError("invalid METRICS_MODE")
    beta, epsilon = env("BETA", float), env("EPSILON", float)
    if not math.isfinite(beta) or beta < 0 or not 0 < epsilon < 1:
        raise ValueError("BETA must be finite/nonnegative; 0 < EPSILON < 1")
    if env("GRPO_ITERATION_NUM", int) != 1:
        raise ValueError("pinned joint DFlash recipe supports GRPO_ITERATION_NUM=1 only")
    lora = (
        env("TARGET_LORA_R", int),
        env("TARGET_LORA_ALPHA", int),
        env("TARGET_LORA_DROPOUT", float),
    )
    if lora != (64, 32, 0.0):
        raise ValueError(
            "TARGET_LORA_* overrides unsupported: native target is full training; matched LoRA is blocked"
        )
    for unsupported in ("MAX_TRAINING_TOKEN", "MAX_TRAINING_PADDING_GAP", "LOGPS_CHUNK_SIZE"):
        if unsupported in os.environ:
            raise ValueError(
                f"{unsupported} has no equivalent DFlash block-loss setting; do not silently remap it"
            )
    if enforce_profile:
        validate_profile(profile)
    block = env("DFLASH_BLOCK_SIZE", int)
    if block < 2:
        raise ValueError("DFLASH_BLOCK_SIZE must be >=2")
    micro = max(1, (batch * responses) // accum)
    # B200-oriented bounded physical microbatch; ACCUMULATION_STEPS fixes the
    # minimum number of microbatches. Large batches split further for memory.
    micro = min(micro, env("TRAIN_MICRO_BATCH_SIZE", int))
    global_batch = batch * responses
    if global_batch % micro:
        raise ValueError("rollout response count must be divisible by TRAIN_MICRO_BATCH_SIZE")
    loss_config = dict(
        reference_policy_kl_penalty=beta,
        ratio_clip_min=epsilon,
        ratio_clip_max=epsilon,
        force_on_policy_ratio=False,
    )
    objective = env("GRPO_OBJECTIVE")
    if objective == "specnaacl_sequence":
        loss_config.update(
            token_level_loss=False,
            force_on_policy_ratio=True,
            reference_policy_kl_type="k3",
            kl_input_clamp_value=None,
            kl_output_clamp_value=None,
            specnaacl_kl_gradient=True,
        )
    elif objective != "original_nemo":
        raise ValueError("GRPO_OBJECTIVE must be specnaacl_sequence or original_nemo")
    return dict(
        defaults=str(ROOT / "third_party/nemo-rl/examples/configs/grpo_math_1B_megatron.yaml"),
        grpo=dict(
            num_prompts_per_step=batch,
            num_generations_per_prompt=responses,
            max_num_epochs=env("TRAIN_EPOCHS", int),
            max_num_steps=env("MAX_GRPO_STEPS", int),
            seed=env("TRAIN_SUBSET_SEED", int),
            val_at_start=False,
            val_at_end=False,
            val_period=0,
            normalize_rewards=True,
            use_leave_one_out_baseline=False,
            use_dynamic_sampling=False,
            async_grpo=dict(enabled=False),
        ),
        loss_fn=loss_config,
        policy=dict(
            model_name=env("MODEL"),
            train_global_batch_size=global_batch,
            train_micro_batch_size=micro,
            logprob_batch_size=1,
            max_total_sequence_length=env("MAX_PROMPT_LENGTH", int)
            + env("GEN_MAX_LENGTH", int)
            + block,
            precision="bfloat16",
            make_sequence_length_divisible_by=1,
            tokenizer=dict(name=env("MODEL"), chat_template="default"),
            sequence_packing=dict(enabled=False),
            dynamic_batching=dict(enabled=False),
            megatron_cfg=dict(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=False,
                activation_checkpointing=True,
                empty_unused_memory_level=1,
                optimizer=dict(
                    lr=env("TARGET_LR", float), min_lr=env("TARGET_LR", float), weight_decay=0.0
                ),
                scheduler=dict(lr_warmup_iters=0, lr_warmup_init=env("TARGET_LR", float)),
            ),
            draft=dict(
                enabled=True,
                speculator_type="dflash",
                model_name=env("DRAFT_CHECKPOINT"),
                gamma=block - 1,
                mask_token_id=env("DFLASH_MASK_TOKEN_ID", int),
                anchors_per_seq=env("DFLASH_ANCHORS_PER_SEQ", int),
                anchor_from_generation_only=True,
                loss_weight=env("DFLASH_LOSS_WEIGHT", float),
                loss_weighting="exp",
                loss_seq_chunk_size=env("DFLASH_LOSS_CHUNK_SIZE", int),
                lr=env("DFLASH_DRAFT_LR", float),
                min_lr=env("DFLASH_DRAFT_LR", float),
                weight_decay=0.0,
            ),
            generation=dict(
                backend="vllm",
                max_new_tokens=env("GEN_MAX_LENGTH", int),
                temperature=env("TEMPERATURE", float),
                top_p=env("TOP_P", float),
                top_k=None,
                colocated=dict(enabled=True),
                vllm_cfg=dict(
                    async_engine=False,
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    expert_parallel_size=1,
                    use_tqdm=False,
                    enable_vllm_metrics_logger=False,
                    gpu_memory_utilization=env("VLLM_MEMORY_UTILIZATION", float),
                    logprobs_mode="raw_logprobs",
                ),
                vllm_kwargs=dict(
                    async_scheduling=False,
                    max_num_seqs=global_batch,
                    speculative_config=dict(
                        method="dflash",
                        model=env("DRAFT_CHECKPOINT"),
                        num_speculative_tokens=block - 1,
                        draft_sample_method="greedy",
                    ),
                ),
            ),
        ),
        data=dict(
            max_input_seq_length=env("MAX_PROMPT_LENGTH", int),
            num_workers=0,
            train=dict(
                dataset_name="dflash_reflex.nemo_data.LocalMathDataset",
                dataset=env("DATASET"),
                data_path=env("DATASET_PATH"),
                model_path=env("MODEL"),
                max_prompt_length=env("MAX_PROMPT_LENGTH", int),
                max_samples=env("MAX_TRAIN_SAMPLES", int),
                seed=env("TRAIN_SUBSET_SEED", int),
            ),
            validation=None,
            default=dict(
                processor="specnaacl", env_name="math", prompt_file=None, system_prompt_file=None
            ),
        ),
        env=dict(math=dict(num_workers=env("REWARD_WORKERS", int), math_verify_impl="specnaacl")),
        logger=dict(
            log_dir=str(Path(env("DFLASH_RUN_DIR")) / "logs/upstream"),
            wandb_enabled=False,
            tensorboard_enabled=False,
            monitor_gpus=False,
        ),
        checkpointing=dict(
            enabled=True,
            checkpoint_dir=str(Path(env("DFLASH_RUN_DIR")) / "checkpoints/resume"),
            save_period=env("SAVE_INTERVAL", int),
            metric_name="train:reward",
            keep_top_k=1,
            save_optimizer=True,
        ),
        cluster=dict(gpus_per_node=1, num_nodes=1),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["pretrain", "train"])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not a.dry_run and a.kind == "train":
        os.environ["DRAFT_CHECKPOINT"] = str(Path(env("DRAFT_CHECKPOINT")).resolve())
    if a.kind == "train":
        config = build_grpo_config()
        from dflash_reflex.config import ReflexConfig

        reflex = asdict(ReflexConfig.from_env()) if env("METHOD") == "dflash_reflex" else None
        if reflex is not None:
            # Profiling is observational and may be changed across sessions.
            reflex.pop("profile")
            reflex.pop("diagnostics")
            reflex.pop("metrics_mode")
            reflex.pop("pipeline")
    else:
        config = {
            k: v
            for k, v in os.environ.items()
            if (k.startswith(("PRETRAIN_", "DFLASH_", "OPD_")) or k == "MODEL")
            and k != "DFLASH_RESUMING"
        }
    if a.dry_run:
        print(json.dumps(config, indent=2))
        return
    if min(env("LOG_INTERVAL", int), env("ROLLOUT_LOG_FLUSH_INTERVAL", int)) < 1:
        raise ValueError("logging intervals must be positive")
    from dflash_reflex.model_contract import draft_config, fingerprint, validate_checkpoint

    fingerprint(env("MODEL"))
    target_config = json.loads((Path(env("MODEL")) / "config.json").read_text())
    draft_config(
        target_config,
        block_size=env("DFLASH_BLOCK_SIZE", int),
        num_layers=env("DFLASH_LAYERS", int),
        mask_token_id=env("DFLASH_MASK_TOKEN_ID", int),
    )
    if not any(Path(env("MODEL")).glob("*.safetensors")):
        raise FileNotFoundError("target directory has no safetensors weights")
    if a.kind == "pretrain" and env("PRETRAIN_WARM_START"):
        validate_checkpoint(env("MODEL"), env("PRETRAIN_WARM_START"), env("DFLASH_BLOCK_SIZE", int))
    source = env("PRETRAIN_DATASET_PATH") if a.kind == "pretrain" else env("DATASET_PATH")
    if not Path(source).is_file():
        raise FileNotFoundError(source)
    runtime_contract = None
    if a.kind == "train":
        runtime_contract = dict(
            method=env("METHOD"),
            model_key=env("MODEL_KEY"),
            reflex=reflex,
            target_fingerprint=fingerprint(env("MODEL")),
            data=dict(
                path=str(Path(source).resolve()),
                size=Path(source).stat().st_size,
                mtime_ns=Path(source).stat().st_mtime_ns,
            ),
        )
    root = Path(env("DFLASH_RUN_DIR"))
    launch_config = root / "launch_config.json"
    if env("DFLASH_RESUMING") == "0" and launch_config.exists():
        raise FileExistsError("run already exists; use RESUME=auto or choose a new RUN_NAME")
    if env("DFLASH_RESUMING") == "1":
        if not launch_config.is_file():
            raise FileNotFoundError(
                f"resume run has no original launch configuration: {launch_config}"
            )
        resume_root = root / "checkpoints/resume"
        if a.kind == "pretrain" and not (resume_root / "latest.pt").is_file():
            raise FileNotFoundError("active pretrain run has no committed resume checkpoint")
        if a.kind == "train" and not any(resume_root.glob("step_*/training_info.json")):
            raise FileNotFoundError("active GRPO run has no committed resume checkpoint")
        saved = json.loads(launch_config.read_text())
        if a.kind == "train" and saved.pop("_dflash_runtime_contract", None) != runtime_contract:
            raise ValueError("resume method, Reflex parameters, target or dataset differs")
        comparison = json.loads(json.dumps(config))
        # Step limits may be extended for a smoke-test continuation. The
        # pretraining schedule is based on full epochs, not this stop limit.
        if a.kind == "pretrain":
            saved.pop("PRETRAIN_MAX_STEPS", None)
            comparison.pop("PRETRAIN_MAX_STEPS", None)
        else:
            saved["grpo"].pop("max_num_steps", None)
            comparison["grpo"].pop("max_num_steps", None)
        if saved != comparison:
            raise ValueError(
                "resume configuration differs from the original run; preserve model/data/optimizer/sampling settings"
            )
    for path in (
        "logs",
        "statistics",
        "checkpoints/target",
        "checkpoints/draft",
        "checkpoints/resume",
        "ipc/feedback",
    ):
        (root / path).mkdir(parents=True, exist_ok=True)
    if a.kind == "train":
        draft = validate_checkpoint(
            env("MODEL"), env("DRAFT_CHECKPOINT"), env("DFLASH_BLOCK_SIZE", int)
        )
        # Respect checkpoint-specific mask IDs; reject a conflicting override.
        if draft["dflash_config"]["mask_token_id"] != env("DFLASH_MASK_TOKEN_ID", int):
            raise ValueError("DFLASH_MASK_TOKEN_ID differs from the draft checkpoint")
        import yaml

        (root / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        if env("METHOD") == "dflash_reflex":
            import torch

            from dflash_reflex.adapter import initialize_projector
            from dflash_reflex.checkpointing import atomic_save

            checkpoint = Path(env("DRAFT_CHECKPOINT")) / "opd_projector.pt"
            if checkpoint.is_file():
                projector = torch.load(checkpoint, weights_only=True)
                if projector["A"].shape[1] != env("OPD_RANK", int):
                    raise ValueError("OPD_RANK differs from the pretrained projector")
            else:
                from safetensors import safe_open

                head = None
                for file in Path(env("MODEL")).glob("*.safetensors"):
                    with safe_open(file, framework="pt") as tensors:
                        key = "lm_head.weight"
                        if key not in tensors.keys() and target_config.get("tie_word_embeddings"):
                            key = "model.embed_tokens.weight"
                        if key in tensors.keys():
                            head = tensors.get_tensor(key)
                            break
                if head is None:
                    raise ValueError(
                        "external draft initialization needs the target's lm_head.weight"
                    )
                projector = dict(A=initialize_projector(head, env("OPD_RANK", int)), version=0)
            atomic_save(projector, root / "ipc/projector_initial.pt")
            if not (root / "ipc/projector.pt").exists():
                atomic_save(projector, root / "ipc/projector.pt")
    (root / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    launch = dict(config)
    if runtime_contract is not None:
        launch["_dflash_runtime_contract"] = runtime_contract
    launch_config.write_text(json.dumps(launch, indent=2) + "\n")
    manifest = Path(env("DRAFT_CHECKPOINT")) / "dflash_manifest.json" if a.kind == "train" else None
    pretrain_time = (
        json.loads(manifest.read_text()).get("pretrain_elapsed_s")
        if manifest and manifest.is_file()
        else None
    )
    (root / "run_metadata.json").write_text(
        json.dumps(
            dict(
                created_utc=datetime.now(timezone.utc).isoformat(),
                python=sys.version,
                source_shas=json.loads((ROOT / "versions.lock").read_text()),
                target_fingerprint=fingerprint(env("MODEL")),
                pretrain_time_s=pretrain_time,
                FAIRNESS_PROFILE=os.environ.get("FAIRNESS_PROFILE"),
                METRICS_MODE=os.environ.get("METRICS_MODE"),
                METHOD=os.environ.get("METHOD"),
                DATASET=os.environ.get("DATASET"),
                MODEL=os.environ.get("MODEL"),
                BETA=os.environ.get("BETA"),
                EPSILON=os.environ.get("EPSILON"),
                GRPO_OBJECTIVE=os.environ.get("GRPO_OBJECTIVE"),
                EFFECTIVE_TARGET_BATCH=config["policy"]["train_global_batch_size"]
                if a.kind == "train"
                else None,
                TARGET_ADAPTATION="full fine-tuning (native_dflash)",
                TARGET_OPTIMIZER_STEP_COUNT=0,
                DRAFT_OPTIMIZER_STEP_COUNT=0,
                OPD_PROJECTOR_LR=os.environ.get("OPD_PROJECTOR_LR"),
                DFLASH_BLOCK_SIZE=os.environ.get("DFLASH_BLOCK_SIZE"),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
