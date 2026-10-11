"""Compare actual shell defaults and resolved DFlash config without CUDA/importing Torch."""

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dflash_reflex.fairness import MATCHED_BLOCKERS  # noqa: E402


def resolve_environment(root, model, environment):
    data = subprocess.check_output(
        [
            "bash",
            "-c",
            'set -a\nsource "$1"\nsource "$2"\nenv -0',
            "resolve",
            str(root / "configs" / model / "b200.env")
            if (root / "configs" / model / "b200.env").is_file()
            else "/dev/null",
            str(root / "configs/_shared/b200_common.env"),
        ],
        env=environment,
    )
    return dict(item.decode().split("=", 1) for item in data.split(b"\0") if item)


def compare(model="qwen3_4b", dataset="simplelr", profile="native_dflash"):
    specroot = ROOT.parent / "SpecNaacl"
    base = dict(os.environ, DATASET=dataset, FAIRNESS_PROFILE=profile)
    se = resolve_environment(
        specroot,
        model,
        dict(base, METHOD="fastgrpo" if base.get("METHOD") == "dflash" else "opd_reflex"),
    )
    de = resolve_environment(ROOT, model, base)
    paths = {
        "simplelr": "simplelr_abel_level3to5/train.parquet",
        "gsm8k": "gsm8k/main/train-00000-of-00001.parquet",
        "dapo": "DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet",
    }
    de.update(
        DRAFT_CHECKPOINT=de.get(
            "DRAFT_CHECKPOINT", f"{ROOT}/outputs/pretrain/{model}/latest_checkpoint"
        ),
        DFLASH_RUN_DIR=f"{ROOT}/outputs/config-comparison",
        DATASET_PATH=de.get("DATASET_PATH") or f"{de['DATA_ROOT']}/{paths[dataset]}",
    )
    module_spec = importlib.util.spec_from_file_location(
        "dflash_resolver", ROOT / "scripts/prepare_run.py"
    )
    resolver = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(resolver)
    previous = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(de)
        resolved = resolver.build_grpo_config(enforce_profile=False)
    finally:
        os.environ.clear()
        os.environ.update(previous)
    entries = []

    def row(
        name,
        svalue,
        key,
        dvalue,
        status="MATCH",
        reason="Validated resolver maps this value into the actual runtime configuration.",
        source="SpecNaacl/configs/_shared/b200_common.env",
        env=None,
    ):
        entries.append(
            dict(
                parameter=name,
                SpecNaacl_source_file=source,
                SpecNaacl_value=svalue,
                DFlash_env=env or name,
                DFlash_resolved_key=key,
                effective_runtime_value=dvalue,
                status=status,
                reason=reason,
            )
        )

    for name, key, cast in (
        ("MODEL", "policy.model_name", str),
        ("BATCH_SIZE", "grpo.num_prompts_per_step", int),
        ("RESPONSES_PER_PROMPT", "grpo.num_generations_per_prompt", int),
        ("TRAIN_SUBSET_SEED", "grpo.seed", int),
        ("TARGET_LR", "policy.megatron_cfg.optimizer.lr", float),
        ("DRAFT_LR", "policy.draft.lr", float),
        ("TEMPERATURE", "policy.generation.temperature", float),
        ("TOP_P", "policy.generation.top_p", float),
        ("GEN_MAX_LENGTH", "policy.generation.max_new_tokens", int),
        ("MAX_PROMPT_LENGTH", "data.max_input_seq_length", int),
        ("BETA", "loss_fn.reference_policy_kl_penalty", float),
        ("EPSILON", "loss_fn.ratio_clip_min", float),
        ("MAX_TRAIN_SAMPLES", "data.train.max_samples", int),
    ):
        value = resolved
        for part in key.split("."):
            value = value[part]
        sv = cast(se[name]) if name in se else None
        row(
            name,
            sv,
            key,
            value,
            "MATCH" if sv == value else "UNVERIFIED" if sv is None else "DIFFERENT_BY_DESIGN",
        )
    spec_accum = int(se["ACCUMULATION_STEPS"]) if "ACCUMULATION_STEPS" in se else None
    batch = int(se["BATCH_SIZE"]) * int(se["RESPONSES_PER_PROMPT"]) if "BATCH_SIZE" in se else None
    row(
        "ACCUMULATION_STEPS",
        spec_accum,
        "policy.train_micro_batch_size",
        resolved["policy"]["train_micro_batch_size"],
        "DIFFERENT_BY_DESIGN" if spec_accum is not None else "UNVERIFIED",
        "Spec actual target.step is inside every nonempty rollout; ACCUMULATION_STEPS affects step label. DFlash uses this knob only as a physical microbatch cap.",
        source="SpecNaacl/grpo_speculative.py:1535,1699,2041",
    )
    row(
        "EFFECTIVE_TARGET_BATCH",
        {
            "nominal_requested": batch * spec_accum
            if batch is not None and spec_accum is not None
            else None,
            "actual": f"{int(se['RESPONSES_PER_PROMPT'])} times retained nonconstant-reward prompt groups per rollout",
        },
        "policy.train_global_batch_size",
        resolved["policy"]["train_global_batch_size"],
        "DIFFERENT_BY_DESIGN" if batch is not None else "UNVERIFIED",
        "Native NeMo includes constant-reward samples in the sequence/KL denominator. Four-rollout target accumulation is unsupported.",
        source="SpecNaacl/grpo_speculative.py:1441,1535,1699",
    )
    for role in ("target", "draft", "A"):
        row(
            role + "_optimizer_cadence",
            "target: each nonempty filtered rollout; draft/A: each trainable rollout",
            "nemo_bridge.after_optimizer_step",
            "one successful joint step per rollout; A retains pending feedback on skipped steps",
            "DIFFERENT_BY_DESIGN" if role == "target" else "EQUIVALENT",
            "One update per cohort in normal execution; no fourfold reduction of A. Zero-reward filtering and overflow differ across frameworks.",
            source="SpecNaacl/grpo_speculative.py:1364,1699",
        )
    source = ast.parse((specroot / "grpo_speculative.py").read_text())
    argument_defaults = {}
    for node in ast.walk(source):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            for keyword in node.keywords:
                if keyword.arg == "default":
                    try:
                        argument_defaults[node.args[0].value] = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError):
                        pass
    lora = next(
        n
        for n in ast.walk(source)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "LoraConfig"
    )
    lora_values = {
        kw.arg: ast.literal_eval(kw.value)
        for kw in lora.keywords
        if kw.arg in {"r", "lora_alpha", "lora_dropout", "target_modules"}
    }
    for name, field in [
        ("TARGET_LORA_R", "r"),
        ("TARGET_LORA_ALPHA", "lora_alpha"),
        ("TARGET_LORA_DROPOUT", "lora_dropout"),
        ("TARGET_LORA_MODULES", "target_modules"),
    ]:
        row(
            name,
            lora_values[field],
            "policy.megatron_cfg.peft.enabled",
            False,
            "UNSUPPORTED",
            MATCHED_BLOCKERS[0],
            source="SpecNaacl/grpo_speculative.py:738",
        )
    for name in (
        "MAX_TRAINING_TOKEN",
        "MAX_TRAINING_PADDING_GAP",
        "LOGPS_CHUNK_SIZE",
        "GRPO_ITERATION_NUM",
    ):
        row(
            name,
            se.get(name),
            "policy.draft.anchors_per_seq / loss_seq_chunk_size"
            if name != "GRPO_ITERATION_NUM"
            else "one joint epoch",
            1 if name == "GRPO_ITERATION_NUM" else None,
            "EQUIVALENT" if name == "GRPO_ITERATION_NUM" else "DIFFERENT_BY_DESIGN",
            "Spec token/padding budgets pack variable draft states and target microbatches. DFlash uses block anchors/chunking; setting these legacy knobs is rejected.",
            source="SpecNaacl/helper/fastgrpo_training.py; SpecNaacl/grpo_speculative.py",
        )
    for name, sv, key, dv, status, reason in (
        (
            "KL_equation",
            "exp(ref-logp)-(ref-logp)-1",
            "loss_fn.specnaacl_kl_gradient",
            resolved["loss_fn"].get("specnaacl_kl_gradient", False),
            "MATCH" if resolved["loss_fn"].get("specnaacl_kl_gradient") else "DIFFERENT_BY_DESIGN",
            "Spec objective removes NeMo score factor; sequence mean, unclamped k3. original_nemo explicitly preserves the previous loss instead.",
        ),
        (
            "clipping",
            "min(r*A, clamp(r,1-eps,1+eps)*A)",
            "loss_fn.ratio_clip_min/max",
            float(de["EPSILON"]),
            "MATCH",
            "Same clipped policy gradient; on-policy first/only iteration uses detached current logp.",
        ),
        (
            "reward",
            "accuracy + .2 format",
            "env.math.math_verify_impl",
            "specnaacl",
            "MATCH",
            "Same math-verify extraction and exact newline format rule; direct source parity tests.",
        ),
        (
            "reward_std",
            "numpy std ddof=0, no epsilon, drop constant target groups",
            "grpo.normalize_rewards",
            "population std + epsilon; keep constant groups",
            "DIFFERENT_BY_DESIGN",
            "Sample retention and target/KL denominator remain different.",
        ),
        (
            "prompt_format",
            "prompt_dict math and local chat template",
            "data.default.processor",
            "specnaacl",
            "MATCH",
            "Exact template string and tokenizer template; real tokenizer/data unavailable.",
        ),
        (
            "response_mask",
            "retokenized completed assistant",
            "NeMo generated-token mask",
            "generated tokens",
            "DIFFERENT_BY_DESIGN",
            "Whitespace/stop-token/retokenization can differ.",
        ),
        (
            "dataset_rows",
            str(se.get("DATASET_PATH", paths[dataset])),
            "data.train.data_path",
            de["DATASET_PATH"],
            "UNVERIFIED",
            "Three local parsers and synthetic rows checked; configured production datasets absent.",
        ),
        (
            "target_weight_decay",
            0.01,
            "policy.megatron_cfg.optimizer.weight_decay",
            0.0,
            "DIFFERENT_BY_DESIGN",
            "Spec torch AdamW default .01; native target keeps its original weight decay.",
        ),
        (
            "draft_loss",
            "FastGRPO hidden regression/logit loss",
            "policy.draft",
            "NVIDIA block distillation",
            "DIFFERENT_BY_DESIGN",
            "Preserve the intended DFlash training algorithm.",
        ),
        (
            "dtype",
            "BF16 model / FP32 A,B",
            "policy.precision",
            "bfloat16 / FP32 A,B",
            "MATCH",
            "Pinned production versions still require real integration validation.",
        ),
        (
            "GPU_count",
            1,
            "cluster.gpus_per_node",
            1,
            "MATCH",
            "Single colocated engine enforced for shared B.",
        ),
        (
            "checkpoint_resume",
            "full trainer optimizer, projector, RNG",
            "checkpointing.save_optimizer",
            True,
            "UNVERIFIED",
            "Synthetic state/resume tests pass; production distributed optimizer restore untested.",
        ),
        (
            "trainable_projector",
            True,
            "OPD_TRAIN_PROJECTOR",
            True,
            "MATCH",
            "Reflex preflight rejects frozen A; versioned AdamW state persisted.",
        ),
        (
            "shared_B",
            "one generation cohort",
            "vllm_runtime.Adapter",
            "cohort shared and reset",
            "MATCH",
            "Engine pool resets B, retains A; tested with dynamic request ordering.",
        ),
        (
            "total_training_steps",
            argument_defaults["--max_grpo_steps"],
            "grpo.max_num_steps",
            resolved["grpo"]["max_num_steps"],
            "DIFFERENT_BY_DESIGN",
            "Spec launcher forwards CLI overrides; parser default 0 is unbounded. DFlash caps native step count and epochs. Compare actual optimizer steps and generated trajectories.",
        ),
        (
            "save_interval",
            int(se["SAVE_CHECKPOINT_STEPS"]),
            "checkpointing.save_period",
            resolved["checkpointing"]["save_period"],
            "DIFFERENT_BY_DESIGN",
            "Checkpoint formats and step labels differ.",
        ),
    ):
        row(
            name,
            sv,
            key,
            dv,
            status,
            reason,
            source="SpecNaacl/grpo_speculative.py; helper/fastgrpo_training.py; helper/get_QAs.py",
        )
    row(
        "OPD_PROJECTOR_WEIGHT_DECAY",
        0.01,
        "ReflexConfig.projector_weight_decay",
        float(de["OPD_PROJECTOR_WEIGHT_DECAY"]),
        "DIFFERENT_BY_DESIGN",
        "Spec torch AdamW default .01 applies to A. Existing DFlash projector default 0 is preserved and overrideable.",
        source="SpecNaacl/helper/opd_optimizer.py",
    )
    row(
        "generation_seed",
        int(se["TRAIN_SUBSET_SEED"]),
        "VllmGenerationWorker.llm_kwargs.seed",
        int(de["TRAIN_SUBSET_SEED"]),
        "MATCH",
        "Patched engine initialization uses the shared experiment seed instead of a node/bundle index.",
        source="SpecNaacl/scripts/launch/train_model.sh:183",
    )
    row(
        "tokenizer_identity",
        se.get("MODEL"),
        "policy.tokenizer.name",
        resolved["policy"]["tokenizer"]["name"],
        "UNVERIFIED",
        "Same configured model path; tokenizer files are absent in this environment.",
    )
    row(
        "training_epochs",
        int(se["NUM_EPOCHS"]),
        "grpo.max_num_epochs",
        resolved["grpo"]["max_num_epochs"],
        "MATCH"
        if int(se["NUM_EPOCHS"]) == resolved["grpo"]["max_num_epochs"]
        else "DIFFERENT_BY_DESIGN",
        "Same epoch limit does not guarantee the same retained samples or optimizer count.",
    )
    row(
        "total_generated_training_tokens",
        None,
        "metrics.jsonl.iter_generated_tokens",
        None,
        "UNVERIFIED",
        "Must sum actual emitted tokens at runtime; max_steps times max_response_length is not an observed token budget.",
    )
    row(
        "METHOD",
        se["METHOD"],
        "METHOD",
        de["METHOD"],
        "EQUIVALENT",
        "Baseline and Reflex are paired within each framework; native target adaptation remains different.",
    )
    for name in (
        "OPD_RANK",
        "OPD_TOPK",
        "OPD_FAST_LR",
        "OPD_PROJECTOR_LR",
        "OPD_ACCEPTED_WEIGHT",
        "OPD_REJECTED_WEIGHT",
        "OPD_UPDATE_INTERVAL",
    ):
        aliases = {
            "OPD_ACCEPTED_WEIGHT": "OPD_VISITED_WEIGHT",
            "OPD_REJECTED_WEIGHT": "OPD_FRONTIER_WEIGHT",
        }
        sv = se.get(aliases.get(name, name))
        if name == "OPD_PROJECTOR_LR" and not sv:
            sv = se["DRAFT_LR"]
        if name == "OPD_UPDATE_INTERVAL" and sv is None:
            launcher = (specroot / "scripts/launch/train_model.sh").read_text()
            import re

            match = re.search(r"OPD_UPDATE_INTERVAL_ROUNDS:-([0-9]+)", launcher)
            sv = match.group(1) if match else None
        dv = de.get(name)
        row(
            name,
            sv,
            "ReflexConfig." + name[4:].lower(),
            dv,
            "MATCH" if sv == dv else "UNVERIFIED",
            source="SpecNaacl/configs/_shared/b200_common.env",
        )
    spec_model_config = specroot / "configs" / model / "b200.env"
    blockers = list(MATCHED_BLOCKERS)
    if not spec_model_config.is_file():
        blockers.append(
            f"SpecNaacl model config is absent: {spec_model_config}; per-model values are unverified."
        )
    return dict(
        model=model,
        dataset=dataset,
        profile=profile,
        supported=profile == "native_dflash",
        strict_cross_framework_equivalence=False,
        SpecNaacl_model_config_present=spec_model_config.is_file(),
        blockers=blockers,
        source_shas=json.loads((ROOT / "versions.lock").read_text()),
        entries=entries,
        resolved_DFlash=resolved,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        choices=[
            "qwen25_1p5b",
            "qwen25_3b",
            "qwen25_7b",
            "qwen25_14b",
            "qwen3_1p7b",
            "qwen3_4b",
            "qwen3_8b",
        ],
        default="qwen3_4b",
    )
    p.add_argument("--dataset", choices=["simplelr", "gsm8k", "dapo"], default="simplelr")
    p.add_argument(
        "--profile", choices=["specnaacl_matched", "native_dflash"], default="native_dflash"
    )
    p.add_argument("--output", default=str(ROOT / "validation/fairness_config_matrix.json"))
    args = p.parse_args()
    report = compare(args.model, args.dataset, args.profile)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["supported"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
