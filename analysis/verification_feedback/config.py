"""Recover the actual native DFlash recipe; no training defaults are guessed."""

from __future__ import annotations

import ast
import copy
import os
from dataclasses import dataclass
from pathlib import Path

from .io import digest, file_hash, read_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = Path(
    "/workspace/storage-shared/nlp/minhpn19/dflash/outputs/train/qwen25_3b/"
    "qwen25_3b__train__dflash__simplelr__seed42__20261012T180453Z__1108564"
)


@dataclass
class Config:
    run_dir: Path = DEFAULT_RUN
    output_dir: Path = ROOT / "outputs/analysis/qwen25_3b_simplelr"
    model: str | None = None
    test_data: str = (
        "/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/test.parquet"
    )
    train_data: str | None = None
    checkpoints: tuple = (250, 600)
    seed: int = 2026
    eval_rollouts: int = 20
    prompts_per_rollout: int = 8
    responses_per_prompt: int = 8
    bootstrap_samples: int = 2000
    smoke: bool = False
    resume: bool = False

    @property
    def destination(self):
        return self.output_dir / "smoke" if self.smoke else self.output_dir

    def validate(self):
        if not self.checkpoints or len(set(self.checkpoints)) != len(self.checkpoints):
            raise ValueError("Checkpoints must be distinct actual step numbers")
        if (
            min(
                *self.checkpoints,
                self.eval_rollouts,
                self.prompts_per_rollout,
                self.responses_per_prompt,
                self.bootstrap_samples,
            )
            < 1
        ):
            raise ValueError("Steps, rollout sizes and bootstrap samples must be positive")
        source = self.run_dir.resolve()
        dest = self.destination.resolve()
        if dest == source or source in dest.parents:
            raise ValueError("Analysis outputs must be outside the original training run")


def read_config(path):
    path = Path(path)
    if path.suffix == ".json":
        return read_json(path)
    import yaml

    return yaml.safe_load(path.read_text())


def at(config, dotted):
    value = config
    for name in dotted.split("."):
        if not isinstance(value, dict) or name not in value:
            raise ValueError(f"Training config lacks required setting {dotted}; no guessed default")
        value = value[name]
    return value


def recover_recipe(c):
    candidates = [
        c.run_dir / "run_config.json",
        c.run_dir / "launch_config.json",
        c.run_dir / "resolved.yaml",
        c.run_dir / "run_metadata.json",
    ]
    candidates += [
        c.run_dir / f"checkpoints/resume/step_{step}/config.yaml" for step in c.checkpoints
    ]
    found = [(p, read_config(p)) for p in candidates if p.is_file()]
    # run_config.json is the expanded runtime recipe. Saved checkpoint configs
    # are equally expanded. A minimal launch file is only a last resort.
    expanded = [(p, v) for p, v in found if p.name in {"run_config.json", "config.yaml"}]
    usable = expanded or [(p, v) for p, v in found if isinstance(v, dict) and "policy" in v]
    if not usable:
        raise FileNotFoundError(f"No recorded policy configuration under {c.run_dir}")
    origin, original = usable[0]
    recipe = copy.deepcopy(original)
    required = [
        "policy.model_name",
        "policy.precision",
        "policy.draft.enabled",
        "policy.draft.model_name",
        "policy.draft.gamma",
        "policy.draft.mask_token_id",
        "policy.generation.backend",
        "policy.generation.max_new_tokens",
        "policy.generation.temperature",
        "policy.generation.top_p",
        "policy.generation.top_k",
        "policy.generation.vllm_kwargs.speculative_config",
        "data.max_input_seq_length",
        "data.default.processor",
        "data.train.data_path",
    ]
    for key in required:
        value = at(recipe, key)
        for p, alternative in found:
            try:
                other = at(alternative, key)
            except ValueError:
                continue
            if other != value:
                raise ValueError(f"Recorded configs disagree on {key}: {origin} versus {p}")
    p = recipe["policy"]
    g = p["generation"]
    metadata = next((v for path, v in found if path.name == "run_metadata.json"), {})
    if metadata.get("METHOD", "dflash") != "dflash":
        raise ValueError(
            "Requested analysis checkpoints must belong to the pure DFlash baseline run"
        )
    if Path(p["model_name"]).name.lower() != "qwen2.5-3b-instruct":
        raise ValueError("This experiment is for the recorded Qwen2.5-3B-Instruct target")
    s = g["vllm_kwargs"]["speculative_config"]
    if not p["draft"]["enabled"] or g["backend"] != "vllm" or s.get("method") != "dflash":
        raise ValueError("Source run must contain a native vLLM DFlash target/draft pair")
    if s.get("draft_sample_method") != "greedy":
        raise ValueError(
            "This experiment requires the source run's singleton greedy draft proposals"
        )
    if s.get("num_speculative_tokens") != p["draft"]["gamma"]:
        raise ValueError("Saved draft gamma and speculative block width disagree")
    if g["vllm_kwargs"].get("async_scheduling") is not False:
        raise ValueError(
            "Pinned passive collector supports the recorded synchronous V1 scheduler only"
        )
    if recipe["data"]["default"]["processor"] != "specnaacl":
        raise ValueError(
            "Unsupported recorded prompt processor; refusing to change prompt formatting"
        )
    if g.get("stop_strings"):
        raise ValueError(
            "Stop strings need a separate token-aligned censor audit; none were expected here"
        )
    if p.get("megatron_cfg", {}).get("peft", {}).get("enabled"):
        raise ValueError(
            "This native full-weight exporter does not support PEFT resume checkpoints"
        )
    if c.model and Path(c.model).resolve() != Path(p["model_name"]).resolve():
        raise ValueError(
            "--model must identify the recorded target; relocation needs identical artifacts"
        )
    for name in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
    ):
        if p.get("megatron_cfg", {}).get(name, 1) != 1:
            raise ValueError(
                f"Only this project's single-rank DFlash checkpoints are supported: {name}"
            )
    templates = {}
    source = ROOT / "dflash_reflex/data.py"
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"SYSTEM_PROMPT", "USER_PROMPT"}:
                    templates[target.id] = ast.literal_eval(node.value)
    if len(templates) != 2:
        raise ValueError("Cannot recover the source prompt templates")
    return dict(
        recipe=recipe,
        source=str(origin),
        recorded_files={str(p): file_hash(p) for p, _ in found},
        templates=templates,
        prompt_source_sha256=file_hash(source),
        recipe_sha256=digest(recipe),
        original_target_fingerprint=metadata.get("target_fingerprint"),
    )


def force_baseline_environment():
    os.environ.update(
        METHOD="dflash",
        OPD_ENABLED="0",
        METRICS_MODE="production",
        OPD_PROFILE="0",
        VLLM_USE_V2_MODEL_RUNNER="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
