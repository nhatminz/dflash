#!/usr/bin/env python3
"""Frozen DFlash Figure 1 experiment; this CLI has no training mode."""

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.verification_feedback.config import (  # noqa: E402
    DEFAULT_RUN,
    Config,
    force_baseline_environment,
    recover_recipe,
)
from analysis.verification_feedback.io import (  # noqa: E402
    atomic_json,
    digest,
    exclusive_lock,
    file_hash,
    read_json,
)


def save_checkpoint_manifest(path, manifest):
    """Allow adding a genuine stage; never silently replace a recorded stage."""
    path = Path(path)
    if path.is_file():
        prior = read_json(path)
        old = {row["checkpoint_step"]: row for row in prior["checkpoints"]}
        for row in manifest["checkpoints"]:
            step = row["checkpoint_step"]
            if step in old and old[step] != row:
                raise ValueError(f"Previously inspected checkpoint identity changed: step {step}")
            old[step] = row
        manifest["checkpoints"] = [old[key] for key in sorted(old)]
        manifest["exports"] = prior.get("exports", {})
    atomic_json(path, manifest)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("check", "collect", "analyze", "all"), default="all")
    p.add_argument(
        "--run-dir", type=Path, default=Path(os.environ.get("SOURCE_RUN", str(DEFAULT_RUN)))
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get("ANALYSIS_OUTPUT_DIR", str(ROOT / "outputs/analysis/qwen25_3b_simplelr"))
        ),
    )
    p.add_argument("--model", default=None)
    p.add_argument(
        "--test-data",
        default=os.environ.get(
            "TEST_DATASET_PATH",
            "/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/test.parquet",
        ),
    )
    p.add_argument("--train-data", default=os.environ.get("TRAIN_DATASET_PATH"))
    p.add_argument("--checkpoints", default=os.environ.get("CHECKPOINTS", "250,600"))
    for name, variable, default in (
        ("seed", "SEED", 2026),
        ("eval-rollouts", "EVAL_ROLLOUTS", 20),
        ("prompts-per-rollout", "PROMPTS_PER_ROLLOUT", 8),
        ("responses-per-prompt", "RESPONSES_PER_PROMPT", 8),
        ("bootstrap-samples", "BOOTSTRAP_SAMPLES", 2000),
    ):
        p.add_argument("--" + name, type=int, default=int(os.environ.get(variable, default)))
    p.add_argument(
        "--smoke",
        action="store_true",
        help="One small genuine rollout/checkpoint in a separate smoke directory",
    )
    p.add_argument("--resume", action="store_true", help="Verify and skip completed valid rollouts")
    p.add_argument("--worker", choices=("inspect", "export", "collect"), help=argparse.SUPPRESS)
    p.add_argument("--payload", type=Path, help=argparse.SUPPRESS)
    return p


def runtime_identity():
    pins = read_json(ROOT / "versions.lock")
    observed = {}
    for package, expected in (("torch", pins["runtime"]["torch"]), ("vllm", pins["vllm_version"])):
        # MCore and vLLM intentionally live in different interpreters. Inspect
        # only the package belonging to this worker environment.
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        if version.split("+")[0] != expected.split("+")[0]:
            raise ValueError(f"Unaudited {package} runtime: {version}; pinned {expected}")
        observed[package] = version
    return observed


def worker(args):
    payload = read_json(args.payload)
    force_baseline_environment()
    if args.worker == "inspect":
        from analysis.verification_feedback.checkpoints import inspect_checkpoint

        observed = runtime_identity()
        inspected = inspect_checkpoint(
            payload["run_dir"], payload["step"], payload["record"], payload["hash_cache"]
        )
        from megatron.core import dist_checkpointing
        from megatron.core.dist_checkpointing.validation import verify_integrity_manifest

        common = dist_checkpointing.load_common_state_dict(inspected["iteration_path"])
        if not isinstance(common, dict):
            raise ValueError("Native Megatron common state is not a checkpoint dictionary")
        inspected["native_common_metadata"] = dict(
            checkpoint_version=common.get("checkpoint_version"), iteration=common.get("iteration")
        )
        if (Path(inspected["iteration_path"]) / "integrity.json").is_file():
            verify_integrity_manifest(inspected["iteration_path"])
            inspected["native_integrity_manifest_verified"] = True
        inspected["runtime_versions"] = observed
        atomic_json(payload["result_path"], inspected)
    elif args.worker == "export":
        from analysis.verification_feedback.checkpoints import export_checkpoint

        runtime_identity()
        export_checkpoint(payload["record"], payload["inspected"], payload["cache"])
    else:
        from analysis.verification_feedback.collector import collect_checkpoint

        runtime_identity()
        collect_checkpoint(payload)


def interpreter(variable, names):
    if os.environ.get(variable):
        path = Path(os.environ[variable])
        if not path.is_file():
            raise FileNotFoundError(f"Missing {variable}: {path}")
        return str(path)
    for name in names:
        path = ROOT / name / "bin/python"
        if path.is_file():
            return str(path)
    raise FileNotFoundError(
        f"Missing interpreter for {variable}; activate the offline project environment"
    )


def child(python, operation, payload, destination, distributed=False):
    payload_path = Path(destination) / f".work/{operation}_payload.json"
    atomic_json(payload_path, payload)
    cmd = [python]
    if distributed:
        cmd += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1"]
    cmd += [str(Path(__file__).resolve()), "--worker", operation, "--payload", str(payload_path)]
    subprocess.run(cmd, check=True)


def implementation_identity():
    paths = sorted((ROOT / "analysis/verification_feedback").glob("*.py"))
    paths += [Path(__file__).resolve(), ROOT / "versions.lock"]
    paths += [
        ROOT / "third_party/vllm/vllm" / name
        for name in (
            "v1/sample/rejection_sampler.py",
            "v1/spec_decode/llm_base_proposer.py",
            "v1/spec_decode/dflash.py",
            "v1/worker/gpu_model_runner.py",
            "model_executor/models/qwen3_dflash.py",
        )
    ]
    paths += [
        ROOT / "third_party/nemo-rl/nemo_rl" / name
        for name in (
            "models/megatron/setup.py",
            "models/megatron/draft/utils.py",
            "models/megatron/draft/dflash.py",
            "models/generation/vllm/vllm_worker.py",
        )
    ]
    paths += [
        ROOT / "dflash_reflex" / name for name in ("data.py", "nemo_data.py", "model_contract.py")
    ]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def run(args):
    from analysis.verification_feedback.analysis import analyze, write_report

    c = Config(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        model=args.model,
        test_data=args.test_data,
        train_data=args.train_data,
        checkpoints=tuple(int(v.strip()) for v in args.checkpoints.split(",")),
        seed=args.seed,
        eval_rollouts=args.eval_rollouts,
        prompts_per_rollout=args.prompts_per_rollout,
        responses_per_prompt=args.responses_per_prompt,
        bootstrap_samples=args.bootstrap_samples,
        smoke=args.smoke,
        resume=args.resume,
    )
    # Smoke has a distinct plan and output tree; production is never overwritten.
    if c.smoke:
        c.eval_rollouts = 1
        c.prompts_per_rollout = min(c.prompts_per_rollout, 2)
        c.responses_per_prompt = min(c.responses_per_prompt, 2)
    c.validate()
    root = c.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / ".analysis.lock"):
        if args.mode == "analyze":
            analyze(root, c.checkpoints, c.bootstrap_samples, c.seed)
            return
        force_baseline_environment()
        record = recover_recipe(c)
        from analysis.verification_feedback.data import prepare_plan

        # Essential dataset/model/checkpoint checks always run, even when the
        # optional heavy environment qualification has a cached success.
        plan = prepare_plan(c, record)
        runtime_check = os.environ.get("RUNTIME_CHECK", "auto")
        if runtime_check != "skip":
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/runtime_checks.py"), "--mode", runtime_check],
                check=True,
            )
        mcore = interpreter("DFLASH_MCORE_PYTHON", (".venv-local-mcore", ".venv-mcore"))
        from tqdm.auto import tqdm

        records = []
        for step in tqdm(c.checkpoints, desc="Inspect checkpoints (no model/optimizer)"):
            result_path = root / f".work/inspect_step_{step}.json"
            child(
                mcore,
                "inspect",
                dict(
                    run_dir=str(c.run_dir.resolve()),
                    step=step,
                    record=record,
                    hash_cache=str(c.output_dir.resolve() / f".cache/source_hash_step_{step}.json"),
                    result_path=str(result_path),
                ),
                root,
            )
            records.append(read_json(result_path))
        manifest = dict(
            checkpoints=records,
            recorded_configuration=record,
            pinned_versions=read_json(ROOT / "versions.lock"),
            baseline="pure dflash",
            optimizer_steps=0,
        )
        save_checkpoint_manifest(root / "checkpoint_manifest.json", manifest)
        if not (root / "analysis_report.md").is_file():
            write_report(
                root,
                {},
                "Đã kiểm tra metadata, storage ranges, step, dataset và cấu hình. "
                "Chưa khôi phục model trên GPU; chưa có số liệu thực nghiệm.",
            )
        if args.mode == "check":
            print(
                f"Checkpoint checks passed; no model/optimizer initialized. Manifest: {root / 'checkpoint_manifest.json'}"
            )
            return
        vllm = interpreter("DFLASH_VLLM_PYTHON", (".venv-local-vllm", ".venv-vllm"))
        source = implementation_identity()
        from analysis.verification_feedback.checkpoints import validate_export
        from analysis.verification_feedback.collector import validate_rollout

        for inspected in tqdm(records, desc="Frozen checkpoints"):
            step = inspected["checkpoint_step"]
            identity = digest(
                dict(source=inspected["source_identity"], plan=digest(plan), implementation=source)
            )
            folders = [
                root / f"collection/step_{step}/rollout_{r['rollout_id']:04d}"
                for r in plan["rollouts"]
            ]
            if c.resume and all(validate_rollout(folder, identity) for folder in folders):
                print(
                    f"Step {step}: all committed rollouts verified; skipping export/model initialization",
                    flush=True,
                )
                continue
            # Smoke/results stay isolated, but genuine frozen weight exports
            # are reusable by both plans without repeating GPU conversion.
            cache = c.output_dir.resolve() / f".cache/inference_step_{step}"
            with exclusive_lock(cache.parent / f"export_step_{step}.lock"):
                exported = validate_export(cache, inspected)
                if not exported:
                    child(
                        mcore,
                        "export",
                        dict(record=record, inspected=inspected, cache=str(cache)),
                        root,
                        distributed=True,
                    )
                    exported = validate_export(cache, inspected)
            current_manifest = read_json(root / "checkpoint_manifest.json")
            current_manifest.setdefault("exports", {})[str(step)] = dict(
                target_source=exported["target_source"],
                draft_source=exported["draft_source"],
                cache_path=str(cache),
                export_manifest_sha256=file_hash(cache / "export_manifest.json"),
                strict_complete_restore=True,
                parameter_parity=exported["frozen_export_parameter_parity"],
                target_parameters=exported["target_parameters"],
                draft_parameters=exported["draft_parameters"],
            )
            atomic_json(root / "checkpoint_manifest.json", current_manifest)
            child(
                vllm,
                "collect",
                dict(
                    record=record,
                    inspected=inspected,
                    plan=plan,
                    root=str(root),
                    cache=str(cache),
                    smoke=c.smoke,
                    resume=c.resume,
                    implementation_sha256=source,
                ),
                root,
            )
        if args.mode == "all":
            analyze(root, c.checkpoints, c.bootstrap_samples, c.seed)
        else:
            if not (root / "summary.json").is_file():
                write_report(
                    root,
                    {},
                    "Collection đã hoàn tất; chạy --mode analyze để tính thống kê và xuất figures.",
                )


def main():
    args = parser().parse_args()
    if args.worker:
        worker(args)
        return
    try:
        run(args)
    except Exception as error:
        root = args.output_dir / "smoke" if args.smoke else args.output_dir
        from analysis.verification_feedback.analysis import write_report

        if (
            root.resolve() != args.run_dir.resolve()
            and args.run_dir.resolve() not in root.resolve().parents
        ):
            atomic_json(
                root / "analysis_failure.json", dict(error=type(error).__name__, message=str(error))
            )
            if not (root / "summary.json").is_file():
                write_report(
                    root,
                    {},
                    f"Chưa hoàn tất: {type(error).__name__}: {error}. Không khẳng định đã restore GPU hoặc có kết quả.",
                )
        raise


if __name__ == "__main__":
    main()
