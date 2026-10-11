"""Boundary integration with NVIDIA's synchronous Megatron DFlash GRPO.

Bulk rollout state is unchanged. Small FP32 projector/gradient packets travel
through an atomic shared-storage mailbox ONLY at rollout/optimizer boundaries.
One inference engine is enforced: distributing a cohort across independent
engines would silently turn shared B into per-shard B.
"""

import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from .checkpointing import atomic_save, restore_rng, rng_state
from .config import ReflexConfig
from .metrics import Metrics
from .projector_training import ProjectorTrainer

_metrics = None
_iteration = None
_pending_checkpoint = None


def run_dir():
    return Path(os.environ["DFLASH_RUN_DIR"])


def enabled():
    return os.environ["METHOD"] == "dflash_reflex"


def generate_cohort(llm, prompts, sampling_params, *, use_tqdm):
    from .vllm_runtime import begin_rpc, finish_rpc

    group = uuid.uuid4().hex
    root = run_dir()
    config = (
        ReflexConfig.from_env()
        if enabled()
        else ReflexConfig(profile=os.environ.get("OPD_PROFILE", "0").lower() in {"1", "true"})
    )
    if len(prompts) != int(os.environ["BATCH_SIZE"]) * int(os.environ["RESPONSES_PER_PROMPT"]):
        # Final incomplete DataLoader batches are allowed; NeMo may not split
        # a cohort across several independent generation engines.
        if len(prompts) > int(os.environ["BATCH_SIZE"]) * int(os.environ["RESPONSES_PER_PROMPT"]):
            raise ValueError("unexpected rollout cohort size")
    llm.collective_rpc(
        begin_rpc,
        args=(
            group,
            str(root / "ipc/projector.pt"),
            enabled(),
            asdict(config),
            str(root / "ipc/inference_rng.pt"),
        ),
    )
    start = time.perf_counter()
    try:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    except BaseException:
        llm.collective_rpc(finish_rpc)
        raise
    packets = llm.collective_rpc(finish_rpc)
    if len(packets) != 1:
        raise RuntimeError("shared-B implementation requires exactly one vLLM inference owner")
    packet = packets[0]
    atomic_save(packet["inference_rng"], root / "ipc/inference_rng.pt")
    packet["generation_time_s"] = time.perf_counter() - start
    packet["generated_tokens"] = sum(len(o.outputs[0].token_ids) for o in outputs)
    packet["iteration_id"] = json.loads((root / "ipc/iteration.json").read_text())["id"]
    packet_path = root / "ipc/feedback" / packet["iteration_id"] / (group + ".pt")
    atomic_save(packet, packet_path)
    if enabled():
        pending = root / "ipc/pending"
        pending.mkdir(parents=True, exist_ok=True)
        os.link(packet_path, pending / (group + ".pt"))
    return outputs


def _resume_sidecar(pretrained_path, rank, stem="reflex_rank"):
    if pretrained_path is None:
        return None
    path = Path(pretrained_path)
    for parent in (path, *path.parents):
        candidate = parent / f"{stem}{rank}.pt"
        if candidate.is_file():
            return candidate
    return None


def initialize_trainer(worker, pretrained_path, init_optimizer):
    worker.dflash_optimizer_steps = 0
    if init_optimizer:
        saved_steps = _resume_sidecar(pretrained_path, dist.get_rank(), "dflash_stats_rank")
        if saved_steps is not None:
            worker.dflash_optimizer_steps = torch.load(
                saved_steps, map_location="cpu", weights_only=True
            )["steps"]
        elif os.environ.get("DFLASH_RESUMING") == "1":
            raise FileNotFoundError("resume checkpoint has no successful-optimizer-step sidecar")
        if dist.get_rank() == 0:
            atomic_save(
                dict(steps=worker.dflash_optimizer_steps), run_dir() / "ipc/optimizer_steps.pt"
            )
    if not enabled() or not init_optimizer:
        return None
    config = ReflexConfig.from_env()
    initial = torch.load(
        run_dir() / "ipc/projector_initial.pt", map_location="cuda", weights_only=True
    )
    trainer = ProjectorTrainer(initial["A"], config)
    saved = _resume_sidecar(pretrained_path, dist.get_rank())
    if saved is not None:
        state = torch.load(saved, map_location="cpu", weights_only=False)
        if state["world_size"] != dist.get_world_size():
            raise ValueError("Reflex resume requires the original trainer topology")
        trainer.load_state_dict(state["projector"])
    elif os.environ.get("DFLASH_RESUMING") == "1":
        raise FileNotFoundError(f"missing projector optimizer sidecar for {pretrained_path}")
    if dist.get_rank() == 0:
        trainer.publish(run_dir() / "ipc/projector.pt")
    dist.barrier()
    return trainer


def before_optimizer_step(worker):
    trainer = worker.reflex_trainer
    if trainer is None:
        return
    # Only the inference owner's packet is submitted, on trainer rank zero.
    if dist.get_rank() == 0:
        for path in sorted((run_dir() / "ipc/pending").glob("*.pt")):
            if path.stem not in trainer.consumed:
                packet = torch.load(path, map_location="cpu", weights_only=True)
                trainer.accumulate(packet)
            path.unlink()


def after_optimizer_step(worker, successful):
    if successful:
        worker.dflash_optimizer_steps += 1
    if dist.get_rank() == 0:
        atomic_save(dict(steps=worker.dflash_optimizer_steps), run_dir() / "ipc/optimizer_steps.pt")
    trainer = worker.reflex_trainer
    if trainer is None:
        return
    success = torch.tensor(int(successful), device=trainer.A.device, dtype=torch.int32)
    dist.all_reduce(success, op=dist.ReduceOp.MIN)
    if success.item():
        trainer.step()
        if dist.get_rank() == 0:
            trainer.publish(run_dir() / "ipc/projector.pt")
        dist.barrier()


def save_trainer(worker, weights_path):
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    atomic_save(
        dict(
            steps=worker.dflash_optimizer_steps,
            rng=rng_state(),
            cuda_tracker=get_cuda_rng_tracker().get_states(),
        ),
        Path(weights_path) / f"dflash_stats_rank{dist.get_rank()}.pt",
    )
    if worker.reflex_trainer is not None:
        atomic_save(
            dict(projector=worker.reflex_trainer.state_dict(), world_size=dist.get_world_size()),
            Path(weights_path) / f"reflex_rank{dist.get_rank()}.pt",
        )


def restore_trainer_rng(worker, pretrained_path, init_optimizer):
    if not init_optimizer or os.environ.get("DFLASH_RESUMING") != "1":
        return
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    path = _resume_sidecar(pretrained_path, dist.get_rank(), "dflash_stats_rank")
    state = torch.load(path, map_location="cpu", weights_only=True)
    restore_rng(state["rng"])
    get_cuda_rng_tracker().set_states(state["cuda_tracker"])


def configure_driver(resume_path=None):
    global _metrics
    state = None
    if resume_path is not None:
        path = Path(resume_path) / "dflash_driver.pt"
        state = torch.load(path, weights_only=False)
        restore_rng(state["driver_rng"])
        inference_rng = run_dir() / "ipc/inference_rng.pt"
        if state["inference_rng"] is not None:
            atomic_save(state["inference_rng"], inference_rng)
        else:
            inference_rng.unlink(missing_ok=True)
        for packet in (run_dir() / "ipc/feedback").rglob("*.pt"):
            # Feedback beyond the checkpoint is regenerated, never replayed.
            if packet.stem not in state["packet_ids"]:
                packet.unlink()
        pending = run_dir() / "ipc/pending"
        pending.mkdir(parents=True, exist_ok=True)
        for packet in pending.glob("*.pt"):
            packet.unlink()
        if enabled():
            projector = torch.load(
                Path(resume_path) / "policy/weights/reflex_rank0.pt",
                map_location="cpu",
                weights_only=False,
            )
            consumed = set(projector["projector"]["consumed"])
            for packet in (run_dir() / "ipc/feedback").rglob("*.pt"):
                if packet.stem not in consumed:
                    os.link(packet, pending / packet.name)
    _metrics = Metrics(
        run_dir(),
        os.environ["METHOD"],
        os.environ["DATASET"],
        None if state is None else state["metrics"],
    )


def tracked_iterations(loader, epoch, global_step):
    global _iteration
    from tqdm.auto import tqdm

    bar = tqdm(
        total=len(loader), desc=f"GRPO epoch {epoch + 1}", unit="iteration", dynamic_ncols=True
    )
    current_step = global_step
    try:
        for batch in loader:
            _iteration = dict(
                id=uuid.uuid4().hex, epoch=epoch, start=time.perf_counter(), recorded=False
            )
            root = run_dir() / "ipc"
            root.mkdir(parents=True, exist_ok=True)
            (root / "iteration.json").write_text(json.dumps(dict(id=_iteration["id"])))
            yield batch
            if not _iteration["recorded"]:
                record_iteration(current_step, {}, {})
            current_step = _iteration["step"]
            bar.update()
            row = _iteration["row"]
            train = _iteration["train_metrics"]
            bar.set_postfix(
                loss=f"{train['loss']:.4f}" if "loss" in train else "N/A",
                reward=f"{train['reward']:.3f}" if "reward" in train else "N/A",
                aal=f"{row['iter_aal']:.3f}",
                acc=f"{row['iter_acceptance_rate']:.3f}",
                tok_s=f"{row['iter_generation_tokens_per_s']:.1f}",
                elapsed_s=f"{row['cumulative_wall_time_s']:.1f}",
                step=_iteration.get("step", global_step),
                refresh=_metrics.state.get("iteration", 0)
                % int(os.environ.get("LOG_INTERVAL", "1"))
                == 0,
            )
    finally:
        bar.close()


def record_iteration(step, timing, metrics):
    global _pending_checkpoint
    if _iteration is None or _iteration["recorded"]:
        return
    packets = []
    for path in (run_dir() / "ipc/feedback" / _iteration["id"]).glob("*.pt"):
        packet = torch.load(path, map_location="cpu", weights_only=True)
        if packet["iteration_id"] == _iteration["id"]:
            packets.append(packet)
    if enabled():
        state = torch.load(run_dir() / "ipc/projector.pt", map_location="cpu", weights_only=True)
        metrics = dict(
            metrics,
            opd_A_grad_norm=state.get("grad_norm"),
            opd_A_version=state["version"],
            opd_A_norm=state.get("A_norm"),
        )
    step_file = run_dir() / "ipc/optimizer_steps.pt"
    if step_file.is_file():
        metrics = dict(
            metrics, total_optimizer_steps=torch.load(step_file, weights_only=True)["steps"]
        )
    row = _metrics.record(
        _iteration["epoch"],
        step,
        packets,
        timing,
        wall=time.perf_counter() - _iteration["start"],
        train_metrics=metrics,
    )
    _iteration["row"], _iteration["train_metrics"] = row, metrics
    _iteration["recorded"], _iteration["step"] = True, step
    if _pending_checkpoint is not None:
        checkpointer, path, wait_fn = _pending_checkpoint
        save_driver_state(path)
        checkpointer.begin_finalization(path, wait_fn=wait_fn)
        _pending_checkpoint = None


def save_driver_state(checkpoint_path):
    inference_path = run_dir() / "ipc/inference_rng.pt"
    atomic_save(
        dict(
            metrics=_metrics.state,
            packet_ids=[p.stem for p in (run_dir() / "ipc/feedback").rglob("*.pt")],
            driver_rng=rng_state(),
            inference_rng=torch.load(inference_path, map_location="cpu", weights_only=True)
            if inference_path.is_file()
            else None,
        ),
        Path(checkpoint_path) / "dflash_driver.pt",
    )


def defer_driver_checkpoint(checkpointer, checkpoint_path, wait_fn):
    global _pending_checkpoint
    if _pending_checkpoint is not None:
        raise RuntimeError("previous driver checkpoint was not committed")
    # Keep the temp directory private until final timings/counters are written.
    _pending_checkpoint = (checkpointer, checkpoint_path, wait_fn)


def finalize_driver():
    if _metrics is not None:
        _metrics.finalize()
