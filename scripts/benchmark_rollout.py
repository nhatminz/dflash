"""Explicit, bounded CUDA microbenchmark; optional real vLLM rollout mode.

Micro timings exclude transformer forwards. They cannot demonstrate a GRPO
speedup. A calibration cache is used only for identical GPU and slot shapes.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def key(device, rows, hidden, vocab, rank, topk, dtype):
    return f"{device}|{rows}|{hidden}|{vocab}|{rank}|{topk}|{dtype}"


def micro(a):
    from dataclasses import replace

    import torch

    from dflash_reflex.adapter import Adapter, initialize_projector
    from dflash_reflex.config import ReflexConfig
    from dflash_reflex.kernels import FeedbackWorkspace, ProposalWorkspace, feedback, propose

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(42)
    head = torch.randn(a.vocab, a.hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    h = torch.randn(a.rows, a.hidden, device="cuda", dtype=torch.bfloat16)
    raw = (h @ head.t()).contiguous()
    adapter = Adapter(
        head,
        initialize_projector(head, a.rank),
        ReflexConfig(rank=a.rank, topk=a.topk, update_stream=False),
    )
    work = ProposalWorkspace(a.rows, a.vocab, a.rank, a.topk, "cuda")
    fw = FeedbackWorkspace(a.rows, a.hidden, a.rank, a.topk, "cuda")

    def measure(function, setup=None):
        for _ in range(3):
            if setup:
                setup()
            function()
        if setup:
            events = []
            for _ in range(a.repeats):
                setup()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                function()
                end.record()
                events.append((start, end))
            events[-1][1].synchronize()
            return sum(start.elapsed_time(end) for start, end in events) / a.repeats
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(a.repeats):
            function()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / a.repeats

    results = []
    for density in (0.02, 0.2, 1.0):
        adapter.B.zero_()
        active = max(1, int(a.vocab * density))
        adapter.B[:active].normal_(std=0.01)
        adapter.active.zero_()
        adapter.active[:active] = 1
        adapter.active_count.fill_(active)
        adapter.active_ids[:active] = torch.arange(active, device="cuda", dtype=torch.int32)
        adapter.has_updated = True
        adapter.host_active_bound = active
        for mode, implementation in (("sparse", "fused"), ("dense", "fused"), ("dense", "gemm")):
            adapter.config = replace(
                adapter.config, proposal_mode=mode, dense_implementation=implementation
            )
            ms = measure(lambda: propose(adapter, h, raw, work))
            results.append(
                dict(density=density, mode=mode, implementation=implementation, proposal_ms=ms)
            )
    ids, q, norm, u = propose(adapter, h, raw, work)
    # Feedback timing uses fixed cached proposal inputs and B_0 each repeat;
    # reset is measured separately and is outside the feedback event interval.
    adapter.reset()
    ids, q, norm, u = propose(adapter, h, raw, work)
    teacher = torch.randn(a.rows, a.vocab, device="cuda").softmax(-1)
    weights = torch.ones(a.rows, device="cuda")
    feedback_ms = measure(
        lambda: feedback(adapter, h, u, ids, q, norm, teacher, weights, fw, raw=raw), adapter.reset
    )
    device = torch.cuda.get_device_name()
    winners = []
    for density in (0.02, 0.2, 1.0):
        winners.append(
            min((r for r in results if r["density"] == density), key=lambda r: r["proposal_ms"])
        )
    report = dict(
        kind="CUDA microbenchmark, no transformer forward or GRPO",
        gpu=device,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        rows=a.rows,
        hidden=a.hidden,
        vocab=a.vocab,
        rank=a.rank,
        topk=a.topk,
        repeats=a.repeats,
        proposal=results,
        feedback_ms=feedback_ms,
    )
    if a.cache:
        path = Path(a.cache)
        cache = json.loads(path.read_text()) if path.exists() else {}
        cache[key(device, a.rows, a.hidden, a.vocab, a.rank, a.topk, str(h.dtype))] = winners
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2) + "\n")
    return report


def rollout(a):
    from dataclasses import asdict

    from vllm import LLM, SamplingParams

    from dflash_reflex.adapter import initialize_projector
    from dflash_reflex.checkpointing import atomic_save
    from dflash_reflex.config import ReflexConfig
    from dflash_reflex.model_contract import validate_checkpoint
    from dflash_reflex.vllm_runtime import begin_rpc, finish_rpc

    validate_checkpoint(a.model, a.draft, a.block_size)
    prompts = json.loads(Path(a.prompts).read_text())
    if not prompts or not all(isinstance(p, str) for p in prompts):
        raise ValueError("--prompts must contain a JSON list of formatted prompt strings")
    checkpoint = Path(a.draft) / "opd_projector.pt"
    if a.method == "dflash_reflex" and not checkpoint.is_file():
        from safetensors import safe_open

        for file in Path(a.model).glob("*.safetensors"):
            with safe_open(file, framework="pt") as handle:
                model_config = json.loads((Path(a.model) / "config.json").read_text())
                key = "lm_head.weight"
                if key not in handle.keys() and model_config.get("tie_word_embeddings"):
                    key = "model.embed_tokens.weight"
                if key in handle.keys():
                    atomic_save(
                        dict(
                            A=initialize_projector(handle.get_tensor(key), a.rank),
                            version=0,
                        ),
                        Path(a.output).with_suffix(".projector.pt"),
                    )
                    checkpoint = Path(a.output).with_suffix(".projector.pt")
                    break
        if not checkpoint.is_file():
            raise ValueError("target LM head not found")
    llm = LLM(
        model=a.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_num_seqs=len(prompts),
        gpu_memory_utilization=a.memory,
        async_scheduling=False,
        seed=42,
        speculative_config=dict(
            method="dflash",
            model=a.draft,
            num_speculative_tokens=a.block_size - 1,
            draft_sample_method="greedy",
        ),
    )
    params = SamplingParams(
        n=1, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, seed=42
    )
    llm.collective_rpc(
        begin_rpc,
        args=(
            "benchmark",
            str(checkpoint),
            a.method == "dflash_reflex",
            asdict(ReflexConfig(rank=a.rank, profile=a.profile, metrics_mode=a.metrics_mode)),
        ),
    )
    start = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=True)
    packet = llm.collective_rpc(finish_rpc)[0]
    elapsed = time.perf_counter() - start
    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    accepted, rounds, accepted_draft, proposed = packet["counters"][:4].tolist()
    return dict(
        kind="real pinned vLLM DFlash rollout",
        method=a.method,
        model=a.model,
        draft=a.draft,
        prompts=len(prompts),
        tokens=tokens,
        wall_time_s=elapsed,
        tokens_per_s=tokens / elapsed,
        aal=accepted / rounds if rounds else None,
        acceptance=accepted_draft / proposed if proposed else None,
        profile_ms=packet["profile_ms"],
        opd_A_gradient_norm=float(packet["grad_sum"].norm())
        if packet["grad_sum"] is not None
        else None,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["micro", "rollout"], default="micro")
    p.add_argument("--output", default="outputs/benchmarks/micro.json")
    p.add_argument("--cache")
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--model")
    p.add_argument("--draft")
    p.add_argument("--prompts")
    p.add_argument("--method", choices=["dflash", "dflash_reflex"], default="dflash_reflex")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--memory", type=float, default=0.7)
    p.add_argument("--metrics-mode", choices=["production", "instrumented"], default="production")
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    os.environ["METHOD"] = a.method
    os.environ["METRICS_MODE"] = a.metrics_mode
    os.environ["OPD_PROFILE"] = "1" if a.profile else "0"
    if a.mode == "rollout" and not all((a.model, a.draft, a.prompts)):
        p.error("rollout requires --model, --draft, --prompts")
    if (
        a.mode == "rollout"
        and a.method == "dflash"
        and a.profile
        and a.metrics_mode == "production"
    ):
        p.error("baseline profiling requires --metrics-mode instrumented")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    report = micro(a) if a.mode == "micro" else rollout(a)
    Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
