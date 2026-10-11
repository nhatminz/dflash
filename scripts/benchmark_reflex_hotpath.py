"""Bounded complete OPD benchmark. No transformer, model weights, or training."""

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def benchmark(args):
    import torch

    from dflash_reflex import kernels
    from dflash_reflex.adapter import Adapter, initialize_projector
    from dflash_reflex.config import ReflexConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU estimates are reported as GPU timings")
    torch.manual_seed(42)
    config = json.loads(Path(args.model_config).read_text())
    hidden, vocab = config["hidden_size"], config["vocab_size"]
    spec = importlib.util.spec_from_file_location(
        "frozen_kernels", ROOT / "tests/reference/original_kernels.py"
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    head = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    A = initialize_projector(head, 8)
    device = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    results, dispatch, pipeline, pipeline_density = [], {}, {}, {}

    def measure(fn, setup=None):
        for _ in range(args.warmup):
            if setup:
                setup()
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        events = []
        # Compilation and reset are outside the CUDA event intervals.
        for _ in range(args.repeats):
            if setup:
                setup()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((start, end))
        events[-1][1].synchronize()
        times = sorted(a.elapsed_time(b) for a, b in events)
        return dict(
            median_ms=statistics.median(times),
            p95_ms=times[min(len(times) - 1, math.ceil(0.95 * len(times)) - 1)],
            allocated_bytes=allocated,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            scratch_peak_bytes=torch.cuda.max_memory_allocated() - allocated,
        )

    for requests in args.requests:
        rows = requests * 15
        h = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
        raw = (h @ head.t()).to(getattr(torch, args.logits_dtype)).contiguous()
        p = torch.randn(rows, vocab, device="cuda").softmax(-1)
        adapter = Adapter(
            head, A.clone(), ReflexConfig(update_stream=False, proposal_mode="sparse")
        )
        initial = torch.zeros_like(adapter.B)
        # Repeated active IDs emulate long rollouts without a growing fake density.
        active = max(32, int(vocab * 0.02))
        initial[:active].normal_(std=0.01)

        def setup():
            adapter.reset()
            adapter.B.copy_(initial)
            adapter.active[:active] = 1
            adapter.active_count.fill_(active)
            adapter.has_updated = True
            adapter.host_active_bound = active  # Frozen implementation only.

        oldpw = original.ProposalWorkspace(rows, vocab, 8, 16, "cuda")
        oldfw = original.FeedbackWorkspace(rows, hidden, 8, 16, "cuda")
        pw = kernels.ProposalWorkspace(rows, vocab, 8, 16, "cuda")
        fw = kernels.FeedbackWorkspace(rows, hidden, 8, 16, "cuda")
        setup()
        oldids, oldq, oldnorm, oldu = original.propose(adapter, h, raw, oldpw)
        ids, q, norm, u = kernels.propose(adapter, h, raw, pw)
        torch.testing.assert_close(ids, oldids, rtol=0, atol=0)
        torch.testing.assert_close(q, oldq, rtol=3e-5, atol=1e-7)
        baseline = measure(lambda: raw.argmax(-1))
        proposal_measurements = {}
        for name, fn, work in (
            ("original", original.propose, oldpw),
            ("fused", kernels.propose, pw),
            ("selective", kernels.greedy_propose, pw),
        ):
            setup()
            proposal_measurements[name] = measure(lambda: fn(adapter, h, raw, work))
        for accepted in (0, 2, 7, 15):
            weights = (
                (torch.arange(15, device="cuda")[None, :] <= accepted)
                .expand(requests, -1)
                .flatten()
                .float()
            )
            measured, isolated_feedback = {}, {}
            for name in ("original", "fused", "selective", "selective_masked"):

                def run():
                    if name == "original":
                        di, dq, no, uu = original.propose(adapter, h, raw, oldpw)
                        original.feedback(adapter, h, uu, di, dq, no, p, weights, oldfw)
                    else:
                        fn = (
                            kernels.greedy_propose
                            if name.startswith("selective")
                            else kernels.propose
                        )
                        di, dq, no, uu = fn(adapter, h, raw, pw)
                        kernels.feedback(
                            adapter,
                            h,
                            uu,
                            di,
                            dq,
                            no,
                            p,
                            weights,
                            fw,
                            raw=raw,
                            proposal_workspace=pw,
                            selective=name.startswith("selective"),
                            masked_teacher=name == "selective_masked",
                        )

                measured[name] = measure(run, setup)
                captured = []

                def prepare_feedback():
                    setup()
                    fn = (
                        original.propose
                        if name == "original"
                        else kernels.greedy_propose
                        if name.startswith("selective")
                        else kernels.propose
                    )
                    work = oldpw if name == "original" else pw
                    captured[:] = fn(adapter, h, raw, work)

                def feedback_only():
                    di, dq, no, uu = captured
                    if name == "original":
                        original.feedback(adapter, h, uu, di, dq, no, p, weights, oldfw)
                    else:
                        kernels.feedback(
                            adapter,
                            h,
                            uu,
                            di,
                            dq,
                            no,
                            p,
                            weights,
                            fw,
                            raw=raw,
                            proposal_workspace=pw,
                            selective=name.startswith("selective"),
                            masked_teacher=name == "selective_masked",
                        )

                isolated_feedback[name] = measure(feedback_only, prepare_feedback)
            results.append(
                dict(
                    requests=requests,
                    rows=rows,
                    accepted_prefix=accepted,
                    feedback_rows=requests * min(accepted + 1, 15),
                    baseline_argmax=baseline,
                    proposal=proposal_measurements,
                    complete=measured,
                    isolated_feedback=isolated_feedback,
                    retained_proposal_logits_bytes=raw.numel() * raw.element_size(),
                    feedback_rows_per_s={
                        name: requests * min(accepted + 1, 15) * 1000 / timing["median_ms"]
                        for name, timing in isolated_feedback.items()
                    },
                    speedup=measured["original"]["median_ms"]
                    / min(
                        measured[x]["median_ms"] for x in ("fused", "selective", "selective_masked")
                    ),
                    incremental_ms={
                        x: measured[x]["median_ms"] - baseline["median_ms"] for x in measured
                    },
                    proposal_rows_per_s={
                        x: rows * 1000 / proposal_measurements[x]["median_ms"]
                        for x in proposal_measurements
                    },
                )
            )
        group = [r for r in results if r["requests"] == requests]
        winner = min(
            ("fused", "selective", "selective_masked"),
            key=lambda x: sum(r["complete"][x]["median_ms"] for r in group),
        )
        key = f"{device}|{requests}|15|{hidden}|{vocab}|8|16|{h.dtype}"
        if raw.dtype != h.dtype:
            key += f"|logits={raw.dtype}"
        fused_total = sum(r["complete"]["fused"]["median_ms"] for r in group)
        winning_total = sum(r["complete"][winner]["median_ms"] for r in group)
        if winning_total >= fused_total * 0.95:
            winner = "fused"
        pipeline[key] = winner
        pipeline_density[key] = [
            dict(
                density=r["feedback_rows"] / r["rows"],
                pipeline=min(
                    ("fused", "selective", "selective_masked"),
                    key=lambda x: r["complete"][x]["median_ms"],
                ),
            )
            for r in group
        ]
        repeated_ids = (
            torch.arange(rows * 32, device="cuda", dtype=torch.long).remainder(64).view(rows, 32)
        )
        repeated_g = torch.randn(rows, 32, device="cuda")
        repeated_u = torch.randn(rows, 8, device="cuda")
        denominator = torch.tensor(float(rows), device="cuda")

        def atomic_update():
            kernels._update[(rows,)](
                repeated_ids,
                repeated_g,
                repeated_u,
                denominator,
                adapter.B,
                adapter.active,
                adapter.active_ids,
                adapter.active_count,
                8,
                16,
                0.01,
                32,
                8,
                enable_fp_fusion=False,
            )

        def aggregate_update():
            unique, inverse = torch.unique(repeated_ids.flatten(), sorted=True, return_inverse=True)
            terms = (-0.01 / denominator) * repeated_g[:, :, None] * repeated_u[:, None, :]
            reduced = torch.zeros(unique.numel(), 8, device="cuda")
            reduced.index_add_(0, inverse, terms.reshape(-1, 8))
            adapter.B.index_add_(0, unique, reduced)
            adapter.active[unique] = 1
            adapter.active_count.fill_(unique.numel())
            adapter.active_ids[: unique.numel()].copy_(unique.int())

        adapter.reset()
        atomic_update()
        atomic_b = adapter.B.clone()
        adapter.reset()
        aggregate_update()
        torch.testing.assert_close(adapter.B, atomic_b, rtol=2e-4, atol=1e-7)
        contention = dict(
            requests=requests,
            unique_ids=64,
            atomic=measure(atomic_update, adapter.reset),
            aggregate=measure(aggregate_update, adapter.reset),
        )
        for entry in group:
            entry["repeated_id_update_comparison"] = contention
        for decision, measurement in zip(pipeline_density[key], group, strict=True):
            chosen = decision["pipeline"]
            if (
                measurement["complete"][chosen]["median_ms"]
                >= measurement["complete"]["fused"]["median_ms"] * 0.95
            ):
                decision["pipeline"] = "fused"
        corrections = []
        for density in (0.02, 0.2, 1.0):
            active = max(1, int(vocab * density))
            initial.zero_()
            initial[:active].normal_(std=0.01)
            setup()
            for mode, implementation in (
                ("sparse", "fused"),
                ("dense", "fused"),
                ("dense", "gemm"),
            ):
                adapter.config = replace(
                    adapter.config, proposal_mode=mode, dense_implementation=implementation
                )
                corrections.append(
                    dict(
                        density=density,
                        mode=mode,
                        implementation=implementation,
                        **measure(lambda: kernels.propose(adapter, h, raw, pw)),
                    )
                )
        # GPU threshold selects fused paths; GEMM remains explicit because its
        # reduction order can change near ties. Require a monotone measured win.
        threshold = vocab + 1
        dense_wins = []
        for density in (0.02, 0.2, 1.0):
            dense = next(
                x
                for x in corrections
                if x["density"] == density
                and x["mode"] == "dense"
                and x["implementation"] == "fused"
            )
            sparse = next(
                x for x in corrections if x["density"] == density and x["mode"] == "sparse"
            )
            dense_wins.append(dense["median_ms"] < sparse["median_ms"] * 0.95)
        for i, density in enumerate((0.02, 0.2, 1.0)):
            if all(dense_wins[i:]):
                threshold = int(vocab * density)
                break
        policy_key = f"{device}|{rows}|{hidden}|{vocab}|8|16|{h.dtype}"
        if raw.dtype != h.dtype:
            policy_key += f"|logits={raw.dtype}"
        adapter.dispatch_cache = {"gpu_thresholds": {policy_key: {"threshold_rows": threshold}}}
        adapter.config = replace(adapter.config, proposal_mode="auto", dense_implementation="fused")
        for density in (0.02, 0.2, 1.0):
            active = max(1, int(vocab * density))
            initial.zero_()
            initial[:active].normal_(std=0.01)
            setup()
            corrections.append(
                dict(
                    density=density,
                    mode="GPU auto policy",
                    implementation="fused",
                    **measure(lambda: kernels.propose(adapter, h, raw, pw)),
                )
            )
        dispatch[policy_key] = dict(threshold_rows=threshold, measurements=corrections)
        torch.cuda.empty_cache()
    start = time.perf_counter()
    torch.cuda.synchronize()
    sync = time.perf_counter() - start
    cache = dict(
        schema_version=2,
        gpu=device,
        capability=list(capability),
        block_size=16,
        logits_dtype=args.logits_dtype,
        gpu_thresholds=dispatch,
        pipelines=pipeline,
        pipelines_by_density=pipeline_density,
    )
    if args.cache:
        Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
        Path(args.cache).write_text(json.dumps(cache, indent=2) + "\n")
    return dict(
        kind="complete synthetic OPD hot path; excludes transformer/verification/refit/GRPO",
        gpu=device,
        capability=list(capability),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        model_config=str(args.model_config),
        hidden=hidden,
        vocab=vocab,
        hidden_dtype=str(h.dtype),
        logits_dtype=str(raw.dtype),
        rank=8,
        topk=16,
        block_size=16,
        repeats=args.repeats,
        warmup=args.warmup,
        results=results,
        dispatch=dispatch,
        selected_pipeline=pipeline,
        idle_synchronize_s=sync,
        b200_validated="B200" in device,
        numerical_proposal_parity="exact IDs; q rtol3e-5 atol1e-7",
        memory_scope="Peak allocated includes shared synthetic inputs and all candidate workspaces in this harness; retained logits lifetime reported separately.",
        interpretation="Kernel/OPD throughput only. No claim of rollout or GRPO speedup.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-config", default=str(ROOT / "third_party/reference_qwen3_4b_config.json")
    )
    p.add_argument("--requests", type=int, nargs="+", default=[1, 8])
    p.add_argument("--logits-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--output", default=str(ROOT / "validation/benchmark_before_after.json"))
    p.add_argument("--cache")
    args = p.parse_args()
    if min(args.requests + [args.repeats, args.warmup]) < 1:
        p.error("requests/repeats/warmup must be positive")
    report = benchmark(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("dispatch", "results")}, indent=2)
    )


if __name__ == "__main__":
    main()
