"""Measure actual Runtime metadata + proposal + feedback + update, without training."""

import argparse
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def benchmark(args):
    import torch
    import triton

    import dflash_reflex

    if args.before_directory:
        path = Path(args.before_directory) / "kernels.py"
        spec = importlib.util.spec_from_file_location("dflash_reflex.kernels", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dflash_reflex.kernels = sys.modules["dflash_reflex.kernels"] = module
        path = Path(args.before_directory) / "vllm_runtime.py"
        spec = importlib.util.spec_from_file_location("dflash_reflex.vllm_runtime", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["dflash_reflex.vllm_runtime"] = module
    from dflash_reflex.adapter import initialize_projector
    from dflash_reflex.config import ReflexConfig
    from dflash_reflex.dispatch import PIPELINE_REVISION
    from dflash_reflex.vllm_runtime import Runtime

    torch.manual_seed(951)
    geometry = json.loads(Path(args.model_config).read_text())
    hidden, vocab = geometry["hidden_size"], geometry["vocab_size"]
    head = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    projector = initialize_projector(head)
    width = 15
    results = []
    for requests in args.requests:
        n = requests * width
        h = torch.randn(n, hidden, device="cuda", dtype=torch.bfloat16)
        raw = torch.randn(n, vocab, device="cuda", dtype=getattr(torch, args.logits_dtype))
        p = torch.randn(n, vocab, device="cuda").softmax(-1)
        runner = SimpleNamespace(
            device=torch.device("cuda"),
            max_num_reqs=requests,
            speculative_config=SimpleNamespace(num_speculative_tokens=width),
        )
        model = SimpleNamespace(lm_head=SimpleNamespace(weight=head), compute_logits=lambda _: raw)
        for prefix in args.prefixes:
            accepted = torch.full((requests,), prefix, device="cuda", dtype=torch.int32)
            output = torch.ones(requests, width + 1, device="cuda", dtype=torch.long)
            output[:, prefix + 1 :] = -1
            for stream in args.streams:
                for pipeline in ("fused", "selective", "selective_masked"):
                    rt = Runtime(
                        runner,
                        "benchmark",
                        projector,
                        0,
                        True,
                        ReflexConfig(
                            pipeline=pipeline,
                            update_stream=bool(stream),
                            diagnostics=True,
                        ),
                    )
                    rt.request_ids = [str(i) for i in range(requests)]
                    rt.remaining = [2048] * requests
                    rt.eos = [[] for _ in range(requests)]
                    rt.discard = torch.zeros(requests, device="cuda", dtype=torch.bool)
                    metadata = SimpleNamespace(num_draft_tokens=[width] * requests)

                    def run(rt=rt):
                        metadata.draft_token_ids = rt.propose(model, h)
                        rt.verified(p, accepted, metadata, output)
                        rt.adapter.wait()  # Include the dependency on the next proposal.

                    for _ in range(args.warmup):
                        run()
                    torch.cuda.synchronize()
                    events = []
                    allocated = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    wall_start = time.perf_counter()
                    for _ in range(args.repeats):
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        start.record()
                        run()
                        end.record()
                        events.append((start, end))
                    events[-1][1].synchronize()
                    wall_ms = (time.perf_counter() - wall_start) * 1000 / args.repeats
                    times = sorted(start.elapsed_time(end) for start, end in events)
                    # Profile separately so CUDA events do not bias selection.
                    rt.config = replace(rt.config, profile=True)
                    rt.adapter.config = rt.config
                    for _ in range(3):
                        run()
                    packet = rt.finish()
                    results.append(
                        dict(
                            requests=requests,
                            accepted_prefix=prefix,
                            density=min(prefix + 1, width) / width,
                            pipeline=pipeline,
                            update_stream=bool(stream),
                            median_ms=statistics.median(times),
                            p95_ms=times[min(len(times) - 1, int(len(times) * 0.95))],
                            wall_ms_per_verification=wall_ms,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            scratch_peak_bytes=torch.cuda.max_memory_allocated() - allocated,
                            retained_raw_bytes=raw.numel() * raw.element_size(),
                            phase_ms={
                                key: value / 3 for key, value in packet["profile_ms"].items()
                            },
                            pipeline_rounds=packet.get("pipeline_rounds"),
                            updates=packet["updates"],
                            gradient_weight=float(packet["grad_weight"]),
                            A_gradient_norm=float(packet["grad_sum"].norm()),
                            B_norm=packet.get("B_norm"),
                        )
                    )
                    print(
                        f"batch={requests} prefix={prefix} stream={stream} {pipeline}: "
                        f"{times[len(times) // 2]:.3f} ms; wall={wall_ms:.3f} ms",
                        flush=True,
                    )
                    del rt, packet
    capability = list(torch.cuda.get_device_capability())
    report = dict(
        implementation_revision=PIPELINE_REVISION
        if not args.before_directory
        else "before_snapshot",
        scope="Synthetic Runtime OPD path including Python metadata, CUDA extraction, proposal correction, feedback, A/B updates and stream dependency; no transformer forward, rejection sampler or GRPO.",
        gpu=torch.cuda.get_device_name(),
        capability=capability,
        torch=torch.__version__,
        triton=triton.__version__,
        cuda=torch.version.cuda,
        hidden=hidden,
        vocab=vocab,
        hidden_dtype=str(h.dtype),
        logits_dtype=str(raw.dtype),
        rank=8,
        topk=16,
        block_size=16,
        warmup=args.warmup,
        repeats=args.repeats,
        b200_measured="B200" in torch.cuda.get_device_name(),
        rollout_tokens_per_s=None,
        real_rollout_AAL=None,
        raw_logits_policy="retain original compute_logits output; no recomputation or B copy",
        results=results,
    )
    if args.cache:
        records = [
            dict(
                requests=r["requests"],
                density=r["density"],
                pipeline=r["pipeline"],
                update_stream=r["update_stream"],
                median_ms=r["wall_ms_per_verification"],
            )
            for r in results
        ]
        cache = dict(
            schema_version=3,
            implementation_revision=PIPELINE_REVISION,
            gpu=report["gpu"],
            capability=capability,
            hidden=hidden,
            vocab=vocab,
            hidden_dtype=report["hidden_dtype"],
            logits_dtype=report["logits_dtype"],
            rank=8,
            topk=16,
            block_size=16,
            measurement_scope="runtime_complete",
            measurements=records,
        )
        Path(args.cache).write_text(json.dumps(cache, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config", default=str(ROOT / "third_party/reference_qwen3_4b_config.json")
    )
    parser.add_argument("--requests", nargs="+", type=int, default=[1, 8, 32, 64])
    parser.add_argument("--prefixes", nargs="+", type=int, default=[0, 1, 2, 4, 8, 15])
    parser.add_argument("--streams", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    parser.add_argument("--logits-dtype", choices=["bfloat16", "float32"], default="float32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--before-directory")
    parser.add_argument("--cache")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.requests + [args.warmup, args.repeats]) < 1 or any(
        not 0 <= p <= 15 for p in args.prefixes
    ):
        parser.error("requests/warmup/repeats must be positive; prefixes must be in 0..15")
    if args.before_directory and args.cache:
        parser.error("before snapshots cannot produce current-runtime calibration")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.cache:
        Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    report = benchmark(args)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
