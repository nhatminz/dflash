"""Explicit hooks for pinned vLLM 0.25.1. One engine owns a complete cohort.

The NeMo synchronous rollout supplies the cohort id before LLM.generate and
drains feedback after ALL requests complete. No file or RPC I/O occurs in a
verification round. CUDA events order the optional feedback stream.
"""

import os
from contextlib import contextmanager
from pathlib import Path

import torch

from .adapter import Adapter
from .checkpointing import restore_rng, rng_state
from .config import ReflexConfig
from .dispatch import PipelinePolicy
from .rollout_state import CohortState

_runtime = None
_pool = None
# Process-level switches let production baseline bypass hooks in pinned vLLM.
HOOKS_ENABLED = (
    os.environ.get("METHOD", "dflash_reflex") == "dflash_reflex"
    or os.environ.get("METRICS_MODE", "production") == "instrumented"
)
PROFILE_ENABLED = os.environ.get("OPD_PROFILE", "0").lower() in {"1", "true"}


def tensor_version(tensor):
    # Some pinned inference loaders create head tensors under inference_mode.
    return None if tensor.is_inference() else tensor._version


def current():
    if (
        _runtime is not None
        and not _runtime.enabled
        and _runtime.config.metrics_mode == "production"
    ):
        return None
    return _runtime


class Runtime:
    @torch.inference_mode(False)
    def __init__(self, runner, group_id, projector, version, enabled, config):
        self.runner, self.group_id = runner, group_id
        self.version, self.enabled, self.config = version, enabled, config
        self.request_ids = []
        self.remaining = []
        self.discard = None
        self.eos = []
        self.adapter = self.cache = self.proposal_workspace = self.feedback_workspace = None
        self.projector = projector
        self.instrumented = config.metrics_mode == "instrumented"
        self.capture = enabled or self.instrumented
        self.raw = None
        self.raw_requests = {}
        self.proposal_stamp = None
        self.pipeline = "fused"
        self.pipeline_policy = None
        self.pipeline_rounds = dict(fused=0, selective=0, selective_masked=0)
        self.metadata_workspace = None
        self.density_pending = None
        self.density_host = self.density_event = None
        self.raw_order = None
        self.positions = (
            torch.arange(runner.speculative_config.num_speculative_tokens + 1, device=runner.device)
            if self.capture
            else None
        )
        self.request_positions = (
            torch.arange(runner.max_num_reqs, device=runner.device) if enabled else None
        )
        self.feedback_positions = (
            torch.arange(
                runner.max_num_reqs * runner.speculative_config.num_speculative_tokens,
                device=runner.device,
            )
            if enabled
            else None
        )
        self.counters = torch.zeros(
            8, device=runner.device if self.capture else "cpu", dtype=torch.float64
        )
        self.accepted_buffer = (
            torch.empty(runner.max_num_reqs, device=runner.device, dtype=torch.int32)
            if self.capture
            else None
        )
        self.closed = False
        self.profile_events = []
        self.verification_start = None
        self.proposal_calls = 0
        self.verification_calls = 0
        self.feedback_capacity_rows = 0

    def acceptance_buffer(self, batch_size):
        if not 0 <= batch_size <= self.runner.max_num_reqs:
            raise ValueError("DFlash verification exceeds request buffer capacity")
        if self.config.profile:
            self.verification_start = torch.cuda.Event(enable_timing=True)
            self.verification_start.record()
        return self.accepted_buffer[:batch_size].zero_()

    @contextmanager
    def phase(self, label):
        if self.config.profile:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            self.profile_events.append((label, start, end))
        else:
            yield

    def context(self, runner):
        self.request_ids = list(runner.input_batch.req_ids)
        self.remaining = []
        self.eos = []
        for request_id in self.request_ids:
            request = runner.requests[request_id]
            params = request.sampling_params
            self.remaining.append(max(0, params.max_tokens - len(request.output_token_ids)))
            # User-provided stop strings are intentionally rejected at setup.
            stop = set(params.stop_token_ids or [])
            if params.stop:
                raise ValueError("stop strings require token-aligned feedback; use stop_token_ids")
            if not params.ignore_eos and params._eos_token_id is not None:
                stop.add(params._eos_token_id)
            self.eos.append(sorted(stop))
        self.discard = runner.discard_request_mask.gpu[: len(self.request_ids)]
        if self.cache is not None:
            for request_id in list(self.cache.requests):
                if request_id not in runner.requests:
                    self.cache.finish_request(request_id)

    @torch.no_grad()
    def propose(self, model, hidden):
        from . import kernels

        self.proposal_calls += 1
        with self.phase("draft_head_ms"):
            raw = model.compute_logits(hidden)
        width = self.runner.speculative_config.num_speculative_tokens
        batch = hidden.shape[0] // width
        if (
            hidden.shape[0] != batch * width
            or len(self.request_ids) != batch
            or batch > self.runner.max_num_reqs
            or raw.shape[0] != hidden.shape[0]
        ):
            raise RuntimeError("DFlash proposal request/slot layout changed")
        if self.adapter is None:
            head = model.lm_head.weight
            if self.projector.shape != (hidden.shape[-1], self.config.rank):
                raise ValueError("inference projector dimension mismatch")
            self.adapter = Adapter(head, self.projector.to(head.device).float(), self.config)
            capacity = self.runner.max_num_reqs
            topk = min(self.config.topk, raw.shape[-1])
            self.cache = CohortState(
                self.group_id,
                capacity,
                width,
                hidden.shape[-1],
                self.config.rank,
                topk,
                hidden.device,
                hidden.dtype,
            )
            self.proposal_workspace = kernels.ProposalWorkspace(
                capacity * width, raw.shape[-1], self.config.rank, topk, hidden.device
            )
            self.feedback_workspace = kernels.FeedbackWorkspace(
                capacity * width, hidden.shape[-1], self.config.rank, topk, hidden.device
            )
        # Refit/sleep may replace a Parameter instead of updating it in place.
        # Selected teacher-ID logits must use the current shared head as well.
        self.adapter.head = model.lm_head.weight
        self.adapter.wait()
        # Synchronous pinned scheduler verifies before proposing the next block.
        # Retain compute_logits output, without copying a vocabulary matrix or B.
        # This also preserves head bias, dtype rounding and logit processing.
        self.raw = raw
        order = tuple(self.request_ids)
        if self.raw_order != order:
            self.raw_requests = {key: i * width for i, key in enumerate(order)}
            self.raw_order = order
        self.proposal_stamp = (
            self.adapter.revision,
            self.version,
            self.adapter.A._version,
            self.adapter.head.data_ptr(),
            tensor_version(self.adapter.head),
        )
        self.pipeline = self.config.pipeline
        if self.pipeline == "auto":
            if self.pipeline_policy is None:
                self.pipeline_policy = PipelinePolicy(
                    self.adapter.dispatch_cache,
                    gpu=self.adapter.device_name,
                    capability=self.adapter.capability,
                    hidden=hidden.shape[-1],
                    vocab=raw.shape[-1],
                    rank=self.config.rank,
                    topk=min(self.config.topk, raw.shape[-1]),
                    width=width,
                    dtype=hidden.dtype,
                    logits_dtype=raw.dtype,
                    update_stream=self.config.update_stream,
                )
            if self.density_pending is not None and self.density_event.query():
                density = float(self.density_host) / self.density_pending
                prior = self.adapter.prior_feedback_density
                self.adapter.prior_feedback_density = (
                    density if prior is None else 0.75 * prior + 0.25 * density
                )
                self.density_pending = None
            self.pipeline = (
                self.pipeline_policy.select(batch, self.adapter.prior_feedback_density)
                if self.config.update_interval == 1
                else "fused"
            )
        # GEMM changes reduction order; selective uses the fused FP32 reference.
        if self.config.pipeline == "auto" and self.config.dense_implementation == "gemm":
            self.pipeline = "fused"
        if self.pipeline.startswith("selective") and self.config.dense_implementation == "gemm":
            raise ValueError("selective pipeline requires fused correction, not GEMM")
        function = (
            kernels.greedy_propose if self.pipeline.startswith("selective") else kernels.propose
        )
        with self.phase("corrected_argmax_ms"):
            ids, q, norm, u = function(self.adapter, hidden, raw, self.proposal_workspace)
        self.cache.proposal_slots(self.request_ids)
        # Only one proposal batch is outstanding (guarded by raw_requests/stamp).
        # Store in proposal order; verification maps reordered requests to these
        # contiguous rows without GPU scatter or a host-to-device slot vector.
        n = hidden.shape[0]
        self.cache.h.view(-1, hidden.shape[-1])[:n].copy_(hidden)
        self.cache.u.view(-1, self.config.rank)[:n].copy_(u)
        if self.pipeline == "fused":
            self.cache.ids.view(-1, ids.shape[-1])[:n].copy_(ids)
            self.cache.q.view(-1, q.shape[-1])[:n].copy_(q)
            self.cache.norm.view(-1, 2)[:n].copy_(norm)
        # vLLM retains this tensor and may read it on its CPU-copy stream.
        # The top-k workspace is reused by later proposal/selective-feedback
        # kernels. Transfer ownership of the small argmax vector to vLLM;
        # contiguous() alone would still alias the workspace when top-k is 1.
        return ids[:, 0].clone(memory_format=torch.contiguous_format)

    @torch.no_grad()
    def verified(self, p, accepted, metadata, output_ids, *, greedy=False):
        if self.closed:
            raise RuntimeError("feedback after cohort end")
        if not self.capture:
            return
        self.verification_calls += 1
        configured_width = self.runner.speculative_config.num_speculative_tokens
        lengths_cpu = metadata.num_draft_tokens
        # These are host metadata checks, without a CUDA synchronization. They
        # must precede both counter extraction and raw/cache address arithmetic.
        if (
            len(lengths_cpu) != len(self.request_ids)
            or output_ids.ndim != 2
            or output_ids.shape[0] != len(self.request_ids)
            or not 1 <= output_ids.shape[1] <= configured_width + 1
            or accepted.shape != (len(self.request_ids),)
            or any(length < 0 or length > configured_width for length in lengths_cpu)
            or max(lengths_cpu, default=0) >= output_ids.shape[1]
            or sum(lengths_cpu) != metadata.draft_token_ids.numel()
            or p.ndim != 2
            or p.shape[0] != metadata.draft_token_ids.numel()
        ):
            raise ValueError("DFlash verification request/token layout changed")
        if (
            self.enabled
            and p.shape[0]
            and (self.raw is None or p.shape[1] != self.raw.shape[1] or not p.is_contiguous())
        ):
            raise ValueError("DFlash teacher/draft vocabulary or contiguous logits mismatch")
        if self.verification_start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.profile_events.append(("target_verification_ms", self.verification_start, end))
            self.verification_start = None
        if not self.enabled or not p.shape[0]:
            device = output_ids.device
            extract_start = torch.cuda.Event(enable_timing=True) if self.config.profile else None
            if extract_start is not None:
                extract_start.record()
            lengths = torch.tensor(metadata.num_draft_tokens, device=device, dtype=torch.long)
            width = output_ids.shape[1] - 1
            slots = self.positions[: width + 1][None, :]
            limits = torch.tensor(self.remaining, device=device, dtype=torch.long)
            stop_width = max((len(ids) for ids in self.eos), default=0)
            if stop_width:
                stops = torch.tensor(
                    [ids + [-2] * (stop_width - len(ids)) for ids in self.eos], device=device
                )
                at_eos = (output_ids[:, :, None] == stops[:, None, :]).any(-1)
                eos_pos = torch.where(at_eos, slots, width + 1).amin(-1)
                limits = torch.minimum(limits, eos_pos + 1)
            live = ~self.discard
            emitted = (output_ids >= 0) & (slots < limits[:, None]) & live[:, None]
            count = emitted.sum(-1)
            accepted_live = torch.minimum(accepted.long(), limits) * live
            verified = (lengths > 0) & live
            proposed = torch.minimum(lengths, limits) * live
            self.counters[:4].add_(
                torch.stack(
                    (
                        torch.where(verified, count, 0).sum(),
                        verified.sum(),
                        torch.where(verified, accepted_live, 0).sum(),
                        proposed.sum(),
                    )
                ).double()
            )
            if not self.enabled or not p.shape[0]:
                if extract_start is not None:
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    self.profile_events.append(("feedback_extraction_ms", extract_start, end))
                return
        if self.adapter is None:
            raise RuntimeError("verification before a captured DFlash proposal")
        stamp = (
            self.adapter.revision,
            self.version,
            self.adapter.A._version,
            self.adapter.head.data_ptr(),
            tensor_version(self.adapter.head),
        )
        if stamp != self.proposal_stamp:
            raise RuntimeError(
                "proposal distribution changed before verification (A/B/head version)"
            )
        if any(
            length and request not in self.raw_requests
            for request, length in zip(self.request_ids, metadata.num_draft_tokens, strict=True)
        ):
            raise RuntimeError("overlapping proposal batches are unsupported by synchronous Reflex")
        self.cache.verification_slots(self.request_ids, metadata.num_draft_tokens)
        from . import kernels
        from .runtime_metadata import MetadataWorkspace

        if self.metadata_workspace is None:
            self.metadata_workspace = MetadataWorkspace(
                self.runner.max_num_reqs,
                configured_width,
                self.cache.h.shape[-1],
                self.config.rank,
                self.cache.ids.shape[-1],
                output_ids.device,
            )
        with self.phase("feedback_extraction_ms"):
            h, u, di, dq, norm, weights, raw_rows = self.metadata_workspace.extract(
                self, accepted, metadata, output_ids
            )
        self.pipeline_rounds[self.pipeline] += 1
        self.feedback_capacity_rows += p.shape[0]
        adapter = self.adapter
        device = output_ids.device
        if adapter.stream is not None:
            adapter.stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(adapter.stream):
                sums = kernels.feedback(
                    adapter,
                    h,
                    u,
                    di,
                    dq,
                    norm,
                    p,
                    weights,
                    self.feedback_workspace,
                    raw=self.raw,
                    raw_rows=raw_rows,
                    proposal_workspace=self.proposal_workspace,
                    selective=self.pipeline.startswith("selective"),
                    masked_teacher=self.pipeline == "selective_masked",
                    greedy_teacher=greedy,
                )
                if sums is not None:
                    self.counters[4:].add_(sums.double())
                self.sample_density(sums, p.shape[0])
                adapter.ready.record()
            for tensor in (h, u, di, dq, norm, p, weights, self.raw, raw_rows):
                tensor.record_stream(adapter.stream)
            adapter.has_event = True
        else:
            sums = kernels.feedback(
                adapter,
                h,
                u,
                di,
                dq,
                norm,
                p,
                weights,
                self.feedback_workspace,
                raw=self.raw,
                raw_rows=raw_rows,
                proposal_workspace=self.proposal_workspace,
                selective=self.pipeline.startswith("selective"),
                masked_teacher=self.pipeline == "selective_masked",
                greedy_teacher=greedy,
            )
            if sums is not None:
                self.counters[4:].add_(sums.double())
            self.sample_density(sums, p.shape[0])

    def sample_density(self, sums, rows):
        # Sparse asynchronous telemetry, never a blocking GPU scalar read.
        # A missing calibration uses a constant fallback and needs no telemetry.
        if (
            sums is None
            or self.pipeline_policy is None
            or not self.pipeline_policy.entries
            or self.verification_calls % 16
            or self.density_pending is not None
        ):
            return
        if self.density_host is None:
            self.density_host = torch.empty((), dtype=torch.float32, pin_memory=True)
            self.density_event = torch.cuda.Event()
        self.density_host.copy_(sums[3], non_blocking=True)
        self.density_event.record()
        self.density_pending = rows

    @torch.no_grad()
    def finish(self):
        if self.closed:
            raise RuntimeError("cohort drained twice")
        if self.adapter is not None:
            self.adapter.wait()
        self.closed = True
        # Exactly one boundary packet; no logits or B are serialized.
        packet = dict(
            group_id=self.group_id,
            version=self.version,
            counters=self.counters.cpu(),
            grad_sum=None,
            grad_weight=None,
            profile_ms={},
            updates=0,
            inference_rng=rng_state(),
            core_metrics_available=self.capture,
            metrics_mode=self.config.metrics_mode,
            proposal_calls=self.proposal_calls,
            verification_calls=self.verification_calls,
            pipeline_rounds=self.pipeline_rounds,
            selected_pipeline=self.pipeline if self.enabled else None,
            model_runner_module=type(self.runner).__module__,
        )
        if self.adapter is not None:
            a = self.adapter
            dispatch_counts = a.dispatch_counts.cpu().tolist()
            packet.update(
                grad_sum=a.grad_sum.cpu(),
                grad_weight=a.grad_weight.cpu(),
                updates=int(a.update_count),
                sparse_dispatch_count=dispatch_counts[0],
                dense_dispatch_count=dispatch_counts[1],
            )
            a.prior_feedback_density = (
                float(self.counters[7]) / self.feedback_capacity_rows
                if self.feedback_capacity_rows
                else None
            )
            packet.update(
                active_rows=int(a.active_count),
                gradient_rows=float(self.counters[7]),
                gradient_weight=float(a.grad_weight),
                A_parameter_norm=float(a.A.norm()),
            )
            if self.config.diagnostics:
                packet.update(B_norm=float(a.B.norm()))
            self.profile_events.extend(a.profile_events)
        for label, start, end in self.profile_events:
            end.synchronize()
            packet["profile_ms"][label] = packet["profile_ms"].get(label, 0.0) + start.elapsed_time(
                end
            )
        self.raw = None
        return packet


def profiled_forward(label, function, *args, **kwargs):
    runtime = current()
    if runtime is None or not runtime.config.profile:
        return function(*args, **kwargs)
    with runtime.phase(label):
        return function(*args, **kwargs)


def begin_rpc(worker, group_id, projector_path, enabled, config_dict, rng_path=None):
    global _runtime, _pool
    module = type(worker.model_runner).__module__
    if module.startswith("vllm.") and module != "vllm.v1.worker.gpu_model_runner":
        raise RuntimeError(
            f"Unsupported live vLLM runner {module}: Reflex hooks require "
            "vllm.v1.worker.gpu_model_runner (VLLM_USE_V2_MODEL_RUNNER=0). "
            "Restart both baseline and Reflex with the supported V1 runner."
        )
    if enabled and not HOOKS_ENABLED:
        raise RuntimeError("Reflex requested but vLLM hooks were disabled at import time")
    if _runtime is None and _pool is None:
        print(
            f"DFlash runtime: runner={module}; Reflex={enabled}; hooks={HOOKS_ENABLED}", flush=True
        )
    if _runtime is not None and not _runtime.closed:
        raise RuntimeError("overlapping rollout cohorts on one vLLM engine")
    if rng_path and Path(rng_path).is_file():
        restore_rng(torch.load(rng_path, map_location="cpu", weights_only=True))
    state = (
        torch.load(Path(projector_path), map_location="cpu", weights_only=True) if enabled else None
    )
    _runtime = Runtime(
        worker.model_runner,
        group_id,
        state["A"] if state else None,
        state["version"] if state else -1,
        enabled,
        ReflexConfig(**config_dict),
    )
    if enabled and _pool is not None:
        old = _pool
        if (
            old.runner is worker.model_runner
            and old.config == _runtime.config
            and old.adapter is not None
        ):
            _runtime.adapter, _runtime.cache = old.adapter, old.cache
            _runtime.proposal_workspace = old.proposal_workspace
            _runtime.feedback_workspace = old.feedback_workspace
            _runtime.metadata_workspace = old.metadata_workspace
            _runtime.adapter.reset()
            _runtime.adapter.A.copy_(_runtime.projector)
            _runtime.cache.requests.clear()
            _runtime.cache.free = list(reversed(range(_runtime.cache.capacity)))
            _runtime.cache.group_id = str(group_id)
    return _runtime.version


def finish_rpc(worker):
    global _runtime, _pool
    result = _runtime.finish()
    _pool = _runtime
    _runtime = None  # End of ALL responses. B is discarded here, never per request.
    return result
