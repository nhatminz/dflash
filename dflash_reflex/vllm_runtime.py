"""Explicit hooks for pinned vLLM 0.25.1. One engine owns a complete cohort.

The NeMo synchronous rollout supplies the cohort id before LLM.generate and
drains feedback after ALL requests complete. No file or RPC I/O occurs in a
verification round. CUDA events order the optional feedback stream.
"""

from contextlib import contextmanager
from pathlib import Path

import torch

from .adapter import Adapter
from .checkpointing import restore_rng, rng_state
from .config import ReflexConfig
from .rollout_state import CohortState

_runtime = None
_pool = None


def current():
    return _runtime


class Runtime:
    def __init__(self, runner, group_id, projector, version, enabled, config):
        self.runner, self.group_id = runner, group_id
        self.version, self.enabled, self.config = version, enabled, config
        self.request_ids = []
        self.remaining = []
        self.discard = None
        self.eos = []
        self.adapter = self.cache = self.proposal_workspace = self.feedback_workspace = None
        self.projector = projector
        self.counters = torch.zeros(8, device=runner.device, dtype=torch.float64)
        self.accepted_buffer = torch.empty(
            runner.max_num_reqs, device=runner.device, dtype=torch.int32
        )
        self.closed = False
        self.profile_events = []
        self.verification_start = None

    def acceptance_buffer(self, batch_size):
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

        with self.phase("draft_head_ms"):
            raw = model.compute_logits(hidden)
        width = self.runner.speculative_config.num_speculative_tokens
        batch = hidden.shape[0] // width
        if len(self.request_ids) != batch:
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
        ids, q, norm, u = kernels.propose(self.adapter, hidden, raw, self.proposal_workspace)
        slots = self.cache.proposal_slots(self.request_ids)
        row = torch.tensor(slots, device=hidden.device)
        # The caches retain MASK states only. The anchor/bonus has no OPD row.
        self.cache.h[row] = hidden.view(batch, width, -1)
        self.cache.u[row] = u.view(batch, width, -1)
        self.cache.ids[row] = ids.view(batch, width, -1)
        self.cache.q[row] = q.view(batch, width, -1)
        self.cache.norm[row] = norm.view(batch, width, 2)
        return ids[:, 0]  # Actual proposal q is one-hot; verifier receives None.

    @torch.no_grad()
    def verified(self, p, accepted, metadata, output_ids, *, greedy=False):
        if self.closed:
            raise RuntimeError("feedback after cohort end")
        if self.verification_start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.profile_events.append(("target_verification_ms", self.verification_start, end))
            self.verification_start = None
        device = output_ids.device
        extract_start = torch.cuda.Event(enable_timing=True) if self.config.profile else None
        if extract_start is not None:
            extract_start.record()
        lengths = torch.tensor(metadata.num_draft_tokens, device=device, dtype=torch.long)
        width = output_ids.shape[1] - 1
        slots = torch.arange(width + 1, device=device)[None, :]
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
        indices = self.cache.verification_slots(self.request_ids, metadata.num_draft_tokens)
        # One GPU vector covers all variable-length requests; no token loop.
        n = metadata.draft_token_ids.numel()
        request_rows = torch.repeat_interleave(
            torch.arange(len(indices), device=device), lengths, output_size=n
        )
        starts = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)[:-1]))
        position = torch.arange(n, device=device) - starts[request_rows]
        cache_row = torch.tensor(indices, device=device)[request_rows]
        accepted_slot = position < accepted[request_rows]
        selected = (
            (position <= accepted[request_rows])
            & (position < limits[request_rows])
            & live[request_rows]
            & (metadata.draft_token_ids >= 0)
        )
        weights = torch.where(
            selected,
            torch.where(accepted_slot, self.config.accepted_weight, self.config.rejected_weight),
            0.0,
        )
        h = self.cache.h[cache_row, position].contiguous()
        u = self.cache.u[cache_row, position].contiguous()
        di = self.cache.ids[cache_row, position].contiguous()
        dq = self.cache.q[cache_row, position].contiguous()
        norm = self.cache.norm[cache_row, position].contiguous()
        if greedy:
            # All-greedy teacher is the constrained verifier argmax delta.
            teacher = torch.zeros_like(p, dtype=torch.float32)
            teacher.scatter_(1, p.argmax(-1, keepdim=True), 1.0)
            p = teacher
        from . import kernels

        if extract_start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.profile_events.append(("feedback_extraction_ms", extract_start, end))
        adapter = self.adapter
        if adapter.stream is not None:
            adapter.stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(adapter.stream):
                sums = kernels.feedback(
                    adapter, h, u, di, dq, norm, p, weights, self.feedback_workspace
                )
                if sums is not None:
                    self.counters[4:].add_(sums.double())
                adapter.ready.record()
            for tensor in (h, u, di, dq, norm, p, weights):
                tensor.record_stream(adapter.stream)
            adapter.has_event = True
        else:
            sums = kernels.feedback(
                adapter, h, u, di, dq, norm, p, weights, self.feedback_workspace
            )
            if sums is not None:
                self.counters[4:].add_(sums.double())

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
        )
        if self.adapter is not None:
            a = self.adapter
            packet.update(
                grad_sum=a.grad_sum.cpu(),
                grad_weight=a.grad_weight.cpu(),
                updates=a.round // self.config.update_interval,
                sparse_dispatch_count=a.sparse_dispatch_count,
                dense_dispatch_count=a.dense_dispatch_count,
            )
            if self.config.diagnostics:
                packet.update(B_norm=float(a.B.norm()), active_rows=int(a.active_count))
            self.profile_events.extend(a.profile_events)
        for label, start, end in self.profile_events:
            end.synchronize()
            packet["profile_ms"][label] = packet["profile_ms"].get(label, 0.0) + start.elapsed_time(
                end
            )
        return packet


def profiled_forward(label, function, *args, **kwargs):
    runtime = current()
    if runtime is None or not runtime.config.profile:
        return function(*args, **kwargs)
    with runtime.phase(label):
        return function(*args, **kwargs)


def begin_rpc(worker, group_id, projector_path, enabled, config_dict, rng_path=None):
    global _runtime, _pool
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
