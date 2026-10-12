"""Persistent A and one GPU fast adapter per rollout cohort."""

import os
from contextlib import contextmanager

import torch

from .config import ReflexConfig
from .dispatch import load_cache
from .feedback import topk_union_tail


def initialize_projector(head, rank=8, seed=42):
    if rank > head.shape[1] or rank < 1:
        raise ValueError("QR rank exceeds head hidden dimension")
    # Same linspace head rows and CPU FP32 QR as SpecNaacl. No sampling RNG use.
    ids = torch.linspace(0, head.shape[0] - 1, rank, device=head.device).long()
    rows = head.detach().index_select(0, ids).float().t().cpu()
    return torch.linalg.qr(rows, mode="reduced").Q.contiguous().to(head.device)


class Adapter:
    @torch.inference_mode(False)
    def __init__(self, head, projector, config: ReflexConfig):
        self.head, self.config = head, config
        # vLLM invokes proposals inside inference_mode. Keep A versioned.
        with torch.inference_mode(False):
            self.A = projector.detach().float().clone()
        self.B = torch.zeros((head.shape[0], config.rank), device=head.device, dtype=torch.float32)
        self.grad_sum = torch.zeros_like(projector, dtype=torch.float32)
        self.grad_weight = torch.zeros((), device=head.device, dtype=torch.float32)
        self.round = 0
        self.dispatch_counts = torch.zeros(2, device=head.device, dtype=torch.int64)
        self.update_count = torch.zeros((), device=head.device, dtype=torch.int32)
        self.active = torch.zeros(head.shape[0], device=head.device, dtype=torch.int32)
        self.active_ids = torch.empty(head.shape[0], device=head.device, dtype=torch.int32)
        self.active_count = torch.zeros((), device=head.device, dtype=torch.int32)
        self.profile_events = []
        self.stream = (
            torch.cuda.Stream(device=head.device) if head.is_cuda and config.update_stream else None
        )
        self.ready = torch.cuda.Event() if self.stream is not None else None
        self.has_event = False
        self.has_updated = False
        self.host_active_bound = 0  # Legacy reference benchmark only; never used in production.
        self.sparse_dispatch_count = self.dense_dispatch_count = 0
        cache = os.environ.get("OPD_DISPATCH_CACHE", "")
        self.dispatch_cache = load_cache(cache)
        self.revision = 0
        self.prior_feedback_density = None
        self.device_name = torch.cuda.get_device_name(head.device) if head.is_cuda else "cpu"
        self.capability = torch.cuda.get_device_capability(head.device) if head.is_cuda else ()

    def wait(self):
        if self.has_event:
            torch.cuda.current_stream(self.head.device).wait_event(self.ready)

    @contextmanager
    def phase(self, label):
        if self.config.profile and self.head.is_cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            self.profile_events.append((label, start, end))
        else:
            yield

    @torch.no_grad()
    def reset(self):
        self.wait()
        self.B.zero_()
        self.active.zero_()
        self.active_count.zero_()
        self.update_count.zero_()
        self.dispatch_counts.zero_()
        self.grad_sum.zero_()
        self.grad_weight.zero_()
        self.round = self.host_active_bound = 0
        self.revision = 0
        self.has_updated = self.has_event = False
        self.profile_events.clear()
        self.sparse_dispatch_count = self.dense_dispatch_count = 0

    @torch.no_grad()
    def correct_reference(self, hidden, raw, sparse=False):
        u = hidden.float() @ self.A.float()
        if sparse:
            ids = torch.nonzero(self.B.abs().sum(-1) > 0).flatten()
            z = raw.float().clone()
            z[:, ids] += u @ self.B[ids].t()
            return z
        return raw.float() + u @ self.B.t()

    @torch.no_grad()
    def feedback_reference(self, hidden, raw, p, weights):
        """CPU numerical oracle; CUDA uses kernels.feedback, without backward."""
        self.round += 1
        if self.round % self.config.update_interval:
            return None
        u = hidden.float() @ self.A.float()
        q = (raw.float() + u @ self.B.t()).softmax(-1)
        f = topk_union_tail(p, q, weights, self.config.topk)
        # Read B_t before any writes; the first round correctly has zero A grad.
        dg_du = (f.gradient[..., None] * self.B[f.ids]).sum(-2)
        self.grad_sum.add_(hidden.float().t() @ dg_du)
        self.grad_weight.add_(f.weight)
        delta = torch.zeros_like(self.B)
        delta.index_add_(
            0,
            f.ids.flatten(),
            (f.gradient[..., None] * u[:, None, :]).reshape(-1, self.config.rank),
        )
        self.B.add_(delta * (-self.config.fast_lr / torch.where(f.weight > 0, f.weight, 1.0)))
        self.has_updated = True
        return f
