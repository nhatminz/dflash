"""Rollout feedback optimizer; global numerator/denominator aggregation."""

from pathlib import Path

import torch
import torch.distributed as dist

from .checkpointing import atomic_save


class ProjectorTrainer:
    def __init__(self, initial, config):
        self.A = torch.nn.Parameter(initial.float())
        self.optimizer = torch.optim.AdamW(
            [
                dict(
                    params=[self.A],
                    lr=config.projector_lr,
                    weight_decay=config.projector_weight_decay,
                    name="draft_opd_projector",
                )
            ]
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        self.pending_sum = torch.zeros_like(self.A)
        self.pending_weight = torch.zeros((), device=self.A.device)
        self.version = 0
        self.consumed = set()
        self.last_grad_norm = None

    @torch.no_grad()
    def accumulate(self, packet):
        if packet["group_id"] in self.consumed:
            raise RuntimeError("duplicate rollout A gradient")
        if packet["version"] != self.version:
            raise RuntimeError(f"stale A feedback {packet['version']} != trainer {self.version}")
        if packet["grad_sum"] is not None:
            self.pending_sum.add_(packet["grad_sum"].to(self.A.device))
            self.pending_weight.add_(packet["grad_weight"].to(self.A.device))
        self.consumed.add(packet["group_id"])

    @torch.no_grad()
    def step(self, group=None):
        numerator, denominator = self.pending_sum.clone(), self.pending_weight.clone()
        if dist.is_initialized():
            # Packets belong to one inference owner. Only rank 0 submits them;
            # sum, never mean, before dividing by the GLOBAL feedback weight.
            dist.all_reduce(numerator, group=group)
            dist.all_reduce(denominator, group=group)
        if denominator.item() == 0:  # optimizer boundary only
            return False
        self.A.grad = numerator / denominator
        if not torch.isfinite(self.A.grad).all():
            raise FloatingPointError("nonfinite rollout projector gradient")
        self.last_grad_norm = float(self.A.grad.norm())
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.pending_sum.zero_()
        self.pending_weight.zero_()
        self.version += 1
        return True

    def state_dict(self):
        return dict(
            A=self.A.detach().cpu(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            pending_sum=self.pending_sum.cpu(),
            pending_weight=self.pending_weight.cpu(),
            version=self.version,
            consumed=sorted(self.consumed),
            last_grad_norm=self.last_grad_norm,
        )

    def load_state_dict(self, state):
        with torch.no_grad():
            self.A.copy_(state["A"])
            self.pending_sum.copy_(state["pending_sum"])
            self.pending_weight.copy_(state["pending_weight"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.version, self.consumed = state["version"], set(state["consumed"])
        self.last_grad_norm = state["last_grad_norm"]

    def publish(self, path):
        atomic_save(
            dict(
                A=self.A.detach().cpu(),
                version=self.version,
                grad_norm=self.last_grad_norm,
                A_norm=float(self.A.norm()),
            ),
            Path(path),
        )
