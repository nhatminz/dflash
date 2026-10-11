"""Block feedback masks and the exact SpecNaacl sparse update oracle."""

from dataclasses import dataclass

import torch


@dataclass
class UnionFeedback:
    ids: torch.Tensor
    valid: torch.Tensor
    gradient: torch.Tensor
    p_tail: torch.Tensor
    q_tail: torch.Tensor
    kl: torch.Tensor
    tv: torch.Tensor
    weight: torch.Tensor


def feedback_weights(
    accepted,
    lengths,
    width,
    *,
    valid=None,
    emitted_limit=None,
    accepted_weight=1.0,
    rejected_weight=1.0,
):
    """accepted comes from the sampler decisions, never token equality."""
    slots = torch.arange(width, device=accepted.device)[None, :]
    accept = slots < accepted[:, None]
    reject = (slots == accepted[:, None]) & (accepted < lengths)[:, None]
    live = slots < lengths[:, None]
    if valid is not None:
        live &= valid[:, None]
    if emitted_limit is not None:
        live &= slots < emitted_limit[:, None]
    return torch.where(
        live & accept, accepted_weight, torch.where(live & reject, rejected_weight, 0.0)
    )


def topk_union_tail(p, q, weights, topk=16):
    """CPU oracle of opd_reflex_kernels._union.

    The tail contributes to the reported KL/TV. SpecNaacl's operational
    update is w*(q-p) on union coordinates only; it does NOT differentiate
    the coarsened tail bucket. Preserve that approximation deliberately.
    """
    k = min(topk, p.shape[-1])
    di = torch.argsort(q, descending=True, stable=True)[..., :k]
    ti = torch.argsort(p, descending=True, stable=True)[..., :k]
    positive = p.gather(-1, ti) > 0
    valid = torch.cat(
        (
            torch.ones_like(di, dtype=torch.bool),
            positive & ~(ti[..., :, None] == di[..., None, :]).any(-1),
        ),
        -1,
    )
    ids = torch.cat((di, ti), -1)
    pu = torch.where(valid, p.gather(-1, ids), 0.0)
    qu = torch.where(valid, q.gather(-1, ids), 0.0)
    full = valid.sum(-1) == p.shape[-1]
    pt = torch.where(full, 0.0, (1 - pu.sum(-1)).clamp_min(0))
    qt = torch.where(full, 0.0, (1 - qu.sum(-1)).clamp_min(0))
    kl = torch.where(
        pu > 0, pu * (pu.log() - qu.clamp_min(torch.finfo(torch.float32).tiny).log()), 0.0
    ).sum(-1)
    kl += torch.where(
        pt > 0, pt * (pt.log() - qt.clamp_min(torch.finfo(torch.float32).tiny).log()), 0.0
    )
    tv = ((pu - qu).abs().sum(-1) + (pt - qt).abs()) * 0.5
    g = torch.where(valid, (qu - pu) * weights[..., None], 0.0)
    return UnionFeedback(ids, valid, g, pt, qt, kl, tv, weights.sum())
