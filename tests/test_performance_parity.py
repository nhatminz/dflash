"""Numerical contract established before optimizing the complete OPD path."""

import pytest
import torch

from dflash_reflex import kernels
from dflash_reflex.adapter import Adapter, initialize_projector
from dflash_reflex.config import ReflexConfig


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("prefix", [0, 1, 2, 4, 8, 15])
@pytest.mark.parametrize("pipeline", ["fused", "selective", "selective_masked"])
def test_complete_round_matches_cpu_oracle_for_short_and_long_prefixes(prefix, pipeline):
    torch.manual_seed(950)
    rows, hidden, vocab = 30, 32, 1031
    head = torch.randn(vocab, hidden, device="cuda") * 0.1
    h = torch.randn(rows, hidden, device="cuda")
    raw = h @ head.t() + torch.linspace(-0.2, 0.3, vocab, device="cuda")
    p = torch.randn(rows, vocab, device="cuda").softmax(-1)
    weights = (torch.arange(rows, device="cuda") % 15 <= prefix).float()
    cfg = ReflexConfig(rank=8, topk=16, update_stream=False)
    gpu = Adapter(head, initialize_projector(head), cfg)
    cpu = Adapter(head.cpu(), gpu.A.cpu(), cfg)
    pw = kernels.ProposalWorkspace(rows, vocab, 8, 16, "cuda")
    fw = kernels.FeedbackWorkspace(rows, hidden, 8, 16, "cuda")
    for _ in range(3):
        fn = kernels.propose if pipeline == "fused" else kernels.greedy_propose
        ids, q, norm, u = fn(gpu, h, raw, pw)
        z = cpu.correct_reference(h.cpu(), raw.cpu())
        torch.testing.assert_close(ids[:, 0].cpu(), z.argmax(-1), rtol=0, atol=0)
        sums = kernels.feedback(
            gpu,
            h,
            u,
            ids,
            q,
            norm,
            p,
            weights,
            fw,
            raw=raw,
            proposal_workspace=pw,
            selective=pipeline != "fused",
            masked_teacher=pipeline == "selective_masked",
        )
        reference = cpu.feedback_reference(h.cpu(), raw.cpu(), p.cpu(), weights.cpu())
        torch.testing.assert_close(gpu.B.cpu(), cpu.B, rtol=3e-4, atol=3e-7)
        torch.testing.assert_close(gpu.grad_sum.cpu(), cpu.grad_sum, rtol=6e-4, atol=5e-7)
        torch.testing.assert_close(
            sums[1].cpu(), (reference.kl * weights.cpu()).sum(), rtol=2e-4, atol=5e-6
        )
        valid = weights > 0
        torch.testing.assert_close(
            pw.q[valid].cpu(),
            z.softmax(-1).gather(1, pw.ids.cpu())[valid.cpu()],
            rtol=5e-5,
            atol=1e-7,
        )
    assert gpu.update_count.item() == 3
    assert gpu.grad_weight.item() == 3 * weights.sum().item()
    assert gpu.grad_sum.norm().item() > 0
