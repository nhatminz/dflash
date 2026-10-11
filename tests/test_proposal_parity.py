import pytest
import torch
from reference import original_kernels as old

from dflash_reflex import kernels
from dflash_reflex.adapter import Adapter, initialize_projector
from dflash_reflex.config import ReflexConfig

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("density", [0.0, 0.01, 1.0])
def test_exact_argmax_and_statistics(dtype, density):
    torch.manual_seed(121)
    vocab, hidden, rows = 151936, 32, 3
    head = torch.randn(vocab, hidden, device="cuda", dtype=dtype) * 0.02
    h = torch.randn(rows, hidden, device="cuda", dtype=dtype)
    a = Adapter(
        head,
        initialize_projector(head, 8),
        ReflexConfig(update_stream=False, proposal_mode="sparse"),
    )
    active = int(vocab * density)
    a.B[:active].normal_(std=0.01)
    a.active[:active] = 1
    a.active_count.fill_(active)
    a.has_updated = active > 0
    raw = (h @ head.t()).contiguous()
    # Include exactly repeated and adjacent representable logits.
    raw[:, 3] = raw[:, 4] = 10
    raw[:, 5] = torch.nextafter(raw[:, 3], torch.full_like(raw[:, 3], float("-inf")))
    ow = old.ProposalWorkspace(rows, vocab, 8, 16, "cuda")
    nw = kernels.ProposalWorkspace(rows, vocab, 8, 16, "cuda")
    di, dq, dnorm, u = old.propose(a, h, raw, ow)
    ids, _, _, nu = kernels.greedy_propose(a, h, raw, nw)
    torch.testing.assert_close(ids[:, 0], di[:, 0], rtol=0, atol=0)
    weights = torch.tensor([1, 0, 2.0], device="cuda")
    kernels.statistics(a, nu, raw, nw, weights)
    valid = weights > 0
    torch.testing.assert_close(nw.ids[valid], di[valid], rtol=0, atol=0)
    torch.testing.assert_close(nw.q[valid], dq[valid], rtol=3e-6, atol=1e-7)
    torch.testing.assert_close(nw.norm[valid], dnorm[valid], rtol=3e-6, atol=1e-7)
