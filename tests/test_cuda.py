import pytest
import torch

from dflash_reflex.adapter import Adapter, initialize_projector
from dflash_reflex.config import ReflexConfig

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
]


@pytest.mark.parametrize("mode,dense", [("sparse", "auto"), ("dense", "auto"), ("dense", "gemm")])
def test_cuda_proposal_feedback_against_cpu(mode, dense):
    from dflash_reflex.kernels import FeedbackWorkspace, ProposalWorkspace, feedback, propose

    torch.manual_seed(11)
    head = torch.randn(259, 32, device="cuda") * 0.15
    h = torch.randn(9, 32, device="cuda")
    a = initialize_projector(head, 4)
    cfg = ReflexConfig(
        rank=4, topk=5, proposal_mode=mode, dense_implementation=dense, update_stream=False
    )
    gpu = Adapter(head, a, cfg)
    cpu = Adapter(head.cpu(), a.cpu(), cfg)
    pw = ProposalWorkspace(12, 259, 4, 5, "cuda")
    fw = FeedbackWorkspace(12, 32, 4, 5, "cuda")
    raw = h @ head.t()
    p = torch.randn(9, 259, device="cuda").softmax(-1)
    weights = torch.tensor([0.01, 0.02, 0, 0.01, 0.01, 0.03, 0, 0.01, 0.01], device="cuda")
    for _ in range(3):
        ids, q, norm, u = propose(gpu, h, raw, pw)
        expected = cpu.correct_reference(h.cpu(), raw.cpu())
        torch.testing.assert_close(
            ids.cpu(), expected.argsort(-1, descending=True)[:, :5], rtol=0, atol=0
        )
        torch.testing.assert_close(
            q.cpu(), expected.softmax(-1).gather(1, ids.cpu()), rtol=3e-5, atol=1e-6
        )
        feedback(gpu, h, u, ids, q, norm, p, weights, fw)
        cpu.feedback_reference(h.cpu(), raw.cpu(), p.cpu(), weights.cpu())
        torch.testing.assert_close(gpu.B.cpu(), cpu.B, rtol=2e-4, atol=2e-7)
        torch.testing.assert_close(gpu.grad_sum.cpu(), cpu.grad_sum, rtol=3e-4, atol=2e-7)
        torch.testing.assert_close(gpu.grad_weight.cpu(), cpu.grad_weight)


@pytest.mark.parametrize("vocab,topk", [(7, 16), (1031, 16)])
def test_bf16_and_full_union(vocab, topk):
    from dflash_reflex.kernels import FeedbackWorkspace, ProposalWorkspace, feedback, propose

    torch.manual_seed(13)
    head = torch.randn(vocab, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    h = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    a = initialize_projector(head, 8)
    cfg = ReflexConfig(rank=8, topk=topk, update_stream=False)
    gpu, cpu = Adapter(head, a, cfg), Adapter(head.cpu(), a.cpu(), cfg)
    k = min(vocab, topk)
    pw = ProposalWorkspace(8, vocab, 8, k, "cuda")
    fw = FeedbackWorkspace(8, 128, 8, k, "cuda")
    raw = h @ head.t()
    teacher = torch.randn(7, vocab, device="cuda").softmax(-1)
    weights = torch.ones(7, device="cuda")
    for _ in range(3):
        ids, q, norm, u = propose(gpu, h, raw, pw)
        feedback(gpu, h, u, ids, q, norm, teacher, weights, fw)
        cpu.feedback_reference(h.cpu(), raw.cpu(), teacher.cpu(), weights.cpu())
    torch.testing.assert_close(gpu.B.cpu(), cpu.B, rtol=0.01, atol=4e-6)
    torch.testing.assert_close(gpu.grad_sum.cpu(), cpu.grad_sum, rtol=0.01, atol=4e-6)
