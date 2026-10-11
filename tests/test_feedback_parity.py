import pytest
import torch

from dflash_reflex import kernels
from dflash_reflex.adapter import Adapter, initialize_projector
from dflash_reflex.config import ReflexConfig
from dflash_reflex.feedback import topk_union_tail


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("masked", [False, True])
def test_selected_feedback_uses_exact_raw_and_old_b(masked):
    torch.manual_seed(31)
    head = torch.randn(1031, 64, device="cuda", dtype=torch.bfloat16) * 0.1
    h = torch.randn(9, 64, device="cuda", dtype=torch.bfloat16)
    cfg = ReflexConfig(rank=8, topk=16, update_stream=False)
    gpu = Adapter(head, initialize_projector(head), cfg)
    cpu = Adapter(head.cpu(), gpu.A.cpu(), cfg)
    # A bias/processor makes a reconstructed h@head explicitly wrong.
    raw = (h @ head.t()).float() + torch.linspace(-0.3, 0.3, 1031, device="cuda")
    pw = kernels.ProposalWorkspace(9, 1031, 8, 16, "cuda")
    fw = kernels.FeedbackWorkspace(9, 64, 8, 16, "cuda")
    p = torch.randn(9, 1031, device="cuda").softmax(-1)
    weights = torch.tensor([1.0, 2.0, 0, 0, 0, 1, 0, 0, 0], device="cuda")
    for _ in range(3):
        di, dq, norm, u = kernels.greedy_propose(gpu, h, raw, pw)
        expected = cpu.correct_reference(h.cpu(), raw.cpu())
        torch.testing.assert_close(di[:, 0].cpu(), expected.argmax(-1), rtol=0, atol=0)
        result = kernels.feedback(
            gpu,
            h,
            u,
            di,
            dq,
            norm,
            p,
            weights,
            fw,
            raw=raw,
            proposal_workspace=pw,
            selective=True,
            masked_teacher=masked,
        )
        f = cpu.feedback_reference(h.cpu(), raw.cpu(), p.cpu(), weights.cpu())
        torch.testing.assert_close(gpu.B.cpu(), cpu.B, rtol=1e-4, atol=2e-7)
        torch.testing.assert_close(gpu.grad_sum.cpu(), cpu.grad_sum, rtol=3e-4, atol=3e-7)
        torch.testing.assert_close(result[0].cpu(), weights.sum().cpu())
        torch.testing.assert_close(
            result[1].cpu(), (f.kl * weights.cpu()).sum(), rtol=1e-4, atol=2e-6
        )
        assert result[3] == 3


def test_zero_q_zero_tail_finite_diagnostic_and_sparse_gradient():
    p = torch.tensor([[1.0, 0, 0], [0.0, 0.5, 0.5]])
    q = torch.tensor([[0.0, 1, 0], [0.0, 0.5, 0.5]])
    f = topk_union_tail(p, q, torch.ones(2), 3)
    assert torch.isfinite(f.kl).all()
    assert f.kl[1] == 0
    assert f.p_tail.sum() == f.q_tail.sum() == 0
    torch.testing.assert_close(f.gradient.sum(-1), torch.zeros(2))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_teacher_topk_ties_are_identical_for_all_candidates():
    torch.manual_seed(81)
    head = torch.randn(259, 32, device="cuda") * 0.1
    h = torch.randn(4, 32, device="cuda")
    raw = h @ head.t()
    # Positive ties cross the top-k boundary, including uniform teacher.
    p = torch.full((4, 259), 1 / 259, device="cuda")
    p[0, :2] = 0.02
    p[0] /= p[0].sum()
    weights = torch.tensor([1.0, 1.0, 0.0, 1.0], device="cuda")
    states = []
    for selective, masked in ((False, False), (True, False), (True, True)):
        cfg = ReflexConfig(rank=4, topk=5, update_stream=False)
        a = Adapter(head, initialize_projector(head, 4), cfg)
        a.B.normal_(std=0.01)
        a.active.fill_(1)
        a.active_count.fill_(259)
        a.has_updated = True
        if states:
            a.B.copy_(states[0][0])
        before = a.B.clone()
        pw = kernels.ProposalWorkspace(4, 259, 4, 5, "cuda")
        fw = kernels.FeedbackWorkspace(4, 32, 4, 5, "cuda")
        fn = kernels.greedy_propose if selective else kernels.propose
        ids, q, norm, u = fn(a, h, raw, pw)
        kernels.feedback(
            a,
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
            selective=selective,
            masked_teacher=masked,
        )
        states.append(
            (
                before,
                a.B.clone(),
                a.grad_sum.clone(),
                (fw.teacher.ids if masked else fw.teacher_sorted).clone(),
            )
        )
    for state in states[1:]:
        torch.testing.assert_close(state[1], states[0][1], atol=2e-7, rtol=1e-5)
        torch.testing.assert_close(state[2], states[0][2], atol=2e-7, rtol=1e-5)
        torch.testing.assert_close(state[3][weights > 0], states[0][3][weights > 0], rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_greedy_teacher_delta_without_vocabulary_probability_allocation():
    torch.manual_seed(118)
    head = torch.randn(1031, 32, device="cuda") * 0.1
    h = torch.randn(6, 32, device="cuda")
    cfg = ReflexConfig(rank=4, topk=5, update_stream=False)
    a = Adapter(head, initialize_projector(head, 4), cfg)
    cpu = Adapter(head.cpu(), a.A.cpu(), cfg)
    logits = torch.randn(6, 1031, device="cuda")
    p = torch.zeros_like(logits).scatter_(1, logits.argmax(-1, keepdim=True), 1.0)
    raw = h @ head.t()
    weights = torch.tensor([1.0, 1, 0, 0, 1, 0], device="cuda")
    pw = kernels.ProposalWorkspace(6, 1031, 4, 5, "cuda")
    fw = kernels.FeedbackWorkspace(6, 32, 4, 5, "cuda")
    for _ in range(3):
        ids, q, norm, u = kernels.propose(a, h, raw, pw)
        sums = kernels.feedback(
            a, h, u, ids, q, norm, logits, weights, fw, raw=raw, greedy_teacher=True
        )
        cpu.feedback_reference(h.cpu(), raw.cpu(), p.cpu(), weights.cpu())
        torch.testing.assert_close(a.B.cpu(), cpu.B, rtol=2e-4, atol=2e-7)
        torch.testing.assert_close(a.grad_sum.cpu(), cpu.grad_sum, rtol=3e-4, atol=2e-7)
        assert torch.isfinite(sums).all()
    assert fw.teacher is None


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fp64_union_tail_a_gradient_and_b_update_oracle():
    from reference.original_feedback import topk_union_tail as double_union

    torch.manual_seed(72)
    head = torch.randn(1009, 64, device="cuda") * 0.1
    h = torch.randn(5, 64, device="cuda")
    cfg = ReflexConfig(rank=4, topk=7, update_stream=False)
    a = Adapter(head, initialize_projector(head, 4), cfg)
    a.B.normal_(std=0.02)
    a.active.fill_(1)
    a.active_count.fill_(1009)
    a.has_updated = True
    b = a.B.cpu().double().clone()
    hd = h.cpu().double()
    ad = a.A.cpu().double()
    raw = h @ head.t()
    p = torch.randn(5, 1009, device="cuda").softmax(-1)
    weights = torch.tensor([1.0, 0.5, 0, 2.0, 1.0], device="cuda")
    ud = hd @ ad
    qd = (raw.cpu().double() + ud @ b.t()).softmax(-1)
    f = double_union(p.cpu().double(), qd, weights.cpu().double(), 7)
    expected_a = hd.t() @ (f.gradient[:, :, None] * b[f.ids]).sum(-2)
    delta = torch.zeros_like(b)
    delta.index_add_(0, f.ids.flatten(), (f.gradient[:, :, None] * ud[:, None, :]).reshape(-1, 4))
    expected_b = b - 0.01 * delta / f.weight
    pw = kernels.ProposalWorkspace(5, 1009, 4, 7, "cuda")
    fw = kernels.FeedbackWorkspace(5, 64, 4, 7, "cuda")
    ids, q, norm, u = kernels.greedy_propose(a, h, raw, pw)
    sums = kernels.feedback(
        a, h, u, ids, q, norm, p, weights, fw, raw=raw, proposal_workspace=pw, selective=True
    )
    torch.testing.assert_close(a.B.cpu().double(), expected_b, rtol=2e-5, atol=3e-9)
    torch.testing.assert_close(a.grad_sum.cpu().double(), expected_a, rtol=2e-5, atol=5e-9)
    torch.testing.assert_close(
        sums[1].cpu().double(), (f.kl * weights.cpu().double()).sum(), rtol=1e-5, atol=1e-7
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_explicit_gemm_feedback_uses_the_exact_corrected_proposal_logits():
    torch.manual_seed(123)
    head = torch.randn(1031, 64, device="cuda", dtype=torch.bfloat16) * 0.1
    h = torch.randn(5, 64, device="cuda", dtype=torch.bfloat16)
    cfg = ReflexConfig(
        rank=8, topk=16, update_stream=False, proposal_mode="dense", dense_implementation="gemm"
    )
    a = Adapter(head, initialize_projector(head), cfg)
    a.B.normal_(std=0.5)
    a.active.fill_(1)
    a.active_count.fill_(1031)
    a.has_updated = True
    before = a.B.cpu().double().clone()
    raw = (h @ head.t()).float() + torch.linspace(-0.3, 0.3, 1031, device="cuda")
    p = torch.randn(5, 1031, device="cuda").softmax(-1)
    weights = torch.tensor([1.0, 0.5, 0, 2.0, 1.0], device="cuda")
    pw = kernels.ProposalWorkspace(5, 1031, 8, 16, "cuda")
    fw = kernels.FeedbackWorkspace(5, 64, 8, 16, "cuda")
    ids, q, norm, u = kernels.propose(a, h, raw, pw)
    assert pw.corrected_raw is not None
    torch.testing.assert_close(ids[:, 0], pw.corrected_raw.argmax(-1), rtol=0, atol=0)
    expected_q = pw.corrected_raw.cpu().double().softmax(-1)
    f = topk_union_tail(p.cpu().double(), expected_q, weights.cpu().double(), 16)
    expected_a = h.cpu().double().t() @ (f.gradient[:, :, None] * before[f.ids]).sum(-2)
    delta = torch.zeros_like(before)
    delta.index_add_(
        0, f.ids.flatten(), (f.gradient[:, :, None] * u.cpu().double()[:, None, :]).reshape(-1, 8)
    )
    kernels.feedback(a, h, u, ids, q, norm, p, weights, fw, raw=raw, proposal_workspace=pw)
    torch.testing.assert_close(a.grad_sum.cpu().double(), expected_a, rtol=3e-5, atol=1e-7)
    torch.testing.assert_close(
        a.B.cpu().double(), before - 0.01 * delta / f.weight, rtol=3e-5, atol=1e-7
    )
    assert a.dispatch_counts.cpu().tolist() == [0, 1]
