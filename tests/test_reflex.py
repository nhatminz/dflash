import torch

from dflash_reflex.adapter import Adapter, initialize_projector
from dflash_reflex.config import ReflexConfig
from dflash_reflex.feedback import feedback_weights, topk_union_tail
from dflash_reflex.rollout_state import CohortRegistry, CohortState


def test_operational_gradient_and_update_order(tensors):
    head, h, p = tensors
    config = ReflexConfig(rank=4, topk=5, accepted_weight=0.01)
    adapter = Adapter(head, initialize_projector(head, 4), config)
    raw = h @ head.t()
    weights = torch.full((h.shape[0],), 0.01)
    assert not adapter.B.any()
    torch.testing.assert_close(adapter.correct_reference(h, raw), raw)
    adapter.feedback_reference(h, raw, p, weights)
    assert adapter.B.norm() > 0
    assert adapter.grad_sum.norm() == 0  # B_0 is zero, deliberately.
    b = adapter.B.clone()
    u = h @ adapter.A
    f = topk_union_tail(p, (raw + u @ b.t()).softmax(-1), weights, 5)
    expected = h.t() @ (f.gradient[..., None] * b[f.ids]).sum(-2)
    adapter.feedback_reference(h, raw, p, weights)
    torch.testing.assert_close(adapter.grad_sum, expected)
    assert adapter.grad_sum.norm() > 0
    torch.testing.assert_close(adapter.grad_weight, 2 * weights.sum())
    torch.testing.assert_close(
        adapter.correct_reference(h, raw, sparse=True), adapter.correct_reference(h, raw)
    )
    adapter.reset()
    assert not adapter.B.any() and not adapter.grad_sum.any()


def test_union_and_tail():
    p = torch.tensor([[0.6, 0.25, 0.1, 0.05]])
    q = torch.tensor([[0.1, 0.3, 0.2, 0.4]])
    f = topk_union_tail(p, q, torch.tensor([2.0]), 1)
    assert f.ids.tolist() == [[3, 0]]
    torch.testing.assert_close(f.p_tail, torch.tensor([0.35]))
    torch.testing.assert_close(f.q_tail, torch.tensor([0.5]))
    torch.testing.assert_close(f.gradient, torch.tensor([[0.7, -1.0]]))
    expected = (
        0.05 * torch.log(torch.tensor(0.05 / 0.4))
        + 0.6 * torch.log(torch.tensor(0.6 / 0.1))
        + 0.35 * torch.log(torch.tensor(0.35 / 0.5))
    )
    torch.testing.assert_close(f.kl[0], expected)
    assert f.gradient.sum().abs() > 0  # Tail derivative is intentionally omitted.


def test_prefix_rejection_finished_and_padding():
    weights = feedback_weights(
        torch.tensor([0, 2, 4, 1]),
        torch.tensor([4, 4, 4, 2]),
        4,
        valid=torch.tensor([True, True, True, False]),
        emitted_limit=torch.tensor([4, 2, 3, 2]),
        rejected_weight=0.5,
    )
    torch.testing.assert_close(
        weights, torch.tensor([[0.5, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]])
    )


def test_request_reorder_and_cohort_isolation():
    state = CohortState("one", 4, 3, 8, 2, 2, "cpu", torch.float32)
    slots = state.proposal_slots(["a", "b"])
    state.h[slots[0]].fill_(11)
    state.h[slots[1]].fill_(22)
    recovered = state.verification_slots(["b", "a"], [3, 3])
    assert state.h[recovered[0], 0, 0] == 22
    assert state.h[recovered[1], 0, 0] == 11
    state.finish_request("a")
    assert "b" in state.requests
    # Request completion changes request storage, never cohort fast weights.
    registry = CohortRegistry()

    class State:
        def reset(self):
            self.closed = True

    first = registry.begin("x", State)
    second = registry.begin("y", State)
    assert first is not second
    registry.end("x")
    assert first.closed
    assert registry.groups["y"] is second
