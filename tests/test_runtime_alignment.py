from types import SimpleNamespace

import pytest
import torch

from dflash_reflex.adapter import initialize_projector
from dflash_reflex.config import ReflexConfig
from dflash_reflex.vllm_runtime import Runtime


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("stream", [False, True])
def test_variable_lengths_reordered_selective_and_version_guard(stream):
    torch.manual_seed(16)
    head = torch.randn(37, 16, device="cuda") * 0.1
    model = SimpleNamespace(
        lm_head=SimpleNamespace(weight=head), compute_logits=lambda x: x @ head.t() + 0.3
    )
    runner = SimpleNamespace(
        device="cuda", max_num_reqs=3, speculative_config=SimpleNamespace(num_speculative_tokens=3)
    )
    rt = Runtime(
        runner,
        "a",
        initialize_projector(head, 4),
        0,
        True,
        ReflexConfig(rank=4, topk=5, update_stream=stream, pipeline="selective_masked"),
    )
    rt.request_ids = ["a", "b"]
    rt.remaining = [8, 1]
    rt.eos = [[], []]
    rt.discard = torch.zeros(2, device="cuda", dtype=torch.bool)
    ids = rt.propose(model, torch.randn(6, 16, device="cuda")).view(2, 3).clone()
    rt.request_ids = ["b", "a"]
    rt.remaining = [1, 8]
    metadata = SimpleNamespace(
        num_draft_tokens=[1, 3], draft_token_ids=torch.cat([ids[1, :1], ids[0]])
    )
    rt.verified(
        torch.randn(4, 37, device="cuda").softmax(-1),
        torch.tensor([0, 3], device="cuda"),
        metadata,
        torch.tensor([[2, -1, -1, -1], [4, 5, 6, 7]], device="cuda"),
    )
    rt.adapter.wait()
    assert rt.adapter.grad_weight == 4
    rt.request_ids = ["a"]
    rt.remaining = [8]
    rt.eos = [[]]
    rt.discard = rt.discard[:1]
    ids = rt.propose(model, torch.randn(3, 16, device="cuda"))
    rt.adapter.A.add_(0.1)
    with pytest.raises(RuntimeError, match="version"):
        rt.verified(
            torch.randn(3, 37, device="cuda").softmax(-1),
            torch.tensor([0], device="cuda"),
            SimpleNamespace(num_draft_tokens=[3], draft_token_ids=ids),
            torch.tensor([[1, -1, -1, -1]], device="cuda"),
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_actual_inference_mode_proposal_has_versioned_projector():
    head = torch.randn(37, 16, device="cuda") * 0.1
    model = SimpleNamespace(
        lm_head=SimpleNamespace(weight=head), compute_logits=lambda x: x @ head.t()
    )
    runner = SimpleNamespace(
        device="cuda", max_num_reqs=1, speculative_config=SimpleNamespace(num_speculative_tokens=3)
    )
    rt = Runtime(
        runner,
        "a",
        initialize_projector(head, 4),
        0,
        True,
        ReflexConfig(rank=4, topk=5, pipeline="selective"),
    )
    rt.request_ids = ["a"]
    rt.remaining = [8]
    rt.eos = [[]]
    rt.discard = torch.zeros(1, device="cuda", dtype=torch.bool)
    with torch.inference_mode():
        ids = rt.propose(model, torch.randn(3, 16, device="cuda"))
        rt.verified(
            torch.randn(3, 37, device="cuda").softmax(-1),
            torch.tensor([0], device="cuda"),
            SimpleNamespace(num_draft_tokens=[3], draft_token_ids=ids),
            torch.tensor([[1, -1, -1, -1]], device="cuda"),
        )
    packet = rt.finish()
    assert packet["grad_weight"] == 1
    rt.adapter.reset()
    assert rt.adapter.B.norm() == 0
