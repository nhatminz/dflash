from types import SimpleNamespace as NS

import pytest
import torch

from dflash_reflex.adapter import initialize_projector
from dflash_reflex.config import ReflexConfig
from dflash_reflex.vllm_runtime import Runtime

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
]


def make_runtime(vocab=37, capacity=2, topk=5, width=3, pipeline="fused"):
    head = torch.randn(vocab, 128, device="cuda", dtype=torch.bfloat16) * 0.05
    model = NS(lm_head=NS(weight=head), compute_logits=lambda h: (h @ head.t()).float())
    rt = Runtime(
        NS(
            device="cuda",
            max_num_reqs=capacity,
            speculative_config=NS(num_speculative_tokens=width),
        ),
        "buffers",
        initialize_projector(head, 8),
        0,
        True,
        ReflexConfig(rank=8, topk=topk, pipeline=pipeline, update_stream=True),
    )
    return rt, model


@pytest.mark.parametrize("topk", [1, 5])
@pytest.mark.parametrize("pipeline", ["fused", "selective"])
def test_argmax_ownership_survives_workspace_reuse(topk, pipeline):
    rt, model = make_runtime(topk=topk, pipeline=pipeline)
    rt.request_ids = ["a", "b"]
    ids = rt.propose(model, torch.randn(6, 128, device="cuda", dtype=torch.bfloat16))
    expected = ids.clone()
    assert ids.is_contiguous()
    # Later proposal/selective feedback kernels write this storage, while
    # vLLM can still own/read the previous draft tensor on another stream.
    rt.proposal_workspace.ids.fill_(-1)
    torch.testing.assert_close(ids, expected)
    assert (ids >= 0).all()


@pytest.mark.parametrize("pipeline", ["fused", "selective"])
def test_qwen_vocab_async_feedback_with_compaction(pipeline):
    torch.manual_seed(17)
    rt, model = make_runtime(vocab=151936, capacity=64, topk=16, width=15, pipeline=pipeline)
    requests = [str(i) for i in range(64)]
    for batch in [64, 64, 54, 54, 32, 32, 17, 17, 4, 4, 1, 1]:
        for request in requests[batch:]:
            rt.cache.finish_request(request)
        requests = requests[:batch]
        rt.request_ids, rt.remaining, rt.eos = requests.copy(), [2048] * batch, [[]] * batch
        rt.discard = torch.zeros(batch, device="cuda", dtype=torch.bool)
        hidden = torch.randn(batch * 15, 128, device="cuda", dtype=torch.bfloat16)
        draft = rt.propose(model, hidden).view(batch, 15)
        requests.reverse()
        rt.request_ids = requests.copy()
        lengths = [1 + i % 15 for i in range(batch)]
        flat = torch.cat([row[:length] for row, length in zip(draft.flip(0), lengths)])
        teacher = torch.randn(flat.numel(), 151936, device="cuda").softmax(-1)
        accepted = torch.tensor(
            [min(2, length) for length in lengths], device="cuda", dtype=torch.int32
        )
        output = torch.ones(batch, max(lengths) + 1, device="cuda", dtype=torch.int32)
        rt.verified(teacher, accepted, NS(num_draft_tokens=lengths, draft_token_ids=flat), output)
        # Do not synchronize here: exercise normal allocation lifetimes and
        # the event dependency between feedback and the next proposal.
    packet = rt.finish()
    assert packet["grad_weight"] > 0
    assert torch.isfinite(packet["grad_sum"]).all()


def test_verification_layout_rejected_before_gpu_addressing():
    rt, model = make_runtime()
    rt.request_ids, rt.remaining, rt.eos = ["a"], [8], [[]]
    rt.discard = torch.zeros(1, device="cuda", dtype=torch.bool)
    ids = rt.propose(model, torch.randn(3, 128, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="layout"):
        rt.verified(
            torch.rand(4, 37, device="cuda"),
            torch.zeros(1, device="cuda", dtype=torch.int32),
            NS(num_draft_tokens=[4], draft_token_ids=ids),
            torch.ones(1, 4, device="cuda", dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="vocabulary"):
        rt.verified(
            torch.rand(3, 38, device="cuda"),
            torch.zeros(1, device="cuda", dtype=torch.int32),
            NS(num_draft_tokens=[3], draft_token_ids=ids),
            torch.ones(1, 4, device="cuda", dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="capacity"):
        rt.acceptance_buffer(3)
