from types import SimpleNamespace

import pytest
import torch

from dflash_reflex.adapter import initialize_projector
from dflash_reflex.config import ReflexConfig
from dflash_reflex.vllm_runtime import Runtime


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("stream", [False, True])
def test_real_runtime_shared_b_eos_reordering_and_stream(stream):
    torch.manual_seed(8)
    head = torch.randn(37, 16, device="cuda") * 0.2

    class Model:
        lm_head = SimpleNamespace(weight=head)

        def compute_logits(self, h):
            return h @ head.t()

    model = Model()
    runner = SimpleNamespace(
        device="cuda", max_num_reqs=4, speculative_config=SimpleNamespace(num_speculative_tokens=3)
    )
    runtime = Runtime(
        runner,
        "cohort",
        initialize_projector(head, 4).cpu(),
        0,
        True,
        ReflexConfig(rank=4, topk=5, update_stream=stream),
    )
    runtime.request_ids = ["one", "two"]
    runtime.remaining = [8, 8]
    runtime.eos = [[], []]
    runtime.discard = torch.zeros(2, dtype=torch.bool, device="cuda")
    h = torch.randn(6, 16, device="cuda")
    draft = runtime.propose(model, h).view(2, 3)
    # The scheduler reorders requests before verifying the cached block.
    runtime.request_ids = ["two", "one"]
    target = torch.randn(6, 37, device="cuda").softmax(-1)
    accepted = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
    out = torch.tensor([[7, 9, -1, -1], [2, 4, 6, 8]], device="cuda", dtype=torch.int32)
    metadata = SimpleNamespace(num_draft_tokens=[3, 3], draft_token_ids=draft.flip(0).flatten())
    runtime.verified(target, accepted, metadata, out)
    runtime.adapter.wait()
    assert runtime.adapter.B.norm() > 0
    assert runtime.adapter.grad_sum.norm() == 0
    assert runtime.adapter.grad_weight == 5
    # Completing one response frees its cache only; B remains shared/live.
    before = runtime.adapter.B.clone()
    runtime.cache.finish_request("one")
    torch.testing.assert_close(runtime.adapter.B, before)
    runtime.request_ids, runtime.remaining, runtime.eos = ["two"], [3], [[9]]
    runtime.discard = runtime.discard[:1]
    next_draft = runtime.propose(model, h[:3])
    runtime.verified(
        target[:3],
        torch.tensor([2], device="cuda", dtype=torch.int32),
        SimpleNamespace(num_draft_tokens=[3], draft_token_ids=next_draft),
        torch.tensor([[7, 9, 5, -1]], device="cuda", dtype=torch.int32),
    )
    packet = runtime.finish()
    assert packet["grad_sum"].norm() > 0
    assert packet["grad_weight"] == 7  # EOS cuts out rejection slot 2.
    assert packet["counters"][:4].tolist() == [8, 3, 6, 8]
    assert "B" not in packet
    with pytest.raises(RuntimeError):
        runtime.finish()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_engine_boundary_refits_a_reuses_storage_and_resets_b(tmp_path, monkeypatch):
    from dataclasses import asdict

    from dflash_reflex import vllm_runtime as engine
    from dflash_reflex.checkpointing import atomic_save

    monkeypatch.setattr(engine, "_runtime", None)
    monkeypatch.setattr(engine, "_pool", None)
    head = torch.randn(37, 16, device="cuda") * 0.2
    model = SimpleNamespace(
        lm_head=SimpleNamespace(weight=head), compute_logits=lambda h: h @ head.t()
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            device="cuda",
            max_num_reqs=2,
            speculative_config=SimpleNamespace(num_speculative_tokens=3),
        )
    )
    path = tmp_path / "projector.pt"
    initial = initialize_projector(head, 4).cpu()
    atomic_save(dict(A=initial, version=0), path)
    config = asdict(ReflexConfig(rank=4, topk=5))
    assert engine.begin_rpc(worker, "first", str(path), True, config) == 0
    runtime = engine.current()
    runtime.request_ids, runtime.remaining, runtime.eos = ["a"], [8], [[]]
    runtime.discard = torch.zeros(1, device="cuda", dtype=torch.bool)
    hidden = torch.randn(3, 16, device="cuda")
    ids = runtime.propose(model, hidden)
    runtime.verified(
        torch.randn(3, 37, device="cuda").softmax(-1),
        torch.tensor([1], device="cuda", dtype=torch.int32),
        SimpleNamespace(num_draft_tokens=[3], draft_token_ids=ids),
        torch.tensor([[1, 2, -1, -1]], device="cuda", dtype=torch.int32),
    )
    adapter = runtime.adapter
    engine.finish_rpc(worker)
    assert adapter.B.norm() > 0
    updated = initial + 0.01
    atomic_save(dict(A=updated, version=1), path)
    assert engine.begin_rpc(worker, "second", str(path), True, config) == 1
    second = engine.current()
    assert second.adapter is adapter
    assert not second.cache.requests
    assert adapter.B.norm() == adapter.grad_sum.norm() == adapter.grad_weight == 0
    torch.testing.assert_close(adapter.A.cpu(), updated)
    engine.finish_rpc(worker)
    assert engine.begin_rpc(worker, "baseline", str(path), False, config) == -1
    assert engine.current().adapter is None
    engine.finish_rpc(worker)
