import torch

from dflash_reflex.checkpointing import atomic_save
from dflash_reflex.config import ReflexConfig
from dflash_reflex.projector_training import ProjectorTrainer


def test_safe_rng_packet_roundtrip(tmp_path):
    import random

    import numpy as np

    from dflash_reflex.checkpointing import restore_rng, rng_state

    random.seed(33)
    np.random.seed(33)
    torch.manual_seed(33)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(33)
        torch.cuda.get_rng_state()  # Initialize before capturing the state.
    state = rng_state()
    atomic_save(state, tmp_path / "rng.pt")

    def draw():
        values = [random.random(), np.random.random(), torch.randn(5)]
        if torch.cuda.is_available():
            values.append(torch.randn(5, device="cuda"))
        return values

    expected = draw()
    restore_rng(torch.load(tmp_path / "rng.pt", weights_only=True))
    actual = draw()
    for first, second in zip(expected, actual, strict=True):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        else:
            assert first == second


def test_driver_checkpoint_transaction_and_future_rows(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from dflash_reflex import nemo_bridge as bridge
    from dflash_reflex.checkpointing import rng_state

    monkeypatch.setenv("DFLASH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("METHOD", "dflash")
    monkeypatch.setenv("DATASET", "simplelr")
    monkeypatch.setattr(bridge, "_iteration", None)
    monkeypatch.setattr(bridge, "_pending_checkpoint", None)
    bridge.configure_driver()
    loader = bridge.tracked_iterations([1, 2], 0, 0)
    checkpoint = tmp_path / "checkpoints/resume/step_1"
    finalizations = []

    def finalize(path, wait_fn):
        # Transaction must contain its final logged iteration before visibility.
        saved = torch.load(path / "dflash_driver.pt", weights_only=True)
        assert saved["metrics"]["iteration"] == 1
        finalizations.append(path)

    assert next(loader) == 1
    atomic_save(rng_state(), tmp_path / "ipc/inference_rng.pt")
    bridge.defer_driver_checkpoint(SimpleNamespace(begin_finalization=finalize), checkpoint, None)
    assert not finalizations
    bridge.record_iteration(1, {}, {"loss": 1.2, "reward": 0.5})
    assert finalizations == [checkpoint]
    assert next(loader) == 2
    future = tmp_path / "ipc/feedback/future/group.pt"
    atomic_save(dict(iteration_id="future", counters=torch.zeros(8)), future)
    bridge.record_iteration(2, {}, {})
    loader.close()
    assert len((tmp_path / "logs/metrics.jsonl").read_text().splitlines()) == 2
    bridge.configure_driver(checkpoint)
    assert not future.exists()
    rows = (tmp_path / "logs/metrics.jsonl").read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["iteration"] == 1
    assert len(list((tmp_path / "statistics").glob("*.json"))) == 1


def test_projector_pending_optimizer_scheduler_resume(tmp_path):
    cfg = ReflexConfig(rank=3)
    initial = torch.randn(5, 3)
    first = ProjectorTrainer(initial.clone(), cfg)

    def packet(i, version):
        return dict(
            group_id=str(i),
            version=version,
            grad_sum=torch.full((5, 3), float(i)),
            grad_weight=torch.tensor(float(i + 2)),
        )

    first.accumulate(packet(1, 0))
    first.step()
    first.accumulate(packet(2, 1))
    atomic_save(first.state_dict(), tmp_path / "checkpoint.pt")
    restored = ProjectorTrainer(initial.clone(), cfg)
    restored.load_state_dict(torch.load(tmp_path / "checkpoint.pt", weights_only=False))
    for trainer in (first, restored):
        trainer.accumulate(packet(3, 1))
        trainer.step()
    torch.testing.assert_close(first.A, restored.A, rtol=0, atol=0)
    assert first.version == restored.version == 2
    assert first.scheduler.state_dict() == restored.scheduler.state_dict()
    assert restored.last_grad_norm > 0
    restored.publish(tmp_path / "inference.pt")
    inference = torch.load(tmp_path / "inference.pt", weights_only=True)
    torch.testing.assert_close(inference["A"], restored.A)
    assert inference["version"] == 2


def test_actual_dflash_rng_and_pending_backward_resume(tmp_path):
    from test_dflash import original_model

    from dflash_reflex.block_training import block_attention_mask
    from dflash_reflex.checkpointing import capture_training, restore_training

    torch.manual_seed(21)
    first = original_model()
    optimizer = torch.optim.AdamW(first.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9**step)
    anchors = torch.tensor([[1]])
    positions = torch.tensor([[0, 1, 2, 3, 4, 1, 2, 3, 4]])
    mask = block_attention_mask(anchors, 5, 4)

    def backward(model):
        output = model(
            position_ids=positions,
            noise_embedding=torch.randn(1, 4, 16),
            target_hidden=torch.randn(1, 5, 32),
            attention_mask=mask,
        )
        output.square().mean().backward()

    backward(first)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    backward(first)  # Save a genuine partially accumulated backward.
    atomic_save(capture_training(first, optimizer, scheduler), tmp_path / "pending.pt")
    backward(first)
    optimizer.step()
    scheduler.step()
    second = original_model()
    restored_optimizer = torch.optim.AdamW(second.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda step: 0.9**step
    )
    restore_training(
        torch.load(tmp_path / "pending.pt", weights_only=False),
        second,
        restored_optimizer,
        restored_scheduler,
    )
    backward(second)
    restored_optimizer.step()
    restored_scheduler.step()
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert scheduler.state_dict() == restored_scheduler.state_dict()
