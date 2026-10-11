import torch

from dflash_reflex.config import ReflexConfig
from dflash_reflex.projector_training import ProjectorTrainer


def test_pending_gradient_survives_checkpoint_and_normalizes_weight(tmp_path):
    cfg = ReflexConfig(rank=2, projector_lr=0.01)
    t = ProjectorTrainer(torch.ones(3, 2), cfg)
    t.accumulate(
        dict(group_id="a", version=0, grad_sum=torch.ones(3, 2) * 6, grad_weight=torch.tensor(3.0))
    )
    state = t.state_dict()
    clone = ProjectorTrainer(torch.zeros(3, 2), cfg)
    clone.load_state_dict(state)
    before = clone.A.clone()
    assert clone.step()
    assert clone.version == 1 and clone.last_grad_norm > 0
    assert not torch.equal(before, clone.A)
    assert clone.pending_weight == 0
    assert clone.optimizer.state[clone.A]["step"] == 1
    clone.publish(tmp_path / "projector.pt")
    packet = torch.load(tmp_path / "projector.pt", weights_only=True)
    assert packet["version"] == 1 and "B" not in packet
