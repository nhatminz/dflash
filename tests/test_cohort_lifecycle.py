import pytest
import torch

from dflash_reflex.rollout_state import CohortState


def test_request_finished_reuse_never_exposes_pending_old_slot():
    state = CohortState("cohort", 2, 3, 8, 2, 2, "cpu", torch.float32)
    slot = state.proposal_slots(["old"])[0]
    state.finish_request("old")
    assert state.proposal_slots(["new"]) == [slot]
    with pytest.raises(RuntimeError, match="without proposal"):
        state.verification_slots(["old"], [1])
    assert state.verification_slots(["new"], [3]) == [slot]
    with pytest.raises(RuntimeError):
        state.verification_slots(["new"], [3])
