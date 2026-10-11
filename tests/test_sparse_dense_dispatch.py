from dataclasses import replace

import pytest
import torch

from dflash_reflex import kernels
from dflash_reflex.adapter import Adapter, initialize_projector
from dflash_reflex.config import ReflexConfig


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_density_dispatch_and_repeated_active_ids():
    torch.manual_seed(5)
    head = torch.randn(257, 32, device="cuda") * 0.1
    h = torch.randn(6, 32, device="cuda")
    a = Adapter(
        head, initialize_projector(head, 4), ReflexConfig(rank=4, topk=4, update_stream=False)
    )
    pw = kernels.ProposalWorkspace(6, 257, 4, 4, "cuda")
    fw = kernels.FeedbackWorkspace(6, 32, 4, 4, "cuda")
    raw = h @ head.t()
    p = torch.zeros_like(raw)
    p[:, 0] = 1
    weights = torch.ones(6, device="cuda")
    key = f"{a.device_name}|6|32|257|4|4|{h.dtype}"
    a.dispatch_cache = {"gpu_thresholds": {key: {"threshold_rows": 20}}}
    assert kernels.dispatch_threshold(a, 6, 32, 257, h.dtype, torch.bfloat16) == 258
    for _ in range(8):
        ids, q, norm, u = kernels.propose(a, h, raw, pw)
        kernels.feedback(a, h, u, ids, q, norm, p, weights, fw, raw=raw)
        assert int(a.active_count) == int(a.active.count_nonzero())
        actual = a.active_ids[: int(a.active_count)]
        assert actual.unique().numel() == actual.numel()
    assert a.host_active_bound == 0
    a.dispatch_counts.zero_()
    for count in (5, 50, 257):
        a.B.zero_()
        a.B[:count].normal_(std=0.01)
        a.active.zero_()
        a.active[:count] = 1
        a.active_count.fill_(count)
        auto = kernels.propose(a, h, raw, pw)[0].clone()
        a.config = replace(a.config, proposal_mode="dense", dense_implementation="fused")
        dense = kernels.propose(a, h, raw, pw)[0].clone()
        torch.testing.assert_close(auto, dense, rtol=0, atol=0)
        a.config = replace(a.config, proposal_mode="auto")
    assert a.dispatch_counts.cpu().tolist() == [1, 5]
    a.reset()
    assert a.dispatch_counts.cpu().tolist() == [0, 0]


def test_legacy_measured_cache_migrates_without_host_density_bound(tmp_path):
    import json

    from dflash_reflex.dispatch import load_cache

    key = "GPU|15|2560|151936|8|16|torch.bfloat16"
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                key: [
                    dict(density=0.02, mode="sparse", implementation="fused"),
                    dict(density=0.2, mode="dense", implementation="fused"),
                    dict(density=1.0, mode="dense", implementation="fused"),
                ]
            }
        )
    )
    migrated = load_cache(path)
    assert migrated["schema_version"] == 2
    assert migrated["gpu_thresholds"][key]["threshold_rows"] == int(0.2 * 151936)
    assert migrated["migrated_from_legacy"]
