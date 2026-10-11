from types import SimpleNamespace

from dflash_reflex.config import ReflexConfig
from dflash_reflex.vllm_runtime import Runtime


def test_production_baseline_allocates_no_device_acceptance_or_reflex():
    runner = SimpleNamespace(
        device="cpu", max_num_reqs=3, speculative_config=SimpleNamespace(num_speculative_tokens=15)
    )
    rt = Runtime(runner, "baseline", None, -1, False, ReflexConfig())
    assert rt.adapter is rt.cache is rt.accepted_buffer is rt.positions is None
    assert rt.counters.device.type == "cpu"
    rt.verified(None, None, None, None)
    packet = rt.finish()
    assert not packet["core_metrics_available"]
