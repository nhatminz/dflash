"""Exercise the actual NeMo tracker against multiple-cluster discovery failures."""

from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party/nemo-rl"
RELATIVE = "nemo_rl/utils/memory_tracker.py"
ADDRESSES = [f"172.29.135.196:{port}" for port in [41841, 39037, 33023, 38315, 46629]]


@pytest.fixture
def ray_boundary(monkeypatch):
    context = SimpleNamespace(gcs_address=ADDRESSES[0])
    calls = []
    ray = ModuleType("ray")
    ray.__path__ = []
    ray.get_runtime_context = lambda: context
    scripts = ModuleType("ray.scripts")
    scripts.__path__ = []
    api = ModuleType("ray.scripts.scripts")

    def summary(address=None, *, stats_only=False, num_entries=None):
        if address is None:
            raise ConnectionError(f"Found multiple active Ray instances: {set(ADDRESSES)}")
        if address not in ADDRESSES:
            raise ConnectionError(f"Connected cluster unavailable: {address}")
        calls.append((address, stats_only, num_entries))
        return "Object store stats for " + address

    api.memory_summary = summary
    ray.scripts = scripts
    scripts.scripts = api
    for module in [ray, scripts, api]:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return context, calls


def load_tracker(monkeypatch, source=None):
    module = ModuleType("nemo_memory_tracker_regression")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = (REPO / RELATIVE).read_text() if source is None else source
    exec(compile(source, RELATIVE, "exec"), module.__dict__)
    return module


def test_pristine_tracker_reproduces_user_failure(ray_boundary, monkeypatch):
    original = subprocess.check_output(
        ["git", "-C", str(REPO), "show", "HEAD:" + RELATIVE], text=True
    )
    tracker = load_tracker(monkeypatch, original).MemoryTracker()
    tracker.snapshot_start_of_stage("Preparation", ["batch"])
    with pytest.raises(ConnectionError, match="multiple active Ray instances"):
        tracker.snapshot_start_of_stage("Generation", ["batch", "sampling_params"])


@pytest.mark.parametrize("connected", [ADDRESSES[0], ADDRESSES[-1]])
def test_actual_tracker_uses_connected_cluster(ray_boundary, monkeypatch, capsys, connected):
    context, calls = ray_boundary
    context.gcs_address = connected
    # An inherited environment setting can describe another job. It must not
    # override the cluster to which this driver is already connected.
    monkeypatch.setenv("RAY_ADDRESS", ADDRESSES[2])
    tracker = load_tracker(monkeypatch).MemoryTracker()
    tracker.snapshot_start_of_stage("Preparation", ["batch"])
    tracker.snapshot_start_of_stage("Generation", ["batch", "sampling_params"])
    assert calls == [(connected, True, 5)]
    assert "Object store stats for " + connected in capsys.readouterr().out
    assert tracker.data_points[0].new_variables == ["sampling_params"]
    assert tracker.data_points[0].memory_used_after_stage_gb is not None
    assert tracker.data_points[1].stage == "Generation"


def test_tracker_does_not_cache_previous_connection(ray_boundary, monkeypatch):
    context, calls = ray_boundary
    tracker = load_tracker(monkeypatch).MemoryTracker()
    tracker.snapshot_start_of_stage("Preparation", [])
    tracker.snapshot_start_of_stage("Generation", [])
    context.gcs_address = ADDRESSES[1]
    tracker.snapshot_start_of_stage("Training", [])
    assert [call[0] for call in calls] == [ADDRESSES[0], ADDRESSES[1]]


def test_real_connected_cluster_errors_are_not_hidden(ray_boundary, monkeypatch):
    context, _ = ray_boundary
    context.gcs_address = "unavailable:9999"
    tracker = load_tracker(monkeypatch).MemoryTracker()
    tracker.snapshot_start_of_stage("Preparation", [])
    with pytest.raises(ConnectionError, match="Connected cluster unavailable"):
        tracker.snapshot_start_of_stage("Generation", [])
