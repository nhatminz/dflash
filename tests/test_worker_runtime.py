"""Regressions for the B200 worker startup tracebacks (no CUDA required)."""

import ast
from datetime import timedelta
import multiprocessing as mp
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from dflash_reflex.worker_runtime import acquire_vllm_port, policy_cuda_device

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("allocated,visible,expected", [
    ([1], "1", 0), ([5], "5", 0), ([0], "0", 0),
    ([2], "4,2", 1), ([4.0], "4,2", 0), ([2], None, 2),
    (["GPU-test"], "GPU-test", 0),
])
def test_physical_to_visible_cuda(allocated, visible, expected):
    assert policy_cuda_device(allocated, visible) == expected


@pytest.mark.parametrize("allocated,visible", [
    ([1], "0"), ([1], ""), ([1], "-1"), ([1], "1,1"),
    ([1], "1,"), ([], "0"), ([0, 1], "0,1"),
])
def test_bad_allocation_fails_before_cuda(allocated, visible):
    with pytest.raises(RuntimeError):
        policy_cuda_device(allocated, visible)


def _constructor(relative, class_name):
    tree = ast.parse((ROOT / "third_party/nemo-rl/nemo_rl" / relative).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")


def test_actual_megatron_constructor_selects_masked_gpu(monkeypatch):
    constructor = _constructor(
        "models/policy/workers/megatron_policy_worker.py", "MegatronPolicyWorkerImpl"
    )
    start = next(i for i, n in enumerate(constructor.body)
                 if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "gpu_ids")
    stop = next(i for i, n in enumerate(constructor.body)
                if isinstance(n, ast.Expr) and ast.unparse(n).startswith("torch.cuda.set_device("))
    calls = []

    def set_device(ordinal):
        # Reproduce a process with CUDA_VISIBLE_DEVICES=1: exactly one CUDA device.
        if ordinal != 0:
            raise RuntimeError("CUDA error: invalid device ordinal")
        calls.append(ordinal)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    namespace = {"os": os, "ray": SimpleNamespace(get_gpu_ids=lambda: [1]),
                 "torch": SimpleNamespace(cuda=SimpleNamespace(set_device=set_device))}
    exec(compile(ast.Module(body=constructor.body[start:stop + 1], type_ignores=[]),
                 "megatron_cuda_startup", "exec"), namespace)
    assert calls == [0]
    assert os.environ["LOCAL_RANK"] == "0"
    numa = next(n for n in constructor.body if isinstance(n, ast.Expr)
                and ast.unparse(n).startswith("bind_to_gpu_numa("))
    physical = []
    namespace["bind_to_gpu_numa"] = physical.append
    exec(compile(ast.Module(body=[numa], type_ignores=[]), "numa_binding", "exec"), namespace)
    assert physical == [1]


def test_actual_vllm_constructor_leases_before_config(monkeypatch, tmp_path):
    import dflash_reflex.worker_runtime as runtime

    constructor = _constructor(
        "models/generation/vllm/vllm_worker.py", "BaseVllmGenerationWorker"
    )
    original = acquire_vllm_port
    monkeypatch.setattr(runtime, "acquire_vllm_port", lambda: original(
        low=21000, high=21999, width=8, lock_dir=tmp_path
    ))
    monkeypatch.setenv("VLLM_PORT", "7000")
    observations = []
    worker = SimpleNamespace(_init_config=lambda *args: observations.append(os.environ["VLLM_PORT"]))
    namespace = dict(os=os, self=worker, bundle_indices=[0], config={},
                     fraction_of_gpus=0.5, seed=42, extra_env_vars=None,
                     ray=SimpleNamespace(get_gpu_ids=lambda: [1]),
                     bind_to_gpu_numa=lambda gpu: None, init_telemetry_worker=lambda: None)
    statements = []
    for node in constructor.body:
        if isinstance(node, ast.ImportFrom) and node.module == "nemo_rl.distributed.numa_utils":
            continue
        statements.append(node)
        if isinstance(node, ast.Expr) and ast.unparse(node).startswith("self._init_config("):
            break
    exec(compile(ast.Module(body=statements, type_ignores=[]), "vllm_startup", "exec"), namespace)
    assert observations == [str(worker._dflash_vllm_port_lease.port)]
    other = original(low=21000, high=21999, width=8, lock_dir=tmp_path)
    assert other.port != worker._dflash_vllm_port_lease.port
    other.close()
    worker._dflash_vllm_port_lease.close()


def test_busy_window_skipped_and_lock_reusable(tmp_path):
    with socket.socket() as occupied:
        occupied.bind(("", 0))
        low = occupied.getsockname()[1]
        lease = acquire_vllm_port(low=low, high=low + 19, width=2, lock_dir=tmp_path)
        assert lease.port >= low + 2
        first = lease.port
        lease.close()
        lease.close()
        again = acquire_vllm_port(low=low, high=low + 19, width=2, lock_dir=tmp_path)
        assert again.port == first
        again.close()


def _concurrent_store(lock_dir, barrier, results):
    import torch.distributed as dist

    lease = acquire_vllm_port(low=23000, high=25999, width=8, lock_dir=lock_dir)
    try:
        # All jobs choose their ports BEFORE any TCPStore binds. A socket
        # probe alone permits duplicate choices in exactly this interval.
        barrier.wait(timeout=20)
        store = dist.TCPStore("127.0.0.1", lease.port, 1, True, timedelta(seconds=10))
        store.set("ready", "yes")
        assert store.get("ready") == b"yes"
        results.put((os.getpid(), lease.port))
        barrier.wait(timeout=20)
        del store
    finally:
        lease.close()


def test_six_concurrent_jobs_bind_real_torch_stores(tmp_path):
    # Linux is the supported production platform. Fork keeps the test small.
    import torch.distributed  # preload before fork; no CUDA context is created

    ctx = mp.get_context("fork")
    results = ctx.Queue()
    barrier = ctx.Barrier(6)
    processes = [ctx.Process(target=_concurrent_store, args=(str(tmp_path), barrier, results))
                 for _ in range(6)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        ports = [results.get(timeout=3)[1] for _ in processes]
        assert len(set(ports)) == 6
        assert all(abs(a - b) >= 8 for a in ports for b in ports if a != b)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def _hold_until_killed(lock_dir, ready):
    lease = acquire_vllm_port(low=26000, high=26999, width=8, lock_dir=lock_dir)
    ready.send(lease.port)
    ready.recv()
    lease.close()


def test_actor_death_releases_lease(tmp_path):
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_hold_until_killed, args=(str(tmp_path), child))
    process.start()
    try:
        assert parent.poll(10)
        port = parent.recv()
        process.terminate()
        process.join(timeout=10)
        lease = acquire_vllm_port(low=26000, high=26999, width=8, lock_dir=tmp_path)
        assert lease.port == port
        lease.close()
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        parent.close()
        child.close()
