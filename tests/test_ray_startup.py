"""Exercise standalone Ray configuration without requiring the B200 runtime."""

import ast
import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dflash_reflex.ray_startup import (
    configure_ray_environment,
    init_local_ray,
    local_ray_options,
    startup_stage,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("DFLASH_RAY_", "RAY_")):
            monkeypatch.delenv(key)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(64)))


def test_bounded_defaults_and_explicit_overrides(monkeypatch):
    options = local_ray_options()
    assert options["address"] == "local"
    assert options["num_cpus"] == 8
    assert options["object_store_memory"] == 1024**3
    assert options["include_dashboard"] is False
    monkeypatch.setenv("DFLASH_RAY_NUM_CPUS", "4")
    monkeypatch.setenv("DFLASH_RAY_OBJECT_STORE_GB", "0.5")
    assert local_ray_options()["num_cpus"] == 4
    assert local_ray_options()["object_store_memory"] == 512 * 1024**2


@pytest.mark.parametrize("cpus", ["0", "1", "65", "-1", "junk"])
def test_bad_cpu_budget_fails_before_start(monkeypatch, cpus):
    monkeypatch.setenv("DFLASH_RAY_NUM_CPUS", cpus)
    with pytest.raises(ValueError):
        local_ray_options()


@pytest.mark.parametrize("gib", ["0", "0.01", "-1", "nan", "inf", "junk"])
def test_bad_object_store_fails_before_start(monkeypatch, gib):
    monkeypatch.setenv("DFLASH_RAY_OBJECT_STORE_GB", gib)
    with pytest.raises(ValueError):
        local_ray_options()


def test_small_affinity_budget_is_respected(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {1, 2, 3, 4})
    assert local_ray_options()["num_cpus"] == 4


def test_native_rpc_pools_are_bounded_and_overrides_survive(monkeypatch):
    monkeypatch.setenv("RAY_num_server_call_thread", "4")
    configure_ray_environment()
    assert os.environ["RAY_num_server_call_thread"] == "4"
    assert os.environ["RAY_gcs_server_rpc_server_thread_num"] == "2"
    assert os.environ["RAY_gcs_server_rpc_client_thread_num"] == "2"
    assert os.environ["RAY_core_worker_num_server_call_thread"] == "2"
    assert os.environ["RAY_worker_num_grpc_internal_threads"] == "2"
    assert os.environ["RAY_prestart_worker_first_driver"] == "false"
    monkeypatch.setenv("RAY_num_server_call_thread", "0")
    with pytest.raises(ValueError, match="positive integer"):
        configure_ray_environment()


def test_rpc_environment_is_set_before_first_nemo_import():
    tree = ast.parse((ROOT / "dflash_reflex/train.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    config_line = next(
        n.lineno
        for n in ast.walk(main)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "configure_ray_environment"
    )
    first_nemo_import = min(
        n.lineno
        for n in ast.walk(main)
        if isinstance(n, ast.ImportFrom) and n.module.startswith("nemo_rl.")
    )
    assert config_line < first_nemo_import
    assert "init_ray()" not in ast.unparse(main)


def _fake_ray(init, shutdown=lambda: None):
    return SimpleNamespace(
        is_initialized=lambda: False,
        init=init,
        shutdown=shutdown,
        get_runtime_context=lambda: SimpleNamespace(gcs_address=f"127.0.0.1:{os.getpid()}"),
        cluster_resources=lambda: {"CPU": 8.0, "GPU": 1.0},
    )


def _start_mock_cluster(root, gpu, queue):
    calls = []
    sys.modules["ray"] = _fake_ray(lambda **kwargs: calls.append(kwargs))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["RAY_ADDRESS"] = "172.0.0.1:12345"
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["PMIX_RANK"] = "0"
    run = Path(root) / str(gpu)
    init_local_ray(run)
    queue.put((calls[0], json.loads((run / "logs/ray_startup.json").read_text())))


def test_eight_concurrent_drivers_use_distinct_local_dirs_and_masks(tmp_path):
    # Real concurrent driver processes, with only Ray's native API mocked.
    # This checks our isolation contract, not live Ray/B200 startup.
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=_start_mock_cluster, args=(str(tmp_path), i, queue)) for i in range(8)
    ]
    directories = []
    try:
        for process in processes:
            process.start()
        records = [queue.get(timeout=15) for _ in processes]
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
        masks = set()
        for options, metadata in records:
            directory = Path(options["_temp_dir"])
            directories.append(directory)
            assert directory.parent == Path("/tmp")
            assert (
                len(str(directory / "session_2026-10-12_00-00-00_123456_123456/sockets/raylet"))
                < 108
            )
            assert options["address"] == "local"
            assert options["num_cpus"] == 8
            env = options["runtime_env"]["env_vars"]
            masks.add(env["CUDA_VISIBLE_DEVICES"])
            assert "RAY_ADDRESS" not in env
            assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" not in env
            assert "PMIX_RANK" not in env
            assert env["RAY_gcs_server_rpc_server_thread_num"] == "2"
            assert metadata["status"] == "ready"
            assert metadata["ray_temp_dir"] == str(directory)
        assert masks == {str(i) for i in range(8)}
        assert len(set(directories)) == 8
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        for directory in directories:
            directory.rmdir()  # The mocked Ray never created session files.
        queue.close()


def test_startup_failure_preserves_error_and_records_own_shutdown(monkeypatch, tmp_path):
    shutdown = []
    failure = RuntimeError("original Ray initialization failure")

    def fail(**kwargs):
        raise failure

    monkeypatch.setitem(sys.modules, "ray", _fake_ray(fail, lambda: shutdown.append(True)))
    with pytest.raises(RuntimeError) as raised:
        init_local_ray(tmp_path)
    assert raised.value is failure
    assert shutdown == [True]
    metadata = json.loads((tmp_path / "logs/ray_startup.json").read_text())
    assert metadata["status"] == "failed"
    Path(metadata["ray_temp_dir"]).rmdir()


@pytest.mark.parametrize("fail", [False, True])
def test_stage_trace_timer_is_cancelled_on_success_and_failure(monkeypatch, capsys, fail):
    import dflash_reflex.ray_startup as startup

    timers = []
    monkeypatch.setattr(
        startup.faulthandler,
        "dump_traceback_later",
        lambda *args, **kwargs: timers.append((args, kwargs)),
    )
    monkeypatch.setattr(
        startup.faulthandler, "cancel_dump_traceback_later", lambda: timers.append("cancel")
    )
    try:
        with startup_stage("Tokenizer loading"):
            if fail:
                raise RuntimeError("tokenizer failure")
    except RuntimeError:
        assert fail
    output = capsys.readouterr().out
    assert "Tokenizer loading: starting" in output
    assert ("failed after" if fail else "ready in") in output
    assert timers == [((120,), {"repeat": True}), "cancel"]


def test_launcher_exports_unbuffered_output_and_small_native_pools(tmp_path):
    environment = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "MODEL",
            "MODEL_ENV",
            "COMMON_ENV",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "TOKENIZERS_PARALLELISM",
        }
    }
    result = subprocess.run(
        ["bash", "-c", "source scripts/launch/common.sh; env"],
        cwd=ROOT,
        env=dict(environment, MODEL_KEY="qwen25_3b", OUTPUT_ROOT=str(tmp_path)),
        text=True,
        capture_output=True,
        check=True,
    )
    exported = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert exported["PYTHONUNBUFFERED"] == "1"
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert exported[key] == "1"
    assert exported["TOKENIZERS_PARALLELISM"] == "false"
    assert exported["DFLASH_RAY_NUM_CPUS"] == "8"
    assert exported["DFLASH_RAY_OBJECT_STORE_GB"] == "1"
