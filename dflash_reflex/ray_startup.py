"""Bound standalone Ray jobs and make slow startup stages observable."""

import faulthandler
import json
import math
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


def configure_ray_environment():
    """Set Ray 2.56.1's host-sized RPC pools before importing its native code.

    num_cpus controls Ray scheduling, but these pools independently default to
    a quarter of the host's CPU limit. Eight single-GPU clusters do not need
    eight host-sized GCS/RPC pools or speculative first-driver worker pools.
    """
    for key in (
        "num_server_call_thread",
        "gcs_server_rpc_server_thread_num",
        "gcs_server_rpc_client_thread_num",
        "core_worker_num_server_call_thread",
        "worker_num_grpc_internal_threads",
    ):
        name = f"RAY_{key}"
        value = os.environ.setdefault(name, "2")
        if int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
    os.environ.setdefault("RAY_prestart_worker_first_driver", "false")


@contextmanager
def startup_stage(name):
    """Print progress even through tee; dump waiting Python stacks every 120s."""
    started = time.perf_counter()
    prefix = f"[startup pid={os.getpid()} GPU={os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}]"
    print(f"{prefix} {name}: starting", flush=True)
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        yield
    except BaseException:
        print(f"{prefix} {name}: failed after {time.perf_counter() - started:.1f}s", flush=True)
        raise
    else:
        print(f"{prefix} {name}: ready in {time.perf_counter() - started:.1f}s", flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()


def local_ray_options():
    """Allocate per-job logical resources rather than a whole host per GPU.

    Ray's CPU resources control scheduling; they are not OS CPU affinity or a
    hard limit on native library threads.
    """
    available = len(os.sched_getaffinity(0))
    cpus = int(os.environ.get("DFLASH_RAY_NUM_CPUS", min(8, available)))
    # The colocated policy + generation placement group reserves two CPUs.
    if not 2 <= cpus <= available:
        raise ValueError(f"DFLASH_RAY_NUM_CPUS must be 2..{available} for this process; got {cpus}")
    gib = float(os.environ.get("DFLASH_RAY_OBJECT_STORE_GB", "1"))
    if not math.isfinite(gib) or gib < 0.125:
        raise ValueError("DFLASH_RAY_OBJECT_STORE_GB must be finite and >= 0.125 GiB")
    return dict(
        # Never discover/attach to another GPU's job or an inherited RAY_ADDRESS.
        address="local",
        num_cpus=cpus,
        object_store_memory=int(gib * 1024**3),
        include_dashboard=False,
        log_to_driver=True,
    )


def init_local_ray(run_dir):
    """Start exactly one owned, isolated single-node cluster for this run."""
    configure_ray_environment()
    import ray

    if ray.is_initialized():
        raise RuntimeError("DFlash standalone driver already has an initialized Ray cluster")
    options = local_ray_options()
    # Preserve NeMo's removal of MPI/Slurm variables before actor env capture.
    for key in list(os.environ):
        if key.startswith(("PMIX_", "PMI_", "MPI_", "OMPI_", "SLURM_")):
            os.environ.pop(key, None)
    env_vars = dict(os.environ)
    env_vars.pop("RAY_ADDRESS", None)
    env_vars.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None)
    # Run names exceed Unix domain socket path limits; keep Ray sockets local
    # and short, and publish their location in the run's own logs directory.
    temp_dir = tempfile.mkdtemp(prefix=f"dflash-ray-{os.getuid()}-", dir="/tmp")
    log_dir = Path(run_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        driver_pid=os.getpid(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "ALL"),
        ray_temp_dir=temp_dir,
        status="starting",
        rpc_threads={
            key: value
            for key, value in os.environ.items()
            if key.startswith("RAY_") and ("thread" in key or "prestart" in key)
        },
        **options,
    )
    metadata_path = log_dir / "ray_startup.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Ray per run: CPUs={options['num_cpus']}, "
        f"object store={options['object_store_memory'] / 1024**3:g} GiB, "
        f"dashboard=off, temp={temp_dir}",
        flush=True,
    )
    try:
        ray.init(
            **options,
            runtime_env={"env_vars": env_vars},
            _temp_dir=temp_dir,
        )
        metadata.update(
            status="ready",
            gcs_address=ray.get_runtime_context().gcs_address,
            resources=ray.cluster_resources(),
        )
    except BaseException:
        metadata["status"] = "failed"
        ray.shutdown()  # Only this driver's cluster; never stop other jobs.
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Ray connected: {metadata['gcs_address']}", flush=True)
