"""Actor startup helpers; deliberately avoid importing Torch or initializing CUDA."""

import fcntl
import os
from pathlib import Path
import socket
import tempfile


def policy_cuda_device(gpu_ids, visible_devices):
    """Translate Ray's physical GPU allocation into a CUDA-visible ordinal.

    MegatronPolicyWorker requests one GPU and disables Ray's automatic CUDA
    mask rewrite. A driver mask such as CUDA_VISIBLE_DEVICES=1 is therefore
    inherited, and physical GPU 1 must be selected as CUDA device 0.
    """
    if len(gpu_ids) != 1:
        raise RuntimeError(f"Expected one policy GPU, got Ray allocation {gpu_ids!r}")
    physical = str(gpu_ids[0])
    # Ray also returns integer GPU IDs as floats in some versions.
    if physical.endswith(".0") and physical[:-2].isdigit():
        physical = physical[:-2]
    if visible_devices is None:
        if physical.isdigit():
            return int(physical)
        raise RuntimeError(f"Cannot resolve GPU {physical!r} without CUDA_VISIBLE_DEVICES")
    visible = [token.strip() for token in visible_devices.split(",")]
    if not all(visible) or "-1" in visible or len(set(visible)) != len(visible):
        raise RuntimeError(f"Invalid CUDA_VISIBLE_DEVICES={visible_devices!r}")
    if physical in visible:
        return visible.index(physical)
    raise RuntimeError(
        f"Ray allocated physical GPU {physical!r}, which is absent from "
        f"CUDA_VISIBLE_DEVICES={visible_devices!r}"
    )


class VllmPortLease:
    """Hold a node-local advisory lock until the owning actor exits."""

    def __init__(self, port, lock):
        self.port = port
        self._lock = lock

    def close(self):
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def __del__(self):
        self.close()


def acquire_vllm_port(*, low=7000, high=8999, width=100, lock_dir=None):
    """Lease a disjoint port window for one single-GPU vLLM engine.

    NeMo's engine slots are relative to each placement group, so concurrent
    drivers can both pick slot zero/port 7000. Holding a file lock for the
    actor's lifetime closes that race between cooperating DFlash jobs. Probe
    the entire window to avoid ports already held by other applications.
    Keep the standard NeMo range outside the OS ephemeral port range.
    """
    if not (0 < low <= high < 65536 and 0 < width <= high - low + 1):
        raise ValueError("Invalid vLLM port range/window")
    directory = Path(lock_dir) if lock_dir is not None else Path(
        tempfile.gettempdir()
    ) / f"dflash-vllm-ports-{os.getuid()}"
    directory.mkdir(parents=True, exist_ok=True)
    for base in range(low, high - width + 2, width):
        # Never unlink lock files: that would let two processes lock different
        # inodes for the same port window. Kernel locks are released on exit.
        lock = (directory / f"{base}.lock").open("a+b")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        try:
            for port in range(base, base + width):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.bind(("", port))
        except OSError:
            lock.close()
            continue
        return VllmPortLease(base, lock)
    raise RuntimeError(
        f"No free vLLM port window in {low}..{high} (width={width}); "
        "check active training jobs/listening services"
    )
