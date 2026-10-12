"""Read-only runtime qualification cache. No GPU/library imports on cache hits."""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1


def digest(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def file_state(path):
    path = Path(path)
    try:
        info = path.stat()
        return [str(path.resolve()), info.st_size, info.st_mtime_ns, info.st_ctime_ns]
    except FileNotFoundError:
        return [str(path), None]


def fingerprint(root=ROOT, environments=None):
    root = Path(root)
    environments = environments or dict(
        base=sys.executable,
        mcore=os.environ.get("DFLASH_MCORE_PYTHON", root / ".venv-mcore/bin/python"),
        vllm=os.environ.get("DFLASH_VLLM_PYTHON", root / ".venv-vllm/bin/python"),
    )
    state = dict(
        schema=SCHEMA,
        root=str(root.resolve()),
        machine=platform.machine(),
        host=platform.node(),
        driver=digest("/proc/driver/nvidia/version"),
    )
    state["files"] = {
        str(p.relative_to(root)): digest(p)
        for p in [
            root / "versions.lock",
            root / "patches/nemo-rl.patch",
            root / "patches/vllm.patch",
            root / "scripts/validate_environment.py",
            root / "scripts/runtime_artifacts.py",
            root / "scripts/patch_upstream.py",
            root / "scripts/runtime_checks.py",
            root / "dflash_reflex/worker_runtime.py",
            root / "third_party/nemo-rl/uv.lock",
        ]
    }
    state["sources"] = {}
    for name in ("nemo-rl", "vllm", "original-dflash"):
        repo = root / "third_party" / name
        head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        changed = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "HEAD", "--name-only", "-z"]
        )
        files = [p for p in changed.decode().split("\0") if p]
        # Hash only changed upstream files, including installed local patches.
        submodules = subprocess.check_output(
            ["git", "-C", str(repo), "submodule", "status", "--recursive"]
        ).decode()
        state["sources"][name] = dict(
            head=head, submodules=submodules, modified={p: digest(repo / p) for p in files}
        )
    probe = "import json,sys,sysconfig; print(json.dumps([sys.version,sys.prefix,sorted(set([sysconfig.get_path('purelib'),sysconfig.get_path('platlib')]))]))"
    state["environments"] = {}
    for name, python in environments.items():
        # Starting Python with stdlib only is cheap; no torch/vLLM/MCore import.
        version, prefix, sites = json.loads(
            subprocess.check_output([str(python), "-c", probe], text=True)
        )
        files = [file_state(python), file_state(Path(prefix) / "pyvenv.cfg")]
        patches = {}
        for site_name in sites:
            site = Path(site_name)
            files.append(file_state(site))
            for pattern in (
                "*.dist-info/METADATA",
                "*.dist-info/RECORD",
                "*.dist-info/direct_url.json",
                "*.pth",
                "*.so",
                "torch/lib/*.so*",
                "vllm/*.so*",
                "flash_attn*/**/*.so*",
                "transformer_engine*/**/*.so*",
            ):
                files.extend(file_state(p) for p in sorted(site.glob(pattern)))
            for relative in (
                "vllm/v1/spec_decode/llm_base_proposer.py",
                "vllm/v1/spec_decode/dflash.py",
                "vllm/v1/sample/rejection_sampler.py",
                "vllm/v1/worker/gpu_model_runner.py",
            ):
                patches[str(site / relative)] = digest(site / relative)
        state["environments"][name] = dict(version=version, files=files, patches=patches)
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), state


@contextmanager
def cache_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def read_stamp(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def write_stamp(directory, key, state):
    temporary = directory / f"validation.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(dict(fingerprint=key, state=state), sort_keys=True) + "\n")
    temporary.replace(directory / "validation.json")


def qualify(mode, directory, *, identify=fingerprint, validate=None):
    """Serialize qualification across simultaneous launches; cache only success."""
    if mode == "skip":
        print(
            "Runtime check: skip (explicit; checkpoint/config checks remain enabled).", flush=True
        )
        return
    if validate is None:

        def validate():
            return subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/validate_environment.py"),
                    "--require-runtime",
                    "--no-runtime-cache",
                    *(["--skip-reference-audit"] if mode == "auto" else []),
                ],
                check=True,
            )

    with cache_lock(directory):
        key, state = identify()
        if mode == "auto" and read_stamp(directory / "validation.json").get("fingerprint") == key:
            print("Runtime check: auto cache hit; heavy imports skipped.", flush=True)
            return
        print(f"Runtime check: {mode}; validating all runtime environments once.", flush=True)
        validate()
        after, _ = identify()
        if after != key:
            raise RuntimeError(
                "Environment changed during validation; successful result was not cached"
            )
        write_stamp(directory, key, state)


def record_setup_validation():
    """Called only after successful explicit setup/full validation."""
    directory = ROOT / ".cache/runtime_checks"
    with cache_lock(directory):
        key, state = fingerprint()
        write_stamp(directory, key, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("auto", "full", "skip"), default="auto")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/runtime_checks")
    args = parser.parse_args()
    qualify(args.mode, args.cache_dir)


if __name__ == "__main__":
    main()
