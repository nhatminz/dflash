"""Fetch exact audited commits; the lock file, never branch heads, is authority."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_head(path):
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def verify_submodules(path):
    result = subprocess.run(
        ["git", "-C", str(path), "submodule", "status", "--recursive"],
        capture_output=True,
        text=True,
        check=True,
    )
    missing = [line for line in result.stdout.splitlines() if line.startswith(("-", "+", "U"))]
    if missing:
        raise RuntimeError(
            "Offline sources have missing or mismatched recursive submodules:\n"
            + "\n".join(missing)
            + "\nOn a connected host run git submodule update --init --recursive, "
            "then transfer the complete third_party directory including nested .git metadata."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    offline = args.offline or any(
        os.environ.get(key, "").lower() in {"1", "true", "yes"}
        for key in ("DFLASH_OFFLINE", "UV_OFFLINE")
    )
    versions = json.loads((ROOT / "versions.lock").read_text())
    for name in ("nemo-rl", "vllm", "original-dflash"):
        path, pin = ROOT / "third_party" / name, versions[name]
        sha = source_head(path)
        if sha is None and offline:
            raise RuntimeError(
                f"Offline source missing or incomplete: {path}\n"
                f"Required commit: {pin['sha']}. No network fetch was attempted.\n"
                "Transfer a complete prepared source bundle; see OFFLINE.md. "
                "An empty .git left by a failed fetch is insufficient."
            )
        if sha is None and not (path / ".git").exists():
            if path.exists() and any(path.iterdir()):
                raise RuntimeError(f"refusing to overwrite a non-git source directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", str(path)], check=True)
            subprocess.run(
                ["git", "-C", str(path), "remote", "add", "origin", pin["url"]], check=True
            )
        if sha is None:
            # A failed first fetch can leave an initialized, empty repository.
            # Recover only that case; preserve user files and existing commits.
            dirty = subprocess.check_output(
                ["git", "-C", str(path), "status", "--porcelain"], text=True
            ).strip()
            if dirty:
                raise RuntimeError(f"refusing to populate a nonempty source repository: {path}")
            subprocess.run(
                ["git", "-C", str(path), "fetch", "--depth", "1", pin["url"], pin["sha"]],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD"], check=True
            )
        sha = source_head(path)
        if sha != pin["sha"]:
            raise RuntimeError(
                f"{name}: {sha} differs from lock; preserve local work and use a clean directory"
            )
    if offline:
        verify_submodules(ROOT / "third_party/nemo-rl")
    subprocess.run([sys.executable, str(ROOT / "scripts/patch_upstream.py")], check=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Source setup failed: {error}", file=sys.stderr)
        raise SystemExit(2) from None
