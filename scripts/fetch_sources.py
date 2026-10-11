"""Fetch exact audited commits; the lock file, never branch heads, is authority."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    versions = json.loads((ROOT / "versions.lock").read_text())
    for name in ("nemo-rl", "vllm", "original-dflash"):
        path, pin = ROOT / "third_party" / name, versions[name]
        if not (path / ".git").exists():
            if path.exists() and any(path.iterdir()):
                raise RuntimeError(f"refusing to overwrite a non-git source directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", str(path)], check=True)
            subprocess.run(
                ["git", "-C", str(path), "remote", "add", "origin", pin["url"]], check=True
            )
            subprocess.run(
                ["git", "-C", str(path), "fetch", "--depth", "1", "origin", pin["sha"]], check=True
            )
            subprocess.run(
                ["git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD"], check=True
            )
        sha = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if sha != pin["sha"]:
            raise RuntimeError(
                f"{name}: {sha} differs from lock; preserve local work and use a clean directory"
            )
    subprocess.run([sys.executable, str(ROOT / "scripts/patch_upstream.py")], check=True)


if __name__ == "__main__":
    main()
