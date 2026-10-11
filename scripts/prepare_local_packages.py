"""Stage non-PyPI artifacts for Linux x86_64, CPython 3.13, CUDA 13.

Run on the connected transfer host. This downloads wheels, never installs the
B200 runtime into that host's environment. Git-only packages retain their
exact source revisions and must be compiled on the CUDA-13 host.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("base", "mcore", "vllm")


def target_environment():
    return dict(
        default_environment(),
        sys_platform="linux",
        platform_system="Linux",
        platform_machine="x86_64",
        python_version="3.13",
        python_full_version="3.13.16",
        implementation_name="cpython",
        platform_python_implementation="CPython",
    )


def wheel_tags():
    platforms = [f"manylinux_2_{v}_x86_64" for v in range(28, 4, -1)]
    platforms += [
        "manylinux2014_x86_64",
        "manylinux2010_x86_64",
        "manylinux1_x86_64",
        "linux_x86_64",
    ]
    return list(cpython_tags((3, 13), platforms=platforms)) + list(
        compatible_tags((3, 13), interpreter="cp313", platforms=platforms)
    )


def select_wheel(package):
    priority = {tag: index for index, tag in enumerate(wheel_tags())}
    choices = []
    for wheel in package.get("wheels", []):
        filename = urllib.parse.unquote(urllib.parse.urlsplit(wheel["url"]).path.rsplit("/", 1)[-1])
        _, _, _, tags = parse_wheel_filename(filename)
        scores = [priority[tag] for tag in tags if tag in priority]
        if scores:
            choices.append((min(scores), wheel))
    if not choices:
        raise RuntimeError(f"No locked CPython-3.13 Linux-x86_64 wheel: {package['name']}")
    return min(choices, key=lambda item: item[0])[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(url, path, expected_hash=None):
    expected_hash = expected_hash.removeprefix("sha256:") if expected_hash else None
    if path.is_file() and (expected_hash is None or digest(path) == expected_hash):
        print(f"Already downloaded: {path.name}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    print(f"Downloading: {path.name}", flush=True)
    urls = [url]
    if urllib.parse.urlsplit(url).netloc == "download-r2.pytorch.org":
        urls.append(
            url.replace("https://download-r2.pytorch.org/", "https://download.pytorch.org/", 1)
        )
    response = None
    for candidate in urls:
        try:
            response = urllib.request.urlopen(candidate, timeout=120)
            break
        except OSError:
            if candidate == urls[-1]:
                raise
    with response, partial.open("wb") as stream:
        size = int(response.headers.get("Content-Length", "0"))
        if size and shutil.disk_usage(path.parent).free < size + 256 * 1024**2:
            raise RuntimeError(f"Insufficient disk space for {path.name} ({size} bytes)")
        shutil.copyfileobj(response, stream, 1024 * 1024)
    if expected_hash and digest(partial) != expected_hash:
        raise RuntimeError(f"SHA256 mismatch: {partial}; the lock was not changed")
    partial.replace(path)


def checkout(url, commit, path):
    from fetch_sources import source_head

    if source_head(path) != commit:
        if (
            path.exists()
            and any(path.iterdir())
            and not (
                (path / ".git").exists()
                and source_head(path) is None
                and not subprocess.check_output(
                    ["git", "-C", str(path), "status", "--porcelain"], text=True
                ).strip()
            )
        ):
            raise RuntimeError(f"Source missing/wrong HEAD in nonempty directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
        if not (path / ".git").exists():
            subprocess.run(["git", "init", str(path)], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "-C", str(path), "remote", "add", "origin", url], check=True)
        subprocess.run(
            ["git", "-C", str(path), "fetch", "--depth", "1", "origin", commit], check=True
        )
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "submodule", "update", "--init", "--recursive", "--depth", "1"],
        check=True,
    )


def build_plan(root, destination):
    project = root / "third_party/nemo-rl"
    lock = tomllib.loads((project / "uv.lock").read_text())
    records, sources, requirements = {}, {}, {}
    for backend in BACKENDS:
        cmd = [
            "uv",
            "export",
            "--frozen",
            "--project",
            str(project),
            "--no-emit-project",
            "--no-hashes",
            "--quiet",
        ]
        if backend != "base":
            cmd += ["--extra", backend]
        raw = subprocess.check_output(cmd, text=True)
        registry, local = [], []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-e "):
                path = (project / line[3:]).resolve()
                local.append("-e ./" + path.relative_to(root).as_posix())
                continue
            req = Requirement(line)
            if req.marker and not req.marker.evaluate(target_environment()):
                continue
            candidates = [
                p
                for p in lock["package"]
                if canonicalize_name(p["name"]) == canonicalize_name(req.name)
                and (
                    not req.specifier
                    or req.specifier.contains(p.get("version", "0"), prereleases=True)
                )
            ]
            if req.url:
                if req.url.startswith("git+"):
                    package = next(
                        p
                        for p in candidates
                        if "git" in p["source"]
                        and urllib.parse.urlsplit(p["source"]["git"]).fragment in req.url
                    )
                    source = urllib.parse.urlsplit(package["source"]["git"])
                    path = destination / "sources" / package["name"]
                    sources[package["name"]] = dict(
                        url=urllib.parse.urlunsplit(
                            (source.scheme, source.netloc, source.path, "", "")
                        ),
                        commit=source.fragment,
                        version=package["version"],
                        path=path.relative_to(root).as_posix(),
                    )
                    subdir = urllib.parse.parse_qs(source.query).get("subdirectory", [""])[0]
                    local.append("./" + (path / subdir).relative_to(root).as_posix())
                    continue
                package = next(p for p in candidates if p["source"].get("url") == req.url)
            else:
                package = next(p for p in candidates if "registry" in p["source"])
                if package["source"]["registry"] in {
                    "https://pypi.org/simple",
                    "https://pypi.python.org/simple",
                }:
                    registry.append(str(req).split(";", 1)[0].strip())
                    continue
            wheel = select_wheel(package)
            name = urllib.parse.unquote(urllib.parse.urlsplit(wheel["url"]).path.rsplit("/", 1)[-1])
            path = destination / "wheels" / name
            records[name] = dict(
                wheel, path=path.relative_to(root).as_posix(), package=package["name"]
            )
            registry.append("./" + path.relative_to(root).as_posix())
        requirements[backend] = dict(registry=registry, sources=local)
    return dict(
        target="linux-x86_64-cpython313-cu130",
        wheels=list(records.values()),
        sources=sources,
        requirements=requirements,
        lock_sha256=digest(project / "uv.lock"),
        ready=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--wheels-only", action="store_true", help="Defer Git source materialization"
    )
    args = parser.parse_args()
    destination = ROOT / "offline_packages"
    destination.mkdir(exist_ok=True)
    plan = build_plan(ROOT, destination)
    manifest = destination / "manifest.json"
    manifest.write_text(json.dumps(plan, indent=2) + "\n")
    if args.plan_only:
        print(
            f"Plan: {len(plan['wheels'])} external wheels, {len(plan['sources'])} Git dependencies"
        )
        return
    for record in plan["wheels"]:
        download(record["url"], ROOT / record["path"], record["hash"])
    # Ship a relocatable interpreter instead of downloading Python on B200.
    release = json.load(
        urllib.request.urlopen(
            "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest",
            timeout=60,
        )
    )
    asset = next(
        a
        for a in release["assets"]
        if a["name"].startswith("cpython-3.13.")
        and a["name"].endswith("x86_64-unknown-linux-gnu-install_only.tar.gz")
        and int(a["name"].split("+")[0].split(".")[-1]) >= 14
    )
    python_tar = destination / asset["name"]
    download(asset["browser_download_url"], python_tar, asset.get("digest"))
    plan["python"] = dict(
        path=python_tar.relative_to(ROOT).as_posix(),
        sha256=digest(python_tar),
        url=asset["browser_download_url"],
    )
    manifest.write_text(json.dumps(plan, indent=2) + "\n")
    if not args.wheels_only:
        subprocess.run([sys.executable, str(ROOT / "scripts/fetch_sources.py")], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT / "third_party/nemo-rl"),
                "submodule",
                "update",
                "--init",
                "--recursive",
                "--depth",
                "1",
            ],
            check=True,
        )
        for name, source in plan["sources"].items():
            print(f"Staging Git source: {name}", flush=True)
            checkout(source["url"], source["commit"], ROOT / source["path"])
        plan["ready"] = True
    for backend, items in plan["requirements"].items():
        for kind, lines in items.items():
            (destination / f"{backend}.{kind}.txt").write_text("\n".join(lines) + "\n")
    manifest.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Staged artifacts: {destination}; ready={plan['ready']}", flush=True)


if __name__ == "__main__":
    main()
