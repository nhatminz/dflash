"""Qualify the locked FlashAttention binary independently of wheel naming."""

import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
import platform
import tomllib
import urllib.parse
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

from packaging.utils import parse_wheel_filename
from packaging.version import Version


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def qualify_flash_attn(root, expected_version, distribution=None):
    """Accept upstream's base metadata only with proof of the pinned binary.

    The locked release wheel has +cu13torch2.10... in its filename but 2.8.1
    in METADATA. Check the local wheel's locked SHA and installed RECORD
    payload, or exact URL/hash provenance for a direct uv install.
    """
    root = Path(root)
    dist = distribution or metadata.distribution("flash-attn")
    lock = tomllib.loads((root / "third_party/nemo-rl/uv.lock").read_text())
    wheels = [
        wheel
        for package in lock["package"]
        if package["name"] == "flash-attn"
        and Version(package["version"]) == Version(expected_version)
        for wheel in package.get("wheels", [])
        if urllib.parse.unquote(wheel["url"]).endswith(f"-linux_{platform.machine()}.whl")
    ]
    if len(wheels) != 1:
        raise RuntimeError("Cannot identify the pinned FlashAttention wheel for this platform")
    wheel = wheels[0]
    filename = urllib.parse.unquote(urllib.parse.urlsplit(wheel["url"]).path.rsplit("/", 1)[-1])
    if parse_wheel_filename(filename)[1] != Version(expected_version):
        raise RuntimeError("FlashAttention wheel filename differs from versions.lock")
    expected_hash = wheel["hash"].removeprefix("sha256:")
    result = dict(
        metadata_version=dist.version,
        locked_version=expected_version,
        wheel=filename,
        sha256=expected_hash,
    )
    path = root / "offline_packages/wheels" / filename
    if path.is_file():
        if file_sha256(path) != expected_hash:
            raise RuntimeError(f"FlashAttention wheel SHA256 mismatch: {path}")
        with zipfile.ZipFile(path) as archive:
            info = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            embedded_version = Parser().parsestr(archive.read(info).decode())["Version"]
            if Version(dist.version) != Version(embedded_version):
                raise RuntimeError(
                    f"FlashAttention metadata={dist.version}, locked wheel metadata={embedded_version}"
                )
            rows = csv.reader(
                io.StringIO(archive.read(info.rsplit("/", 1)[0] + "/RECORD").decode())
            )
            verified = 0
            extension_verified = False
            for name, digest, size in rows:
                if not digest:
                    continue  # RECORD itself is rewritten by the installer.
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts or ".data" in relative.parts:
                    raise RuntimeError(f"Unexpected FlashAttention wheel layout: {name}")
                installed = Path(dist.locate_file(name))
                algorithm, encoded = digest.split("=", 1)
                if algorithm != "sha256":
                    raise RuntimeError(f"Unexpected FlashAttention RECORD hash: {algorithm}")
                actual = (
                    base64.urlsafe_b64encode(bytes.fromhex(file_sha256(installed)))
                    .decode()
                    .rstrip("=")
                    if installed.is_file() and installed.stat().st_size == int(size)
                    else None
                )
                if actual != encoded:
                    raise RuntimeError(
                        f"Installed FlashAttention differs from the locked wheel: {name}"
                    )
                verified += 1
                extension_verified |= name.startswith("flash_attn_2_cuda.") and name.endswith(".so")
            if not extension_verified:
                raise RuntimeError("Locked FlashAttention wheel has no verified CUDA extension")
        result.update(verification="locked-wheel-record", files_verified=verified)
        return result
    # uv sync installs directly from the locked URL without a staged wheel.
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    archive_info = direct.get("archive_info", {})
    hashes = archive_info.get("hashes", {})
    digest = hashes.get("sha256") or archive_info.get("hash", "").removeprefix("sha256=")
    if (
        urllib.parse.unquote(direct.get("url", "")) != urllib.parse.unquote(wheel["url"])
        or digest != expected_hash
        or Version(dist.version)
        not in (Version(expected_version), Version(Version(expected_version).public))
    ):
        raise RuntimeError(
            "Cannot prove the installed FlashAttention is the locked CUDA-13 wheel. "
            f"Restore {path}, or install the exact locked URL with SHA256 provenance. "
            f"Installed metadata={dist.version}; expected artifact={expected_version}"
        )
    result["verification"] = "locked-direct-url-sha256"
    return result
