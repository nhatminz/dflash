"""Atomic analysis artifacts, immutable identities and process locks."""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    atomic_bytes(
        path,
        (
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        ).encode(),
    )


def read_json(path):
    return json.loads(Path(path).read_text())


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Analysis identity changed: {path}; use a separate output directory")
    else:
        atomic_json(path, value)


@contextmanager
def exclusive_lock(path):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another analysis process owns {path}") from error
        yield


def files_snapshot(root):
    root = Path(root)
    result = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            s = p.stat()
            result[str(p.relative_to(root))] = [s.st_size, s.st_mtime_ns, s.st_ctime_ns]
    return result


def verify_files(root, hashes):
    for name, expected in hashes.items():
        p = Path(root) / name
        if not p.is_file() or file_hash(p) != expected:
            raise ValueError(f"Analysis artifact missing or corrupt: {p}")
