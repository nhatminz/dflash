"""Read current calibration, or safely migrate measured legacy fused decisions."""

import json
from pathlib import Path


def load_cache(path):
    if not path:
        return {}
    cache = json.loads(Path(path).read_text())
    if cache.get("schema_version") == 2:
        return cache
    if "schema_version" in cache:
        raise ValueError("unsupported OPD_DISPATCH_CACHE schema; regenerate calibration")
    migrated = {
        "schema_version": 2,
        "gpu_thresholds": {},
        "pipelines": {},
        "migrated_from_legacy": True,
    }
    for key, entries in cache.items():
        parts = key.split("|")
        if len(parts) != 7 or not isinstance(entries, list):
            raise ValueError("invalid legacy OPD dispatch cache")
        vocab = int(parts[3])
        threshold = vocab + 1
        ordered = sorted(entries, key=lambda x: x["density"])
        for i, entry in enumerate(ordered):
            if all(e["mode"] == "dense" and e["implementation"] == "fused" for e in ordered[i:]):
                threshold = int(entry["density"] * vocab)
                break
        migrated["gpu_thresholds"][key] = {"threshold_rows": threshold}
    return migrated
