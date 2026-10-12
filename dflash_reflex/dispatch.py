"""Read current calibration, or safely migrate measured legacy fused decisions."""

import json
import math
from pathlib import Path

PIPELINE_REVISION = "metadata1-teacher2048-argmax1024"


def load_cache(path):
    if not path:
        return {}
    cache = json.loads(Path(path).read_text())
    if cache.get("schema_version") in {2, 3}:
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


class PipelinePolicy:
    """Compile compatible calibration once; choose using past feedback only."""

    def __init__(
        self,
        cache,
        *,
        gpu,
        capability,
        hidden,
        vocab,
        rank,
        topk,
        width,
        dtype,
        logits_dtype,
        update_stream,
    ):
        self.entries = {}
        self.curves = {}
        self.buckets = {}
        self.fallback = "selective_masked"
        self.initial_density = 1.0 / width
        if cache.get("gpu") != gpu or tuple(cache.get("capability", ())) != tuple(capability):
            return
        if cache.get("schema_version") == 3:
            if cache.get("implementation_revision") != PIPELINE_REVISION:
                return
            identity = dict(
                hidden=hidden,
                vocab=vocab,
                rank=rank,
                topk=topk,
                block_size=width + 1,
                hidden_dtype=str(dtype),
                logits_dtype=str(logits_dtype),
            )
            if any(cache.get(key) != value for key, value in identity.items()):
                return
            if cache.get("measurement_scope") != "runtime_complete":
                return
            for entry in cache.get("measurements", []):
                if (
                    entry.get("update_stream") == update_stream
                    and entry.get("pipeline") in {"selective_masked", "selective", "fused"}
                    and isinstance(entry.get("requests"), int)
                    and entry["requests"] > 0
                    and isinstance(entry.get("density"), (int, float))
                    and 0 <= entry["density"] <= 1
                    and isinstance(entry.get("median_ms"), (int, float))
                    and math.isfinite(entry["median_ms"])
                    and entry["median_ms"] > 0
                ):
                    self.entries.setdefault(entry["requests"], []).append(entry)
        self.curves = {
            n: {
                pipeline: sorted(
                    (e["density"], e["median_ms"]) for e in entries if e["pipeline"] == pipeline
                )
                for pipeline in ("selective_masked", "selective", "fused")
            }
            for n, entries in self.entries.items()
        }
        # Schema 2 measured kernels without Runtime metadata/stream costs.
        # Its sparse/dense thresholds remain supported, but pipeline selection
        # requires complete-path schema 3 measurements on this exact GPU.

    def select(self, batch, density=None):
        if not self.entries:
            return self.fallback
        calibrated = self.buckets.get(batch)
        if calibrated is None:
            calibrated = min(self.entries, key=lambda n: (abs(math.log(n / batch)), -n))
            self.buckets[batch] = calibrated
        density = self.initial_density if density is None else min(1.0, max(0.0, density))
        # Interpolate measured complete-path times in density, instead of
        # extrapolating a single winning pipeline to all acceptance lengths.
        timings = {}
        for pipeline, points in self.curves[calibrated].items():
            if not points:
                continue
            low = max((p for p in points if p[0] <= density), default=points[0])
            high = min((p for p in points if p[0] >= density), default=points[-1])
            fraction = (density - low[0]) / (high[0] - low[0]) if high[0] != low[0] else 0
            timings[pipeline] = low[1] + fraction * (high[1] - low[1])
        return min(timings, key=timings.get) if timings else self.fallback
