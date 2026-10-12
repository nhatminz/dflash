import pytest

from dflash_reflex.dispatch import PIPELINE_REVISION, PipelinePolicy

IDENTITY = dict(
    gpu="B200",
    capability=(10, 0),
    hidden=2560,
    vocab=151936,
    rank=8,
    topk=16,
    width=15,
    dtype="torch.bfloat16",
    logits_dtype="torch.float32",
    update_stream=True,
)


def calibration():
    records = []
    for batch in (1, 8, 32, 64):
        for density in (1 / 15, 1.0):
            for pipeline, time in (
                ("selective_masked", 2 if density < 1 else 20),
                ("selective", 3 if density < 1 else 15),
                ("fused", 10 if density < 1 else 12),
            ):
                records.append(
                    dict(
                        requests=batch,
                        density=density,
                        pipeline=pipeline,
                        median_ms=time,
                        update_stream=True,
                    )
                )
    return dict(
        schema_version=3,
        implementation_revision=PIPELINE_REVISION,
        gpu="B200",
        capability=[10, 0],
        hidden=2560,
        vocab=151936,
        rank=8,
        topk=16,
        block_size=16,
        hidden_dtype="torch.bfloat16",
        logits_dtype="torch.float32",
        measurement_scope="runtime_complete",
        measurements=records,
    )


@pytest.mark.parametrize("batch", [1, 8, 16, 32, 47, 63, 64])
def test_dynamic_batch_nearest_density_and_long_prefix(batch):
    policy = PipelinePolicy(calibration(), **IDENTITY)
    assert policy.select(batch) == "selective_masked"
    assert policy.select(batch, 2 / 15) == "selective_masked"
    assert policy.select(batch, 1.0) == "fused"
    assert policy.buckets[batch] == {1: 1, 8: 8, 16: 32, 32: 32, 47: 64, 63: 64, 64: 64}[batch]


@pytest.mark.parametrize(
    "field,value",
    [
        ("implementation_revision", "old-kernel"),
        ("gpu", "RTX 3090"),
        ("capability", [8, 6]),
        ("hidden", 2048),
        ("vocab", 1000),
        ("rank", 4),
        ("topk", 8),
        ("block_size", 8),
        ("hidden_dtype", "torch.float32"),
        ("logits_dtype", "torch.bfloat16"),
        ("measurement_scope", "kernel_only"),
    ],
)
def test_incompatible_cache_uses_masked_fallback(field, value):
    cache = calibration()
    cache[field] = value
    policy = PipelinePolicy(cache, **IDENTITY)
    assert not policy.entries
    assert policy.select(47, 1.0) == "selective_masked"


def test_no_cache_and_stream_mismatch():
    assert PipelinePolicy({}, **IDENTITY).select(63) == "selective_masked"
    policy = PipelinePolicy(calibration(), **dict(IDENTITY, update_stream=False))
    assert not policy.entries


def test_measured_selective_winner_is_not_overridden():
    cache = calibration()
    for record in cache["measurements"]:
        if record["pipeline"] == "selective":
            record["median_ms"] = 1
    assert PipelinePolicy(cache, **IDENTITY).select(47, 0.05) == "selective"
