import csv

import torch

from dflash_reflex.metrics import Metrics


def test_sums_not_average_of_ratios_and_resume(tmp_path):
    def packet(a, r, t):
        return dict(
            counters=torch.tensor([a, r, a - r, 15, 2, 0.6, 0.2, 2]),
            generated_tokens=a,
            updates=2,
            profile_ms={},
            generation_time_s=t,
        )

    logger = Metrics(tmp_path, "dflash_reflex", "simplelr")
    logger.record(0, 1, [packet(5, 2, 1)], {}, wall=1)
    state = dict(logger.state)
    logger.record(0, 2, [packet(9, 1, 1)], {}, wall=1)
    resumed = Metrics(tmp_path, "dflash_reflex", "simplelr", state)
    row = resumed.record(0, 2, [packet(4, 1, 1)], {}, wall=1)
    assert row["cumulative_aal"] == 3.0
    with (tmp_path / "logs/rollout_timing.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 2
    assert len((tmp_path / "logs/metrics.jsonl").read_text().splitlines()) == 2


def test_pretrain_log_resume_discards_future_and_partial_write(tmp_path):
    import json

    from dflash_reflex.metrics import truncate_jsonl

    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step":1}\n{"step":2}\n{"ste')
    truncate_jsonl(path, "step", 1)
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [1]


def test_final_wall_time_includes_checkpoint_drain_and_survives_resume(tmp_path, monkeypatch):
    import json

    from dflash_reflex import metrics

    tick = [0.0]
    monkeypatch.setattr(metrics.time, "perf_counter", lambda: tick[0])
    logger = Metrics(tmp_path, "dflash", "simplelr")
    packet = dict(
        counters=torch.tensor([4, 2, 2, 5, 0, 0, 0, 0]),
        generated_tokens=4,
        generation_time_s=1,
        updates=0,
        profile_ms={},
    )
    tick[0] = 10
    logger.record(0, 1, [packet], {}, wall=10)
    committed = dict(logger.state)
    tick[0] = 12  # Final checkpoint drain.
    logger.finalize()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["total_training_wall_time_s"] == 12
    assert summary["end_to_end_tokens_per_s"] == 4 / 12
    tick[0] = 100  # Downtime is excluded.
    restored = Metrics(tmp_path, "dflash", "simplelr", committed)
    tick[0] = 103
    row = restored.record(0, 2, [packet], {}, wall=3)
    assert row["cumulative_wall_time_s"] == 15


def test_skipped_iteration_keeps_latest_completed_global_step(tmp_path, monkeypatch):
    import json

    from dflash_reflex import nemo_bridge as bridge

    monkeypatch.setenv("DFLASH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("METHOD", "dflash")
    monkeypatch.setenv("DATASET", "simplelr")
    monkeypatch.setattr(bridge, "_pending_checkpoint", None)
    bridge.configure_driver()
    for batch in bridge.tracked_iterations([1, 2, 3], 0, 3):
        if batch == 1:
            bridge.record_iteration(4, {}, dict(loss=0.2, reward=0.3))
        elif batch == 3:
            bridge.record_iteration(5, {}, dict(loss=0.1, reward=0.4))
    rows = [json.loads(line) for line in (tmp_path / "logs/metrics.jsonl").read_text().splitlines()]
    assert [row["global_step"] for row in rows] == [4, 4, 5]
    assert [row["iteration"] for row in rows] == [1, 2, 3]
