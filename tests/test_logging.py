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


def test_no_verification_is_missing_aal_not_zero_acceptance(tmp_path):
    packet = dict(
        counters=torch.zeros(8),
        generated_tokens=100,
        updates=0,
        generation_time_s=1.0,
        profile_ms={},
        core_metrics_available=True,
        proposal_calls=0,
        verification_calls=0,
    )
    m = Metrics(tmp_path, "dflash_reflex", "simplelr")
    row = m.record(0, 1, [packet], {}, wall=1.0)
    assert row["iter_verification_rounds"] == 0
    assert row["iter_aal"] is row["iter_acceptance_rate"] is None
    assert row["step_aal"] is row["cumulative_aal"] is None
    # Actual zero acceptance has valid rounds/proposals and still emits a
    # recovered target token each round: AAL=1, acceptance=0.
    packet["counters"] = torch.tensor([10, 10, 0, 150, 0, 0, 0, 0])
    row = m.record(0, 2, [packet], {}, wall=1.0)
    assert row["iter_aal"] == row["cumulative_aal"] == 1
    assert row["iter_acceptance_rate"] == row["cumulative_acceptance_rate"] == 0


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


def test_production_missing_metrics_remain_null_on_resume(tmp_path):
    import json

    import torch

    from dflash_reflex.metrics import Metrics

    packet = dict(
        counters=torch.zeros(8),
        generated_tokens=5,
        generation_time_s=0.1,
        updates=0,
        profile_ms={},
        core_metrics_available=False,
    )
    m = Metrics(tmp_path, "dflash", "simplelr")
    row = m.record(0, 1, [packet], {}, wall=0.2)
    assert row["iter_aal"] is row["iter_verification_rounds"] is None
    m.finalize()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["aal"] is summary["total_verification_rounds"] is None
    state = dict(m.state)
    resumed = Metrics(tmp_path, "dflash", "simplelr", state)
    packet["core_metrics_available"] = True
    packet["counters"] = torch.tensor([5, 1, 4, 15, 0, 0, 0, 0])
    row = resumed.record(0, 2, [packet], {}, wall=0.2)
    assert row["iter_aal"] == 5 and row["cumulative_aal"] is None


def test_fused_profiling_does_not_invent_separate_normalization_or_refit_times(tmp_path):
    import json

    packet = dict(
        counters=torch.tensor([5, 1, 4, 15, 2, 0.6, 0.2, 2]),
        generated_tokens=5,
        generation_time_s=0.1,
        updates=1,
        profile_ms=dict(corrected_argmax_ms=2.0, topk_union_ms=3.0, union_kl_ms=0.5),
    )
    m = Metrics(tmp_path, "dflash_reflex", "simplelr")
    row = m.record(0, 1, [packet], {"prepare_for_generation/total": 0.01}, wall=0.2)
    assert row["opd_correction_time_s"] == 0.002
    record = json.loads((tmp_path / "logs/metrics.jsonl").read_text())
    assert record["profile_ms"]["weight_sync_ms"] == 10.0
    assert record["profile_ms"]["feedback_normalization_ms"] is None
    assert record["profile_ms"]["weight_refit_ms"] is None


def test_production_resume_wall_includes_startup_and_completed_session_drain(tmp_path, monkeypatch):
    from dflash_reflex import metrics

    tick = [10.0]  # Model/Ray setup has already consumed ten seconds.
    monkeypatch.setattr(metrics.time, "perf_counter", lambda: tick[0])
    packet = dict(
        counters=torch.zeros(8),
        generated_tokens=5,
        generation_time_s=1.0,
        updates=0,
        profile_ms={},
        core_metrics_available=False,
    )
    first = metrics.Metrics(tmp_path, "dflash", "simplelr", wall_start=0.0)
    tick[0] = 12.0
    first.record(0, 1, [packet], {}, wall=2.0)
    committed = dict(first.state)
    tick[0] = 15.0
    first.finalize()  # Completes the same committed iteration, with null rounds.
    tick[0] = 100.0
    resumed = metrics.Metrics(tmp_path, "dflash", "simplelr", committed, wall_start=98.0)
    tick[0] = 105.0
    row = resumed.record(0, 2, [packet], {}, wall=5.0)
    assert row["cumulative_wall_time_s"] == 22.0  # 15 + 2 setup + 5 iteration.
    assert row["cumulative_aal"] is None


def test_pipeline_counts_survive_logging_and_resume(tmp_path):
    import json

    packet = dict(
        counters=torch.tensor([4, 2, 2, 30, 2, 0.1, 0.2, 2]),
        generated_tokens=4,
        generation_time_s=1.0,
        updates=2,
        profile_ms={},
        pipeline_rounds=dict(fused=0, selective=1, selective_masked=1),
        selected_pipeline="selective_masked",
    )
    logger = Metrics(tmp_path, "dflash_reflex", "simplelr")
    row = logger.record(0, 1, [packet], {}, wall=1.0)
    assert row["iter_pipeline_selective_masked_rounds"] == 1
    extra = json.loads((tmp_path / "logs/metrics.jsonl").read_text().splitlines()[-1])
    assert extra["opd_pipeline_rounds"]["selective"] == 1
    logger = Metrics(tmp_path, "dflash_reflex", "simplelr", dict(logger.state))
    logger.record(0, 2, [packet], {}, wall=1.0)
    assert logger.state["pipeline_rounds"]["selective_masked"] == 2
    assert (
        json.loads((tmp_path / "summary.json").read_text())["pipeline_rounds"]["selective_masked"]
        == 2
    )
