import numpy as np
import pandas as pd
import pytest

from analysis.verification_feedback.statistics import (
    ClusterBootstrap,
    add_controls,
    analyze_frames,
    annotate,
    comparison,
    distance_bin,
)


def event(
    boundary, response, prompt, y=7, miss=True, round_index=None, reason="ordinary", rollout=0
):
    return dict(
        checkpoint_step=250,
        rollout_id=rollout,
        response_id=response,
        prompt_id=prompt,
        question_sha256=prompt,
        global_verification_batch_id=boundary,
        response_verification_round=boundary if round_index is None else round_index,
        token_position=boundary,
        block_slot_index=0,
        draft_block_length=15,
        valid_proposal_count=15,
        active_response_count=64,
        target_selected_token_id=y,
        draft_proposed_token_id=1 if miss else y,
        is_miss=miss,
        termination_or_censor_reason=reason,
    )


def test_same_batch_and_same_response_slots_are_simultaneous():
    rows = [
        event(0, 0, "p"),
        event(0, 1, "p"),
        event(0, 0, "p"),
        event(1, 0, "p"),
        event(1, 2, "p"),
        event(1, 3, "q"),
    ]
    frame = annotate(pd.DataFrame(rows))
    assert list(frame["token_source"]) == [
        "none",
        "none",
        "none",
        "same_response",
        "sibling_response",
        "different_prompt",
    ]
    assert frame.iloc[3]["distance"] == "1"
    assert frame.iloc[4]["token_group"]
    assert not frame.iloc[5]["token_group"]


def test_distance_is_per_response_while_causality_is_global():
    rows = [event(1, 0, "p", round_index=0), event(20, 0, "p", round_index=2)]
    frame = annotate(pd.DataFrame(rows))
    assert frame.iloc[-1]["distance"] == "2–4"
    assert frame.iloc[-1]["same_y_exposure"] == 1
    for value, expected in ((1, "1"), (4, "2–4"), (16, "5–16"), (64, "17–64"), (65, ">64")):
        assert distance_bin(value) == expected
    with pytest.raises(ValueError):
        distance_bin(0)


def test_availability_denominators_censoring_and_inclusive_prompt():
    rows = [
        event(0, 0, "p"),
        event(1, 0, "p"),
        event(2, 1, "p"),
        event(3, 1, "p", miss=False),
        event(4, 2, "q", miss=None, reason="EOS"),
    ]
    result = analyze_frames({0: annotate(pd.DataFrame(rows))}, 40)
    same = result["fig1c"]["within_response_prior_miss_share"]
    group = result["fig1c"]["within_prompt_prior_miss_share"]
    assert same["numerator"] == 1 and same["denominator"] == 3
    assert group["numerator"] == 2 and group["denominator"] == 3
    assert same["ci95"] == [None, None]
    assert result["ordinary_events"] == 4


def test_donor_recipient_halves_no_reuse_no_future():
    frames = {
        i: annotate(
            pd.DataFrame([event(2, 0, f"p{i}", rollout=i), event(3, 0, f"p{i}", rollout=i)])
        )
        for i in range(5)
    }
    cross, pairs = add_controls(frames)
    assert pairs == [dict(donor=0, recipient=2), dict(donor=1, recipient=3)]
    assert set(cross["rollout_id"]) == {2, 3}
    assert cross.groupby("rollout_id")["cross_prior"].apply(list).to_dict() == {
        2: [False, True],
        3: [False, True],
    }
    assert all(cross.loc[cross["global_verification_batch_id"] == 2, "cross_exposure_bin"] == 0)
    frames[2]["question_sha256"] = "p0"
    with pytest.raises(ValueError, match="overlapping"):
        add_controls(frames)


def test_whole_rollout_bootstrap_not_token_or_response_bootstrap():
    boot = ClusterBootstrap([0, 1, 2], 200, 9)
    assert boot.weights.shape == (200, 3)
    assert np.all(boot.weights.sum(1) == 3)
    assert np.array_equal(boot.weights, ClusterBootstrap([2, 1, 0], 200, 9).weights)
    frame = pd.DataFrame(dict(rollout_id=[0] * 20 + [1] * 20, is_miss=[True] * 20 + [False] * 20))
    result = boot.ratio(frame, frame["is_miss"], np.ones(40))
    assert result["estimate"] == 0.5
    assert result["ci95"] == [0, 1]


def matched_frame():
    rows = []
    for rollout in range(3):
        for exposed in (False, True):
            for outcome in (
                [True, True, False, False] if not exposed else [True, False, False, False]
            ):
                rows.append(
                    dict(
                        rollout_id=rollout,
                        exposed=exposed,
                        is_miss=outcome,
                        y=1,
                        round_bin="2–4",
                        position_bin="0–15",
                        block_slot_index=0,
                        draft_block_length=15,
                        valid_proposal_count=15,
                        active_bin="64+",
                        exposure_bin=2,
                        same_events_exposure=1,
                        same_y_exposure=1,
                    )
                )
    return pd.DataFrame(rows)


def test_raw_and_matched_risks_can_be_below_one_no_forced_lift():
    frame = matched_frame()
    result = comparison(frame, "exposed", ClusterBootstrap([0, 1, 2], 100))
    assert result["raw"]["risk_ratio"] == 0.5
    assert result["matched"]["risk_ratio"] == 0.5
    assert result["matched"]["matched_events"] == len(frame)
    assert result["matched"]["unsupported_events"] == 0
    assert result["matched"]["risk_ratio_ci95"] == [0.5, 0.5]


def test_exact_token_common_support_is_not_imputed():
    frame = matched_frame()
    frame.loc[frame["exposed"], "y"] = 2
    result = comparison(frame, "exposed", ClusterBootstrap([0, 1, 2], 100))["matched"]
    assert result["risk_ratio"] is None
    assert result["matched_events"] == 0
    assert result["unsupported_strata"] == 2
    assert result["unsupported_events"] == len(frame)


def test_empty_observations_are_unavailable_not_zero():
    result = analyze_frames({0: pd.DataFrame()}, 20)
    assert result["fig1a"]["within_response"]["matched"]["risk_ratio"] is None
    assert result["fig1c"]["within_prompt_prior_miss_share"]["estimate"] is None
