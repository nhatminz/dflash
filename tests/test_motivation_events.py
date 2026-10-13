import copy

import pytest

from analysis.verification_feedback.events import (
    accepted_prefix,
    reconstruct,
    validate_reconstruction,
)


def request(response=0, prompt="p", stop=(9,)):
    return dict(
        response_id=response,
        prompt_id=prompt,
        question_sha256=prompt,
        sample_index=response,
        stop_token_ids=list(stop),
    )


def packet(ids, drafts, outputs, lengths=None, discard=None):
    return dict(
        request_ids=ids,
        lengths=lengths if lengths is not None else [len(drafts) // len(ids)] * len(ids),
        draft_ids=drafts,
        outputs=outputs,
        discard=discard or [False] * len(ids),
    )


def response(tokens, finish="stop"):
    return dict(token_ids=tokens, finish_reason=finish, stop_reason=None)


def test_actual_rejected_prefix_and_bonus_reconstruct():
    requests = {"a": request()}
    packets = [
        packet(["a"], [], [[1]], lengths=[0]),
        packet(["a"], [2, 3, 4], [[2, 5, -1, -1]]),
        packet(["a"], [6, 7, 8], [[6, 7, 8, 9]]),
    ]
    output = {"a": response([1, 2, 5, 6, 7, 8, 9])}
    events, rounds = reconstruct(packets, requests, output, 250, 0)
    assert [e["draft_proposed_token_id"] for e in events] == [None, 2, 3, 6, 7, 8, None]
    assert [e["accepted_or_rejected"] for e in events] == [
        None,
        "accepted",
        "rejected",
        "accepted",
        "accepted",
        "accepted",
        None,
    ]
    assert [e["is_miss"] for e in events] == [None, False, True, False, False, False, None]
    assert rounds[0]["accepted_prefix_length"] == 1
    assert rounds[0]["visited_proposal_count"] == 2
    assert validate_reconstruction(events, output)


def test_dynamic_request_reordering_and_discarded_prefill():
    requests = {"a": request(0, "p0"), "b": request(1, "p1")}
    packets = [
        packet(["a", "b"], [], [[1], [2]], [0, 0], [True, False]),
        packet(["b", "a"], [3, 4], [[9, -1], [4, 9]], [1, 1]),
    ]
    outputs = {"a": response([4, 9]), "b": response([2, 9])}
    events, _ = reconstruct(packets, requests, outputs, 600, 3)
    assert [e["response_id"] for e in events if e["request_id"] == "a"] == [0, 0]
    assert [e["target_selected_token_id"] for e in events if e["request_id"] == "b"] == [2, 9]
    assert events[0]["active_response_count"] == 1


def test_eos_and_length_cap_exclude_later_sampled_slots():
    req = {"a": request()}
    packets = [packet(["a"], [2, 9, 4], [[2, 9, 4, 5]])]
    events, _ = reconstruct(packets, req, {"a": response([2, 9])}, 250, 1)
    assert len(events) == 2
    assert events[-1]["termination_or_censor_reason"] == "EOS"
    assert events[-1]["is_miss"] is None
    events, _ = reconstruct(
        [packet(["a"], [2, 3, 4], [[2, 6, -1, -1]])], req, {"a": response([2, 6], "length")}, 250, 1
    )
    assert events[-1]["termination_or_censor_reason"] == "length_cap"
    assert events[-1]["is_miss"] is None


def test_singleton_recovery_never_uses_equality_as_rejection_label():
    assert accepted_prefix([4, 8, -1, -1], 3)[0] == 1
    # Shape labels rejection at slot 1; equal tokens there are inconsistent
    # with singleton recovery, so fail rather than relabel it accepted.
    with pytest.raises(ValueError, match="semantics changed"):
        reconstruct(
            [packet(["a"], [4, 8, 2], [[4, 8, -1, -1]])],
            {"a": request()},
            {"a": response([4, 8])},
            250,
            0,
        )


def test_padding_not_an_ordinary_miss():
    events, _ = reconstruct(
        [packet(["a"], [-1, 3], [[5, -1, -1]]), packet(["a"], [], [[9]], [0])],
        {"a": request()},
        {"a": response([5, 9])},
        250,
        0,
    )
    assert events[0]["termination_or_censor_reason"] == "invalid_proposal"
    assert events[0]["is_miss"] is None


def test_engine_generation_config_eos_is_censored_even_if_not_explicit_stop_id():
    events, _ = reconstruct(
        [packet(["a"], [1], [[8, -1]])],
        {"a": request(stop=(9,))},
        {"a": response([8], "stop")},
        250,
        0,
    )
    assert events[0]["termination_or_censor_reason"] == "EOS"
    assert events[0]["is_miss"] is None


def test_invalid_alignment_and_response_mismatch_fail():
    with pytest.raises(ValueError, match="offsets"):
        reconstruct(
            [packet(["a"], [1], [[2, -1]], lengths=[2])],
            {"a": request()},
            {"a": response([2])},
            250,
            0,
        )
    with pytest.raises(ValueError, match="differ"):
        reconstruct([packet(["a"], [1], [[2, -1]])], {"a": request()}, {"a": response([3])}, 250, 0)
    with pytest.raises(ValueError, match="contiguous"):
        accepted_prefix([1, -1, 2], 2)


def test_no_mutation_of_packets_or_request_state():
    inputs = ([packet(["a"], [1], [[2, -1]])], {"a": request()}, {"a": response([2])})
    before = copy.deepcopy(inputs)
    reconstruct(*inputs, 250, 0)
    assert before == inputs
