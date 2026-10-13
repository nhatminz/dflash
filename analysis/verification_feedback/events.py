"""Decode actual sampler outputs, including first rejection, never later MASK slots."""

from collections import defaultdict

CENSORED = {"prefill", "target_only", "bonus", "EOS", "length_cap", "invalid_proposal"}


def accepted_prefix(output, draft_length):
    """Pinned sampler emits accepted prefix + recovery OR accepted prefix + bonus.

    Both kernels leave every other output slot at -1. Thus the real prefix
    length is emitted_count - 1, without comparing any token IDs or logits.
    This is also checked against an optional real kernel counter in tests.
    """
    valid = [int(v) for v in output if int(v) >= 0]
    if list(output[: len(valid)]) != valid or any(int(v) >= 0 for v in output[len(valid) :]):
        raise ValueError("Sampler output is not a contiguous emitted prefix")
    if not 1 <= len(valid) <= draft_length + 1:
        raise ValueError("Pinned sampler output count violates prefix + recovery/bonus contract")
    return len(valid) - 1, valid


def reconstruct(packets, requests, responses, step, rollout_id):
    """Join by canonical request ID, truncate using actual engine final outputs.

    The engine's final token IDs are authoritative for EOS/length truncation.
    A packet may contain GPU-sampled slots after natural termination; those
    slots are never events and never feed histories.
    """
    if set(requests) != set(responses):
        raise ValueError("Response/request identity mismatch after dynamic batching")
    tokens = {r: [] for r in requests}
    round_index = defaultdict(int)
    events, rounds = [], []
    for boundary, packet in enumerate(packets):
        ids = packet["request_ids"]
        lengths, drafts, outputs = packet["lengths"], packet["draft_ids"], packet["outputs"]
        discard = packet["discard"]
        if len(ids) != len(set(ids)) or not (
            len(ids) == len(lengths) == len(outputs) == len(discard)
        ):
            raise ValueError("Request/slot alignment failure")
        if len(drafts) != sum(lengths):
            raise ValueError("Ragged draft offsets disagree with sampler metadata")
        active = sum(not value for value in discard)
        offset = 0
        for i, request_id in enumerate(ids):
            n = int(lengths[i])
            proposed = list(map(int, drafts[offset : offset + n]))
            offset += n
            if discard[i]:
                continue
            if request_id not in requests:
                raise ValueError(f"Unknown request from vLLM worker: {request_id}")
            req = requests[request_id]
            final = responses[request_id]
            prefix, emitted = accepted_prefix(outputs[i], n)
            current = len(tokens[request_id])
            remaining = len(final["token_ids"]) - current
            if remaining < 0:
                raise ValueError("Events continue beyond a reconstructed final response")
            emitted = emitted[:remaining]
            if emitted != final["token_ids"][current : current + len(emitted)]:
                raise ValueError(
                    "Logged target-selected tokens differ from actual emitted response"
                )
            response_round = round_index[request_id] if n else -1
            # Count actual verification calls, including terminal censored contexts.
            if n and emitted:
                round_index[request_id] += 1
                rounds.append(
                    dict(
                        checkpoint_step=step,
                        rollout_id=rollout_id,
                        request_id=request_id,
                        response_id=req["response_id"],
                        prompt_id=req["prompt_id"],
                        global_verification_batch_id=boundary,
                        response_verification_round=response_round,
                        active_response_count=active,
                        draft_block_length=n,
                        valid_proposal_count=sum(v >= 0 for v in proposed),
                        accepted_prefix_length=prefix,
                        emitted_tokens=len(emitted),
                        visited_proposal_count=min(len(emitted), n),
                    )
                )
            for slot, y in enumerate(emitted):
                position = len(tokens[request_id])
                is_proposal = bool(n and slot < n and slot <= prefix)
                d = proposed[slot] if is_proposal else None
                status = ("accepted" if slot < prefix else "rejected") if is_proposal else None
                reason = (
                    "ordinary"
                    if is_proposal
                    else ("bonus" if n else ("prefill" if current == 0 else "target_only"))
                )
                terminal = position + 1 == len(final["token_ids"])
                if y in req["stop_token_ids"] or (terminal and final["finish_reason"] == "stop"):
                    if not terminal:
                        raise ValueError("Actual response continued after a stop token")
                    reason = "EOS"
                elif terminal and final["finish_reason"] == "length":
                    reason = "length_cap"
                elif is_proposal and d < 0:
                    reason = "invalid_proposal"
                # A greedy DFlash candidate is a point mass. The recovered
                # distribution excludes that ID; rejection and absence agree.
                if is_proposal and d >= 0 and ((status == "accepted") != (y == d)):
                    raise ValueError(
                        "Singleton rejection-sampler semantics changed; equality is NOT a rejection label"
                    )
                miss = (y != d) if reason == "ordinary" else None
                events.append(
                    dict(
                        checkpoint_step=step,
                        rollout_id=rollout_id,
                        prompt_id=req["prompt_id"],
                        question_sha256=req["question_sha256"],
                        response_id=req["response_id"],
                        sample_index=req["sample_index"],
                        request_id=request_id,
                        global_verification_batch_id=boundary,
                        response_verification_round=response_round,
                        token_position=position,
                        block_slot_index=slot if is_proposal else None,
                        target_selected_token_id=y,
                        draft_proposed_token_id=d,
                        accepted_or_rejected=status,
                        is_miss=miss,
                        active_response_count=active,
                        draft_block_length=n,
                        valid_proposal_count=sum(v >= 0 for v in proposed),
                        accepted_prefix_length=prefix,
                        termination_or_censor_reason=reason,
                        event_order=len(events),
                    )
                )
                tokens[request_id].append(y)
    for request_id, response in responses.items():
        if tokens[request_id] != response["token_ids"]:
            raise ValueError(f"Incomplete event reconstruction for {request_id}")
    return events, rounds


def validate_reconstruction(events, responses):
    found = {key: [] for key in responses}
    last_boundary = -1
    last_round = {}
    for event in sorted(events, key=lambda row: row["event_order"]):
        request_id = event["request_id"]
        if request_id not in found or event["token_position"] != len(found[request_id]):
            raise ValueError("Noncontiguous or unknown response event")
        reason = event["termination_or_censor_reason"]
        boundary = event["global_verification_batch_id"]
        if boundary < last_boundary:
            raise ValueError("Verification completion ordering moved backwards")
        last_boundary = boundary
        r = event["response_verification_round"]
        if r >= 0:
            if request_id in last_round and r < last_round[request_id]:
                raise ValueError("Response verification rounds moved backwards")
            last_round[request_id] = r
        if reason not in CENSORED | {"ordinary"}:
            raise ValueError("Unknown verification event reason")
        if reason == "ordinary":
            if event["is_miss"] is None:
                raise ValueError("Ordinary verification position lacks an outcome")
            if bool(event["is_miss"]) != (
                event["target_selected_token_id"] != event["draft_proposed_token_id"]
            ):
                raise ValueError("Incorrect singleton candidate-set miss")
            slot, prefix = event["block_slot_index"], event["accepted_prefix_length"]
            if slot is None or not 0 <= slot < event["draft_block_length"] or slot > prefix:
                raise ValueError("Unverified post-rejection slot entered the ordinary event log")
            status = "accepted" if slot < prefix else "rejected"
            if event["accepted_or_rejected"] != status or bool(event["is_miss"]) != (
                status == "rejected"
            ):
                raise ValueError("Sampler acceptance metadata disagrees with ordinary event")
        elif event["is_miss"] is not None:
            raise ValueError("Censored positions must not enter ordinary miss denominators")
        found[request_id].append(event["target_selected_token_id"])
    if found != {key: value["token_ids"] for key, value in responses.items()}:
        raise ValueError("Final responses differ from event reconstruction")
    return True
