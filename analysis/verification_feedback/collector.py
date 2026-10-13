"""Frozen vLLM inference only; atomic rollout commits and exact replay audit."""

import copy
import os
import time
from pathlib import Path

from .events import reconstruct, validate_reconstruction
from .io import atomic_json, digest, file_hash, immutable_json, read_json, verify_files

EVENT_TYPES = {
    "checkpoint_step": "int64",
    "rollout_id": "int64",
    "prompt_id": "string",
    "question_sha256": "string",
    "response_id": "int64",
    "sample_index": "int64",
    "request_id": "string",
    "global_verification_batch_id": "int64",
    "response_verification_round": "int64",
    "token_position": "int64",
    "block_slot_index": "int64",
    "target_selected_token_id": "int64",
    "draft_proposed_token_id": "int64",
    "accepted_or_rejected": "string",
    "is_miss": "bool",
    "active_response_count": "int64",
    "draft_block_length": "int64",
    "valid_proposal_count": "int64",
    "accepted_prefix_length": "int64",
    "termination_or_censor_reason": "string",
    "event_order": "int64",
}


def parquet(path, rows, types=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = (
        pa.schema([(name, pa.type_for_alias(kind)) for name, kind in types.items()])
        if types
        else None
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    pq.write_table(table, tmp, compression="zstd")
    with tmp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_rollout(folder, identity=None):
    import pyarrow.parquet as pq

    folder = Path(folder)
    marker = folder / "complete.json"
    if not marker.is_file():
        return False
    info = read_json(marker)
    required = {"events.parquet", "rounds.parquet", "responses.json", "requests.json", "audit.json"}
    if not required <= info["files_sha256"].keys():
        raise ValueError(f"Incomplete committed rollout artifact list: {folder}")
    if identity is not None and info["identity"] != identity:
        raise ValueError(f"Completed rollout belongs to a changed analysis: {folder}")
    verify_files(folder, info["files_sha256"])
    events = pq.read_table(folder / "events.parquet").to_pylist()
    validate_reconstruction(events, read_json(folder / "responses.json"))
    for key in ("checkpoint_step", "rollout_id"):
        if key in info and any(row[key] != info[key] for row in events):
            raise ValueError(f"Event {key} differs from committed identity: {folder}")
    requests_file = folder / "requests.json"
    if requests_file.is_file():
        requests = read_json(requests_file)
        for event in events:
            req = requests[event["request_id"]]
            if any(
                event[key] != req[key]
                for key in ("response_id", "sample_index", "prompt_id", "question_sha256")
            ):
                raise ValueError("Canonical request identity differs from logged events")
    if not info["frozen_parameter_parity"]:
        raise ValueError(f"Frozen weight parity was not verified: {folder}")
    return info


def native_counters(llm):
    counters = {}
    for metric in llm.get_metrics():
        if "spec_decode" in metric.name:
            if hasattr(metric, "values"):
                counters[metric.name] = list(map(int, metric.values))
            elif hasattr(metric, "value"):
                counters[metric.name] = int(metric.value)
    if not counters:
        raise RuntimeError("Pinned vLLM did not expose native speculative acceptance counters")
    return counters


def counter_difference(after, before):
    result = {}
    for name, value in after.items():
        prior = before.get(name, [0] * len(value) if isinstance(value, list) else 0)
        result[name] = (
            [a - b for a, b in zip(value, prior)] if isinstance(value, list) else value - prior
        )
    return result


def inference_kwargs(record, cache, seed):
    import torch

    policy = record["recipe"]["policy"]
    g = policy["generation"]
    cfg = g["vllm_cfg"]
    kwargs = copy.deepcopy(g["vllm_kwargs"])
    kwargs["speculative_config"]["model"] = str(Path(cache) / "draft")
    if kwargs.get("async_scheduling") is not False:
        raise ValueError("Original synchronous scheduling is required")
    kwargs.update(
        model=str(Path(cache) / "target"),
        tokenizer=str(Path(cache) / "target"),
        dtype=policy["precision"],
        seed=seed,
        load_format="safetensors",
        trust_remote_code=True,
        tensor_parallel_size=cfg["tensor_parallel_size"],
        pipeline_parallel_size=cfg["pipeline_parallel_size"],
        enforce_eager=cfg["enforce_eager"],
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        enable_prefix_caching=(
            cfg["enable_prefix_caching"]
            if cfg.get("enable_prefix_caching") is not None
            else torch.cuda.get_device_capability()[0] >= 8
        ),
        disable_log_stats=False,
    )
    for key in ("kv_cache_dtype", "logprobs_mode"):
        if key in cfg:
            kwargs[key] = cfg[key]
    return kwargs


def generate(llm, requests, generation, *, logged):
    from vllm import SamplingParams

    from .hook import worker_begin, worker_drain, worker_synchronize

    # Equal cache state for logged/plain parity, while retaining the source
    # engine's prefix-caching setting within each independent rollout.
    if not llm.reset_prefix_cache():
        raise RuntimeError("Could not reset idle prefix cache for independent frozen rollouts")
    if logged:
        llm.collective_rpc(worker_begin)
    initial_counters = native_counters(llm)
    internal, external = {}, {}
    engine = llm.llm_engine
    for index, req in enumerate(requests):
        name = f"spark-response-{index}"
        params = SamplingParams(
            n=1,
            seed=req["seed"],
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            top_k=generation["top_k"] if generation["top_k"] is not None else -1,
            max_tokens=req["max_tokens"],
            stop_token_ids=generation.get("stop_token_ids") or req["stop_token_ids"],
            ignore_eos=False,
            bad_words=generation.get("bad_words"),
            include_stop_str_in_output=True,
            logprobs=0,
        )
        request_id = engine.add_request(name, {"prompt_token_ids": req["prompt_token_ids"]}, params)
        internal[request_id] = req
        external[name] = request_id
    llm.collective_rpc(worker_synchronize)
    started = time.perf_counter()
    responses = {}
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                if len(output.outputs) != 1:
                    raise ValueError("Expected one canonical response per request")
                response = output.outputs[0]
                responses[external[output.request_id]] = dict(
                    token_ids=list(map(int, response.token_ids)),
                    finish_reason=response.finish_reason,
                    stop_reason=response.stop_reason,
                )
    llm.collective_rpc(worker_synchronize)
    elapsed = time.perf_counter() - started
    # Engine's native counters have completed with the final request output.
    counts = counter_difference(native_counters(llm), initial_counters)
    packets = llm.collective_rpc(worker_drain)[0] if logged else None
    return internal, responses, packets, counts, elapsed


def canonical_responses(requests, responses):
    return {requests[key]["response_id"]: value for key, value in responses.items()}


def collect_checkpoint(payload):
    from tqdm.auto import tqdm
    from vllm import LLM

    from .hook import parameter_audit, verify_contract

    record, inspected, plan = payload["record"], payload["inspected"], payload["plan"]
    root, cache, step = Path(payload["root"]), Path(payload["cache"]), inspected["checkpoint_step"]
    directory = root / f"collection/step_{step}"
    identity = digest(
        dict(
            source=inspected["source_identity"],
            plan=digest(plan),
            implementation=payload["implementation_sha256"],
        )
    )
    immutable_json(
        directory / "manifest.json",
        dict(
            identity=identity,
            checkpoint_step=step,
            source_identity=inspected["source_identity"],
            evaluation_manifest_sha256=digest(plan),
        ),
    )
    limit = 1 if payload["smoke"] else len(plan["rollouts"])
    pending = []
    for rollout in plan["rollouts"][:limit]:
        folder = directory / f"rollout_{rollout['rollout_id']:04d}"
        if validate_rollout(folder, identity):
            if not payload["resume"]:
                raise FileExistsError(
                    f"Completed rollout exists: {folder}; pass --resume to verify and skip"
                )
            print(f"Step {step}, rollout {rollout['rollout_id']}: verified, skipped", flush=True)
        else:
            pending.append((rollout, folder))
    if not pending:
        return
    source_contract = verify_contract()
    immutable_json(directory / "installed_vllm_contract.json", source_contract)
    llm = LLM(**inference_kwargs(record, cache, plan["seed"]))
    before = llm.collective_rpc(parameter_audit)[0]
    immutable_json(directory / "frozen_parameters.json", before)
    generation = plan["generation"]
    try:
        for pending_index, (rollout, folder) in enumerate(
            tqdm(pending, desc=f"Step {step}: frozen rollouts")
        ):
            requests, responses, packets, counters, logged_time = generate(
                llm, rollout["requests"], generation, logged=True
            )
            events, rounds = reconstruct(packets, requests, responses, step, rollout["rollout_id"])
            parity = dict(status="not_replayed", inference_latency_s=None)
            # One genuine logging-disabled replay per checkpoint/session.
            # No forwards are added inside the logged generation itself.
            if pending_index == 0:
                replay_requests, replay, _, replay_counts, replay_time = generate(
                    llm, rollout["requests"], generation, logged=False
                )
                if canonical_responses(requests, responses) != canonical_responses(
                    replay_requests, replay
                ):
                    raise RuntimeError(
                        "Logging-enabled/disabled replay differs in sampled sequences/termination; "
                        "deterministic passivity was not certified and the rollout was not committed"
                    )
                if counters != replay_counts:
                    raise RuntimeError(
                        "Logging-enabled/disabled native acceptance counters differ; "
                        "deterministic passivity was not certified"
                    )
                parity = dict(
                    status="passed",
                    generated_sequences_identical=True,
                    native_acceptance_counters_identical=True,
                    inference_latency_s=replay_time,
                )
            after = llm.collective_rpc(parameter_audit)[0]
            if before != after:
                raise RuntimeError("Target/draft parameters changed during frozen evaluation")
            folder.mkdir(parents=True, exist_ok=True)
            parquet(folder / "events.parquet", events, EVENT_TYPES)
            parquet(folder / "rounds.parquet", rounds)
            atomic_json(folder / "responses.json", responses)
            atomic_json(folder / "requests.json", requests)
            atomic_json(
                folder / "audit.json",
                dict(
                    parity=parity,
                    native_counters=counters,
                    logged_wall_s=logged_time,
                    logging_overhead_excluded_from_inference_latency=True,
                    frozen_parameters_sha256=digest(before),
                    optimizer_steps=0,
                    projector_created=False,
                ),
            )
            files = {
                name: file_hash(folder / name)
                for name in (
                    "events.parquet",
                    "rounds.parquet",
                    "responses.json",
                    "requests.json",
                    "audit.json",
                )
            }
            atomic_json(
                folder / "complete.json",
                dict(
                    identity=identity,
                    checkpoint_step=step,
                    rollout_id=rollout["rollout_id"],
                    files_sha256=files,
                    frozen_parameter_parity=True,
                    prompts=len({r["prompt_id"] for r in requests.values()}),
                    responses=len(responses),
                    rounds=len(rounds),
                    events=len(events),
                    ordinary_events=sum(row["is_miss"] is not None for row in events),
                    misses=sum(row["is_miss"] is True for row in events),
                ),
            )
    finally:
        llm.llm_engine.engine_core.shutdown()
