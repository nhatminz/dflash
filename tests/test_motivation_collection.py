import copy
import sys
import types

import numpy as np
import pytest
import torch

from analysis.verification_feedback.collector import EVENT_TYPES, parquet, validate_rollout
from analysis.verification_feedback.events import reconstruct
from analysis.verification_feedback.hook import PassiveHook
from analysis.verification_feedback.io import atomic_json, file_hash


def test_passive_compact_capture_preserves_rng_and_output_ownership():
    hook = PassiveHook()
    hook.begin()
    hook.context = dict(request_ids=["a"], discard=[False])
    output = torch.tensor([[4, 5, -1]], dtype=torch.int32)
    draft = torch.tensor([4, 7])
    rng = torch.get_rng_state().clone()
    hook.capture(output, [2], draft)
    output.fill_(0)
    draft.fill_(0)
    packets = hook.drain()
    assert packets[0]["outputs"] == [[4, 5, -1]]
    assert packets[0]["draft_ids"] == [4, 7]
    assert torch.equal(rng, torch.get_rng_state())
    assert not hook.enabled


def test_optional_hook_enabled_disabled_replay_cpu_contract(monkeypatch):
    from analysis.verification_feedback import hook as module

    sampler = types.ModuleType("vllm.v1.sample.rejection_sampler")
    calls = []

    def rejection(draft_token_ids, num_draft_tokens, draft_probs=None, synthetic_mode=False):
        calls.append("sampler")
        value = int(torch.randint(10, 20, (1,)))
        return torch.tensor([[value, -1, -1]], dtype=torch.int32)

    sampler.rejection_sample = rejection

    class Runner:
        use_async_scheduling = False
        input_batch = types.SimpleNamespace(req_ids=["a"])
        discard_request_mask = types.SimpleNamespace(np=np.array([False]))

        def _sample(self):
            return types.SimpleNamespace(
                sampled_token_ids=sampler.rejection_sample(torch.tensor([1, 2]), [2])
            )

    class Proposer:
        method = "dflash"

        def _greedy_sample(self, hidden):
            calls.append("proposer")
            return hidden.argmax(-1)

    modules = {
        "vllm": {},
        "vllm.v1": {},
        "vllm.v1.sample": {"rejection_sampler": sampler},
        "vllm.v1.sample.rejection_sampler": sampler,
        "vllm.v1.spec_decode": {},
        "vllm.v1.spec_decode.llm_base_proposer": {"SpecDecodeBaseProposer": Proposer},
        "vllm.v1.worker": {},
        "vllm.v1.worker.gpu_model_runner": {"GPUModelRunner": Runner},
    }
    for name, attributes in modules.items():
        if isinstance(attributes, dict):
            fake = types.ModuleType(name)
            fake.__dict__.update(attributes)
        else:
            fake = attributes
        monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setattr(module, "_installed", False)
    monkeypatch.setattr(module, "_hook", None)
    monkeypatch.setattr(module, "verify_contract", lambda: {})
    hook = module.install()
    torch.manual_seed(2026)
    hook.begin()
    logged = Runner()._sample().sampled_token_ids
    packets = hook.drain()
    torch.manual_seed(2026)
    plain = Runner()._sample().sampled_token_ids
    assert torch.equal(logged, plain)
    assert packets[0]["outputs"] == plain.tolist()
    assert calls == ["sampler", "sampler"]


def make_commit(tmp_path):
    requests = {
        "a": dict(
            response_id=0, prompt_id="p", question_sha256="p", sample_index=0, stop_token_ids=[9]
        )
    }
    responses = {"a": dict(token_ids=[2, 9], finish_reason="stop", stop_reason=None)}
    packets = [
        dict(request_ids=["a"], lengths=[1], draft_ids=[1], outputs=[[2, -1]], discard=[False]),
        dict(request_ids=["a"], lengths=[0], draft_ids=[], outputs=[[9]], discard=[False]),
    ]
    events, rounds = reconstruct(packets, requests, responses, 250, 0)
    folder = tmp_path / "rollout_0000"
    parquet(folder / "events.parquet", events, EVENT_TYPES)
    parquet(folder / "rounds.parquet", rounds)
    atomic_json(folder / "responses.json", responses)
    atomic_json(folder / "requests.json", requests)
    atomic_json(folder / "audit.json", dict(parity={"status": "passed"}))
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
        dict(identity="id", frozen_parameter_parity=True, files_sha256=files),
    )
    return folder


def test_resume_verifies_completed_rollout_instead_of_recollecting(tmp_path):
    folder = make_commit(tmp_path)
    before = {str(p): (p.stat().st_mtime_ns, file_hash(p)) for p in folder.iterdir()}
    assert validate_rollout(folder, "id")
    assert before == {str(p): (p.stat().st_mtime_ns, file_hash(p)) for p in folder.iterdir()}
    with pytest.raises(ValueError, match="changed analysis"):
        validate_rollout(folder, "different")


def test_corrupt_finalized_rollout_fails_and_partial_rollout_is_not_valid(tmp_path):
    folder = make_commit(tmp_path)
    (folder / "responses.json").write_text("{}")
    with pytest.raises(ValueError, match="corrupt"):
        validate_rollout(folder)
    (folder / "complete.json").unlink()
    assert validate_rollout(folder) is False


def test_mutation_of_stored_miss_label_is_detected(tmp_path):
    folder = make_commit(tmp_path)
    import pyarrow.parquet as pq

    events = pq.read_table(folder / "events.parquet").to_pylist()
    modified = copy.deepcopy(events)
    modified[0]["is_miss"] = False
    parquet(folder / "events.parquet", modified, EVENT_TYPES)
    from analysis.verification_feedback.io import read_json

    commit = read_json(folder / "complete.json")
    commit["files_sha256"]["events.parquet"] = file_hash(folder / "events.parquet")
    atomic_json(folder / "complete.json", commit)
    with pytest.raises(ValueError, match="Incorrect"):
        validate_rollout(folder)
