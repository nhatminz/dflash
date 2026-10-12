"""Request alignment/weights are checked against host indexing, not the GPU kernel."""

from types import SimpleNamespace as NS

import pytest
import torch

from dflash_reflex.adapter import initialize_projector
from dflash_reflex.config import ReflexConfig
from dflash_reflex.dispatch import PIPELINE_REVISION
from dflash_reflex.vllm_runtime import Runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")]


@pytest.mark.parametrize("pipeline", ["fused", "selective_masked", "selective"])
@pytest.mark.parametrize("stream", [False, True])
def test_eos_discard_zero_length_reorder_and_staging_growth(pipeline, stream):
    torch.manual_seed(892)
    width, d, vocab = 15, 32, 1031
    head = torch.randn(vocab, d, device="cuda") * 0.02
    h = torch.randn(6 * width, d, device="cuda", dtype=torch.bfloat16)
    raw = h.float() @ head.t()
    model = NS(lm_head=NS(weight=head), compute_logits=lambda _: raw)
    rt = Runtime(
        NS(device="cuda", max_num_reqs=6, speculative_config=NS(num_speculative_tokens=width)),
        "metadata",
        initialize_projector(head),
        0,
        True,
        ReflexConfig(
            pipeline=pipeline, update_stream=stream, accepted_weight=0.5, rejected_weight=1.5
        ),
    )
    rt.request_ids = list("abcdef")
    proposed = rt.propose(model, h).view(6, width)
    rt.request_ids = list("fbdace")
    lengths = [0, 3, 1, 15, 8, 0]
    prefixes = [0, 2, 0, 4, 1, 0]
    rt.remaining = [4, 2, 10, 12, 8, 20]
    rt.eos = [[], [9, 99], [], [], [], []]
    discarded = [False, False, True, False, False, False]
    rt.discard = torch.tensor(discarded, device="cuda")
    draft = torch.cat(
        [
            proposed[list("abcdef").index(key), :length]
            for key, length in zip(rt.request_ids, lengths)
        ]
    )
    draft[3 + 1 + 4] = -1  # Invalid draft position must not contribute.
    output_cpu = [[2] * 16 for _ in lengths]
    output_cpu[1][0] = 9
    out = torch.tensor(output_cpu, device="cuda").repeat_interleave(2, dim=1)[:, ::2]
    p = torch.randn(sum(lengths), vocab, device="cuda").softmax(-1)
    rt.verified(
        p,
        torch.tensor(prefixes, device="cuda", dtype=torch.int32),
        NS(num_draft_tokens=lengths, draft_token_ids=draft),
        out,
    )
    rt.adapter.wait()
    expected_rows, expected_weights, expected_counters = [], [], [0] * 4
    draft_cpu = draft.cpu().tolist()
    offset = 0
    for key, length, prefix, rem, eos, discard, output in zip(
        rt.request_ids, lengths, prefixes, rt.remaining, rt.eos, discarded, output_cpu
    ):
        limit = (
            min(rem, next((i + 1 for i, token in enumerate(output) if token in eos), 17))
            if eos
            else rem
        )
        if length and not discard:
            expected_counters[0] += sum(token >= 0 and i < limit for i, token in enumerate(output))
            expected_counters[1] += 1
            expected_counters[2] += min(prefix, limit)
            expected_counters[3] += min(length, limit)
        for position in range(length):
            expected_rows.append(list("abcdef").index(key) * width + position)
            valid = (
                not discard
                and position <= prefix
                and position < limit
                and draft_cpu[offset + position] >= 0
            )
            expected_weights.append((0.5 if position < prefix else 1.5) if valid else 0)
        offset += length
    ws, n = rt.metadata_workspace, sum(lengths)
    assert ws.raw_rows[:n].cpu().tolist() == expected_rows
    assert ws.weights[:n].cpu().tolist() == expected_weights
    assert rt.counters[:4].cpu().tolist() == expected_counters
    valid = torch.tensor(expected_weights, device="cuda") > 0
    torch.testing.assert_close(ws.h[:n][valid], h[expected_rows].float()[valid], rtol=0, atol=0)
    assert rt.adapter.grad_weight.item() == sum(expected_weights)
    packet = rt.finish()
    assert packet["pipeline_rounds"][pipeline] == 1
    assert rt.adapter.prior_feedback_density == sum(w > 0 for w in expected_weights) / n
    assert packet["updates"] == 1 and rt.adapter.B.norm() > 0
    assert packet["grad_sum"].norm() == 0  # First round uses B_t=0, before updating B.
    # Ready staging storage is replaced when stop width grows, then reused.
    torch.cuda.synchronize()
    ws.upload([1], [8], [[9] * 20], [0])
    torch.cuda.synchronize()
    count = len(ws.staging)
    ws.upload([1], [8], [[9] * 20], [0])
    assert len(ws.staging) == count


def test_auto_uses_calibration_recent_density_and_logs_nonzero_updates(tmp_path, monkeypatch):
    import json

    torch.manual_seed(129)
    d, v, width = 32, 1031, 15
    cache = dict(
        schema_version=3,
        implementation_revision=PIPELINE_REVISION,
        gpu=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        hidden=d,
        vocab=v,
        rank=8,
        topk=16,
        block_size=16,
        hidden_dtype="torch.float32",
        logits_dtype="torch.float32",
        measurement_scope="runtime_complete",
        measurements=[
            dict(requests=8, density=density, pipeline=pipeline, median_ms=time, update_stream=True)
            for density in (1 / 15, 1.0)
            for pipeline, time in [
                ("selective_masked", 1 if density < 1 else 20),
                ("fused", 10 if density < 1 else 5),
            ]
        ],
    )
    path = tmp_path / "dispatch.json"
    path.write_text(json.dumps(cache))
    monkeypatch.setenv("OPD_DISPATCH_CACHE", str(path))
    head = torch.randn(v, d, device="cuda") * 0.1
    rt = Runtime(
        NS(device="cuda", max_num_reqs=8, speculative_config=NS(num_speculative_tokens=width)),
        "auto",
        initialize_projector(head),
        0,
        True,
        ReflexConfig(diagnostics=True, update_stream=True),
    )
    model = NS(lm_head=NS(weight=head), compute_logits=lambda h: h @ head.t())
    for round_id in range(20):
        batch = 8 if round_id < 17 else 3
        rt.request_ids = [str(i) for i in range(batch)]
        rt.remaining, rt.eos = [2048] * batch, [[] for _ in range(batch)]
        rt.discard = torch.zeros(batch, device="cuda", dtype=torch.bool)
        h = torch.randn(batch * width, d, device="cuda")
        if round_id == 17:
            # Prior completed feedback, never the upcoming acceptance outcome.
            torch.cuda.synchronize()
            rt.density_pending = None
            rt.adapter.prior_feedback_density = 1.0
        ids = rt.propose(model, h)
        assert rt.pipeline == ("selective_masked" if round_id < 17 else "fused")
        rt.verified(
            torch.randn(batch * width, v, device="cuda").softmax(-1),
            torch.zeros(batch, device="cuda", dtype=torch.int32),
            NS(num_draft_tokens=[width] * batch, draft_token_ids=ids),
            torch.ones(batch, width + 1, device="cuda", dtype=torch.int32),
        )
    packet = rt.finish()
    assert packet["updates"] == 20
    assert packet["pipeline_rounds"] == dict(fused=3, selective=0, selective_masked=17)
    assert packet["grad_sum"].norm() > 0 and packet["B_norm"] > 0
