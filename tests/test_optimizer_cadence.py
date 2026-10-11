import ast
from pathlib import Path

import pytest
import torch

from dflash_reflex.grpo_loss import sequence_objective

ROOT = Path(__file__).resolve().parents[1]


def spec_target_loss():
    file = ROOT.parent / "SpecNaacl/helper/fastgrpo_training.py"
    fn = next(
        x
        for x in ast.parse(file.read_text()).body
        if isinstance(x, ast.FunctionDef) and x.name == "compute_target_loss"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(file), "exec"), namespace)
    return namespace[fn.name]


@pytest.mark.parametrize("iteration", [0, 1])
def test_real_spec_loss_value_and_gradient_parity(iteration):
    torch.manual_seed(88)
    x = torch.randn(3, 6, 9, requires_grad=True)
    reference = torch.randn_like(x)
    labels = torch.randint(9, (3, 6))
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0.0]])
    adv = torch.tensor([[1.0], [-2.0], [0.3]])
    logp = x[:, :-1].float().log_softmax(-1).gather(-1, labels[:, 1:, None]).squeeze(-1)
    ref = reference[:, :-1].log_softmax(-1).gather(-1, labels[:, 1:, None]).squeeze(-1)
    old = (logp.detach() + torch.tensor([[0.4], [-0.4], [0]])) if iteration else None
    expected = (
        spec_target_loss()(
            x, reference if iteration == 0 else ref, old, labels, mask, adv, 0.1, 0.04, iteration
        )[0]
        / 3
    )
    got = sequence_objective(logp, ref, adv, mask[:, :-1], 0.04, 0.1, old)
    torch.testing.assert_close(got, expected, rtol=2e-6, atol=1e-7)
    ge = torch.autograd.grad(expected, x, retain_graph=True)[0]
    gg = torch.autograd.grad(got, x, retain_graph=True)[0]
    torch.testing.assert_close(gg, ge, rtol=2e-6, atol=1e-7)
    no_kl = sequence_objective(logp, ref, adv, mask[:, :-1], 0, 0.1, old)
    assert not torch.allclose(torch.autograd.grad(no_kl, x)[0], gg)


def test_spec_actual_step_is_in_rollout_loop_and_pending_feedback_survives_skip():
    source = (ROOT.parent / "SpecNaacl/grpo_speculative.py").read_text()
    assert "all_ranks_ready = len(batch_data['messages']) > 0" in source
    assert "optimizer_target.step()" in source
    # No readiness threshold of 8*4 exists before the step in this revision.
    window = source[source.index("all_ranks_ready =") : source.index("optimizer_target.step()")]
    assert "len(batch_data['messages']) >= batch_size * accumulation_steps" not in window
    from dflash_reflex.config import ReflexConfig
    from dflash_reflex.projector_training import ProjectorTrainer

    t = ProjectorTrainer(torch.ones(4, 2), ReflexConfig(rank=2))
    t.accumulate(
        dict(group_id="a", version=0, grad_sum=torch.ones(4, 2), grad_weight=torch.tensor(2.0))
    )
    # A skipped joint step must not call trainer.step or clear these buffers.
    saved = t.state_dict()
    clone = ProjectorTrainer(torch.zeros(4, 2), ReflexConfig(rank=2))
    clone.load_state_dict(saved)
    assert clone.version == 0 and clone.pending_weight == 2
    assert clone.step() and clone.version == 1


def test_actual_patched_nemo_loss_matches_spec_value_and_gradient():
    # Execute the pinned loss class and its actual reduction helpers, without
    # importing Ray/Megatron (not installed on this CUDA-12 development host).
    import math
    from types import SimpleNamespace
    from typing import Optional

    from pydantic import BaseModel, Field

    source = ROOT / "third_party/nemo-rl/nemo_rl/algorithms/loss/loss_functions.py"
    names = {"ClippedPGLossConfig", "ClippedPGLossFn"}
    nodes = [
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name in names
    ]
    utility = ROOT / "third_party/nemo-rl/nemo_rl/algorithms/utils.py"
    helpers = [
        n
        for n in ast.parse(utility.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in {"calculate_kl", "masked_mean"}
    ]
    ns = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "Optional": Optional,
        "math": math,
        "BaseModel": BaseModel,
        "Field": Field,
        "LossFunction": object,
        "LossType": SimpleNamespace(TOKEN_LEVEL=1, SEQUENCE_LEVEL=2),
        "LossInputType": SimpleNamespace(LOGPROB=1),
        "MetricNormalizer": SimpleNamespace(TOKENS=1, SEQUENCES=2, NONE=3),
    }
    # Postpone type-only annotations; the complete numerical methods are intact.
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        + helpers
        + nodes,
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), "exec"), ns)
    cfg = ns["ClippedPGLossConfig"](
        reference_policy_kl_penalty=0.04,
        ratio_clip_min=0.1,
        ratio_clip_max=0.1,
        token_level_loss=False,
        force_on_policy_ratio=True,
        specnaacl_kl_gradient=True,
        kl_input_clamp_value=None,
        kl_output_clamp_value=None,
    )
    fn = ns["ClippedPGLossFn"](cfg)
    torch.manual_seed(4)
    logp = torch.randn(3, 5, requires_grad=True)
    ref = torch.randn(3, 5)
    mask = torch.tensor([[0, 1, 1, 1, 1], [0, 1, 1, 0, 0], [0, 1, 1, 1, 0.0]])
    adv = torch.tensor([[1.0], [-1.0], [0.5]]).expand_as(logp)

    def pad(t):
        return torch.cat([torch.zeros(3, 1), t], -1)

    data = dict(
        token_mask=pad(mask),
        sample_mask=torch.ones(3),
        advantages=pad(adv),
        prev_logprobs=pad(logp.detach()),
        generation_logprobs=pad(logp.detach()),
        reference_policy_logprobs=pad(ref),
    )
    actual, metrics = fn(logp, data, torch.tensor(3.0), mask.sum())
    expected = sequence_objective(logp, ref, adv, mask, 0.04, 0.1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logp, retain_graph=True)[0],
        torch.autograd.grad(expected, logp)[0],
        rtol=2e-6,
        atol=1e-7,
    )
    assert metrics["kl_penalty"] > 0
