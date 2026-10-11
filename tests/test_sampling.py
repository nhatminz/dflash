"""Run the actual patched vLLM decision kernels without importing its engine.

AST extraction keeps their source/decorators intact and avoids installing a
CUDA-13 wheel on this CUDA-12 host. This tests kernels, not a live scheduler.
"""

import ast
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_disabled_hook_calls_original_proposal_without_reflex(monkeypatch):
    from dflash_reflex import vllm_runtime

    file = ROOT / "third_party/vllm/vllm/v1/spec_decode/llm_base_proposer.py"
    tree = ast.parse(file.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_greedy_sample"
    )
    function.decorator_list = []
    namespace = {"torch": torch, "HOOKS_ENABLED": False, "current": vllm_runtime.current}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(file), "exec"), namespace)

    def forbidden(*args):
        raise AssertionError("baseline entered OPD")

    monkeypatch.setattr(vllm_runtime, "_runtime", SimpleNamespace(enabled=False, propose=forbidden))
    hidden = torch.randn(3, 8)
    head = torch.randn(11, 8)
    runner = SimpleNamespace(
        method="dflash",
        use_local_argmax_reduction=False,
        use_heterogeneous_vocab=False,
        model=SimpleNamespace(compute_logits=lambda h: h @ head.t()),
    )
    actual = namespace["_greedy_sample"](runner, hidden)
    torch.testing.assert_close(actual, (hidden @ head.t()).argmax(-1))


def load_kernels(tmp_path, original=False):
    file = ROOT / "third_party/vllm/vllm/v1/sample/rejection_sampler.py"
    source = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT / "third_party/vllm"),
                "show",
                "HEAD:vllm/v1/sample/rejection_sampler.py",
            ],
            text=True,
        )
        if original
        else file.read_text()
    )
    tree = ast.parse(source)
    names = {
        "rejection_greedy_sample_kernel",
        "rejection_random_sample_kernel",
        "sample_recovered_tokens_kernel",
    }
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    code = "import triton\nimport triton.language as tl\n"
    for node in nodes:
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        code += "".join(source.splitlines(keepends=True)[start : node.end_lineno]) + "\n\n"
    path = tmp_path / ("original_sampler.py" if original else "patched_sampler.py")
    path.write_text(code)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("greedy", [True, False])
def test_actual_sampler_prefix_and_baseline_equivalence(tmp_path, greedy):
    patched, original = load_kernels(tmp_path), load_kernels(tmp_path, True)
    device = "cuda"
    cumulative = torch.tensor([3, 5, 5, 8], dtype=torch.int32, device=device)
    draft = torch.tensor([1, 2, 3, 2, 1, 2, -1, 3], dtype=torch.int32, device=device)
    target = torch.tensor([1, 0, 3, 2, 1, 2, 1, 3], dtype=torch.int32, device=device)
    bonus = torch.tensor([5, 5, 5, 5], dtype=torch.int32, device=device)
    counts = torch.full((4,), -99, dtype=torch.int32, device=device)
    result, baseline = [torch.full((4, 4), -1, dtype=torch.int32, device=device) for _ in range(2)]
    if greedy:
        args = (cumulative, draft, target, bonus, None, 3, None, None)
        patched.rejection_greedy_sample_kernel[(4,)](
            result, *args, SYNTHETIC_MODE=False, accepted_prefix_ptr=counts
        )
        original.rejection_greedy_sample_kernel[(4,)](baseline, *args, SYNTHETIC_MODE=False)
    else:
        probs = torch.full((8, 6), 0.1, device=device)
        probs[torch.arange(8, device=device), target.long()] = 0.5
        uniform = torch.tensor(
            [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], device=device, dtype=torch.float64
        )
        recovered = target.clone()
        isgreedy = torch.zeros(4, dtype=torch.bool, device=device)
        args = (cumulative, draft, None, probs, bonus, recovered, uniform, isgreedy, 3, 6, None)
        patched.rejection_random_sample_kernel[(4,)](
            result, *args, NO_DRAFT_PROBS=True, SYNTHETIC_MODE=False, accepted_prefix_ptr=counts
        )
        original.rejection_random_sample_kernel[(4,)](
            baseline, *args, NO_DRAFT_PROBS=True, SYNTHETIC_MODE=False
        )
    torch.testing.assert_close(result, baseline, rtol=0, atol=0)
    assert counts.cpu().tolist() == [1, 2, 0, 1]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_actual_greedy_draft_preserves_target_distribution(tmp_path):
    sampler = load_kernels(tmp_path)
    n, vocab = 40000, 3
    p = torch.tensor([0.2, 0.5, 0.3], device="cuda").expand(n, -1).contiguous()
    draft = torch.ones(n, dtype=torch.int32, device="cuda")
    cumulative = torch.arange(1, n + 1, device="cuda", dtype=torch.int32)
    inv_q = torch.empty_like(p).exponential_().reciprocal_()
    recovered = torch.empty_like(draft)
    sampler.sample_recovered_tokens_kernel[(n, 1)](
        recovered,
        cumulative,
        draft,
        None,
        p,
        inv_q,
        vocab,
        128,
        NO_DRAFT_PROBS=True,
        USE_FP64_GUMBEL=False,
    )
    result = torch.full((n, 2), -1, device="cuda", dtype=torch.int32)
    counts = torch.empty_like(draft)
    sampler.rejection_random_sample_kernel[(n,)](
        result,
        cumulative,
        draft,
        None,
        p,
        torch.zeros_like(draft),
        recovered,
        torch.rand(n, device="cuda", dtype=torch.float64),
        torch.zeros(n, device="cuda", dtype=torch.bool),
        1,
        vocab,
        None,
        NO_DRAFT_PROBS=True,
        SYNTHETIC_MODE=False,
        accepted_prefix_ptr=counts,
    )
    empirical = torch.bincount(result[:, 0].long(), minlength=vocab) / n
    torch.testing.assert_close(empirical.cpu(), torch.tensor([0.2, 0.5, 0.3]), atol=0.012, rtol=0)
