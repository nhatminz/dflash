import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from dflash_reflex.worker_runtime import configure_vllm_runner, inspect_vllm_hooks

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("architecture", ["Qwen2ForCausalLM", "Qwen3ForCausalLM"])
@pytest.mark.parametrize("eager", [False, True])
def test_actual_pinned_auto_selection_bypasses_v1_hooks_until_explicitly_pinned(
    monkeypatch, architecture, eager
):
    source = ROOT / "third_party/vllm/vllm/config/vllm.py"
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "VllmConfig"
    )
    nodes = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {
            "use_v2_model_runner",
            "_is_default_v2_model_runner_model",
        }
    ]

    class Env:
        @property
        def VLLM_USE_V2_MODEL_RUNNER(self):
            value = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
            return None if value is None else value == "1"

    namespace = dict(envs=Env(), HAS_TRITON=True, DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES=set())
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    model = NS(
        runner_type="generate",
        is_hybrid=False,
        is_attention_free=False,
        architectures=[architecture],
        is_moe=False,
        is_diffusion=False,
        enforce_eager=eager,
    )
    config = NS(
        model_config=model,
        speculative_config=NS(method="dflash"),
        _get_v2_model_runner_unsupported_features=lambda: [],
    )
    config._is_default_v2_model_runner_model = lambda: namespace[
        "_is_default_v2_model_runner_model"
    ](config)
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    assert namespace["use_v2_model_runner"](config) is True
    configure_vllm_runner()
    assert namespace["use_v2_model_runner"](config) is False


def test_v2_override_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with pytest.raises(ValueError, match="bypasses"):
        configure_vllm_runner()


def test_false_runner_setting_is_normalized_for_vllm_int_parser(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "false")
    configure_vllm_runner()
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_live_v2_runner_rejected_before_any_cuda_or_projector_load():
    from dflash_reflex.vllm_runtime import begin_rpc

    cls = type("GPUModelRunner", (), {"__module__": "vllm.v1.worker.gpu.model_runner"})
    with pytest.raises(RuntimeError, match="Unsupported live vLLM runner"):
        begin_rpc(NS(model_runner=cls()), "group", "/missing/projector.pt", True, {})


def test_all_three_installed_hook_sites_are_required(tmp_path):
    root = ROOT / "third_party/vllm/vllm"
    assert all(inspect_vllm_hooks(root).values())
    for relative in inspect_vllm_hooks(root):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((root / relative).read_text())
    proposer = tmp_path / "v1/spec_decode/llm_base_proposer.py"
    proposer.write_text(
        proposer.read_text().replace("return reflex.propose(self.model, hidden_states)", "pass")
    )
    result = inspect_vllm_hooks(tmp_path)
    assert not result["v1/spec_decode/llm_base_proposer.py"]
    assert result["v1/sample/rejection_sampler.py"]
