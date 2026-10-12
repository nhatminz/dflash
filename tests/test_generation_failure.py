from types import SimpleNamespace

import pytest

from dflash_reflex.nemo_bridge import generate_cohort


@pytest.mark.parametrize("unwritable_report", [False, True])
def test_generation_failure_preserves_original_without_cuda_finish(
    tmp_path, monkeypatch, capsys, unwritable_report
):
    monkeypatch.setenv("DFLASH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("METHOD", "dflash")
    monkeypatch.setenv("BATCH_SIZE", "8")
    monkeypatch.setenv("RESPONSES_PER_PROMPT", "8")
    if unwritable_report:
        (tmp_path / "generation_errors").write_text("directory unavailable")
    calls = []
    original = RuntimeError("CUDA error: an illegal memory access was encountered")

    def rpc(function, **kwargs):
        calls.append(function.__name__)
        if function.__name__ == "finish_rpc":
            raise AssertionError("failed CUDA engine must not be asked to finish")

    def generate(*args, **kwargs):
        raise original

    llm = SimpleNamespace(collective_rpc=rpc, generate=generate)
    with pytest.raises(RuntimeError) as caught:
        generate_cohort(llm, list(range(64)), None, use_tqdm=False)
    assert caught.value is original
    assert calls == ["begin_rpc"]
    assert "illegal memory access" in capsys.readouterr().err
    if not unwritable_report:
        reports = list((tmp_path / "generation_errors").glob("*.log"))
        assert len(reports) == 1
        assert "illegal memory access" in reports[0].read_text()
    assert not (tmp_path / "ipc/feedback").exists()


def test_reflex_without_hooks_never_publishes_feedback(tmp_path, monkeypatch):
    import torch

    from dflash_reflex import nemo_bridge as bridge

    monkeypatch.setenv("DFLASH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("METHOD", "dflash_reflex")
    monkeypatch.setenv("BATCH_SIZE", "8")
    monkeypatch.setenv("RESPONSES_PER_PROMPT", "8")
    monkeypatch.setattr(bridge, "atomic_save", lambda *args: None)
    output = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 2, 3])])
    packet = dict(inference_rng={}, counters=torch.zeros(8), proposal_calls=0, verification_calls=0)
    llm = SimpleNamespace(
        collective_rpc=lambda *args, **kwargs: [packet],
        generate=lambda *args, **kwargs: [output] * 64,
    )
    with pytest.raises(RuntimeError, match="without proposal/verification hooks"):
        generate_cohort(llm, list(range(64)), None, use_tqdm=False)
    assert not (tmp_path / "ipc/feedback").exists()
