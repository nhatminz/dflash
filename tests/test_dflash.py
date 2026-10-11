import importlib.util
import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config

from dflash_reflex.block_training import (
    block_attention_mask,
    block_loss_reference,
    construct_blocks,
    slot_mask,
    slot_weights,
)
from dflash_reflex.model_contract import draft_config, fingerprint, validate_checkpoint


def original_model():
    file = Path(__file__).resolve().parents[1] / "third_party/original-dflash/dflash/model.py"
    spec = importlib.util.spec_from_file_location("audited_dflash_model", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = Qwen3Config(
        vocab_size=48,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        attention_dropout=0.0,
        num_target_layers=4,
        dflash_config=dict(target_layer_ids=[0, 2], mask_token_id=47, block_size=4),
    )
    cfg._attn_implementation = "sdpa"
    return module.DFlashDraftModel(cfg)


def test_masks_targets_and_prefix():
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9]])
    anchors = torch.tensor([[1, 4]])
    blocks, positions, targets = construct_blocks(ids, anchors, 4, 47)
    assert blocks.tolist() == [[[4, 47, 47, 47], [7, 47, 47, 47]]]
    assert targets[0, 0].tolist() == [5, 6, 7]
    mask = block_attention_mask(anchors, 7, 4)
    assert mask.shape == (1, 1, 8, 15)
    assert mask[0, 0, 0, :7].tolist() == [True, False, False, False, False, False, False]
    assert mask[0, 0, 0, 7:].tolist() == [True, True, True, True, False, False, False, False]
    valid = slot_mask(
        torch.tensor([[0, 0, 1, 0, 1, 1, 1]]),
        torch.ones(1),
        torch.tensor([[1]]),
        torch.ones(1, 1, dtype=torch.bool),
        4,
    )
    assert valid.tolist() == [[[True, False, False, False]]]
    assert slot_weights(16) == pytest.approx([__import__("math").exp(-j / 7) for j in range(15)])


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
            ],
        ),
    ],
)
def test_actual_original_dflash_forward_backward_and_save(tmp_path, device):
    torch.manual_seed(9)
    model = original_model().to(device)
    b, s, n, width = 2, 7, 2, 4
    anchors = torch.tensor([[1, 3], [2, 4]], device=device)
    positions = anchors[..., None] + torch.arange(width, device=device)
    position_ids = torch.cat(
        (torch.arange(s, device=device).expand(b, -1), positions.flatten(1)), 1
    )
    taps = torch.randn(b, s, 32, device=device)
    noise = torch.randn(b, n * width, 16, device=device)
    mask = block_attention_mask(anchors, s, width)
    output = model(
        position_ids=position_ids, noise_embedding=noise, target_hidden=taps, attention_mask=mask
    )
    assert output.shape == (b, n * width, 16)
    head = torch.randn(48, 16, device=device)
    student = (output @ head.t()).view(b, n, width, 48)[:, :, 1:]
    teacher = torch.randn(b, s, 48, device=device)
    live = slot_mask(
        torch.ones(b, s, device=device),
        torch.ones(b, device=device),
        anchors,
        torch.ones(b, n, dtype=torch.bool, device=device),
        3,
    )
    loss = block_loss_reference(student, teacher, anchors, live, slot_weights(width))
    assert loss.isfinite() and loss > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.fc.weight.detach().clone()
    loss.backward()
    assert model.fc.weight.grad.norm() > 0
    optimizer.step()
    assert not torch.equal(before, model.fc.weight)
    model.save_pretrained(tmp_path)
    restored = type(model).from_pretrained(tmp_path).to(device)
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_supported_contract():
    target = dict(
        model_type="qwen3",
        num_hidden_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        vocab_size=151936,
    )
    assert draft_config(target, block_size=16)["dflash_config"]["block_size"] == 16
    for size in (0, 1, 65):
        with pytest.raises(ValueError):
            draft_config(target, block_size=size)
    with pytest.raises(ValueError):
        draft_config(dict(target, model_type="llama"))


def test_header_validation_before_allocating_weights(tmp_path):
    from safetensors.torch import save_file

    model = original_model()
    target = tmp_path / "target"
    target.mkdir()
    cfg = model.config.to_dict()
    cfg["num_hidden_layers"] = 4
    (target / "config.json").write_text(json.dumps(cfg))
    (target / "tokenizer.json").write_text("{}")
    (target / "tokenizer_config.json").write_text("{}")
    checkpoint = tmp_path / "draft"
    model.save_pretrained(checkpoint)
    (checkpoint / "dflash_manifest.json").write_text(
        json.dumps(dict(target_fingerprint=fingerprint(target)))
    )
    validate_checkpoint(target, checkpoint, 4)
    weights = model.state_dict()
    weights["fc.weight"] = torch.zeros(16, 31)
    save_file(weights, str(checkpoint / "model.safetensors"))
    with pytest.raises(ValueError, match="fc.weight"):
        validate_checkpoint(target, checkpoint, 4)
