"""Model identity and DFlash architecture checks before CUDA initialization."""

import hashlib
import json
import math
import struct
from pathlib import Path


def fingerprint(model):
    model = Path(model)
    files = [model / "config.json", model / "tokenizer.json", model / "tokenizer_config.json"]
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def draft_config(target, *, block_size=16, num_layers=3, mask_token_id=None):
    if target["model_type"] != "qwen3" or target.get("attention_bias", False):
        raise ValueError(
            "pinned NVIDIA DFlash supports bias-free Qwen3 here; Qwen2.5 is not advertised"
        )
    if not 2 <= block_size <= 64:
        raise ValueError("DFLASH_BLOCK_SIZE includes the anchor and must be in [2,64]")
    if num_layers < 1:
        raise ValueError("DFLASH_LAYERS must be positive")
    if mask_token_id is None:
        # Official DFlash uses the first unused Qwen3 padded embedding row.
        mask_token_id = 151669
    if not 0 <= mask_token_id < target["vocab_size"]:
        raise ValueError("MASK id is outside the target vocabulary")
    n = target["num_hidden_layers"]
    layers = sorted(set((1, max(0, n // 2 - 1), max(1, n - 4))))
    config = dict(target)
    config.update(
        architectures=["DFlashDraftModel"],
        num_hidden_layers=num_layers,
        num_target_layers=n,
        model_type="qwen3",
        attention_bias=False,
        layer_types=["full_attention"] * num_layers,
        dflash_config=dict(
            block_size=block_size,
            mask_token_id=mask_token_id,
            target_layer_ids=layers,
            use_aux_hidden_state=True,
            causal=False,
        ),
    )
    return config


def validate_checkpoint(target_path, checkpoint, block_size):
    if not 2 <= block_size <= 64:
        raise ValueError("block size must be in [2,64]")
    target = json.loads((Path(target_path) / "config.json").read_text())
    path = Path(checkpoint)
    draft = json.loads((path / "config.json").read_text())
    for key in (
        "hidden_size",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    ):
        if draft.get(key) != target.get(key):
            raise ValueError(f"target/draft {key} mismatch: {target.get(key)} vs {draft.get(key)}")
    d = draft["dflash_config"]
    if d.get("block_size", draft.get("block_size")) != block_size:
        raise ValueError("checkpoint block size differs from DFLASH_BLOCK_SIZE")
    if (
        draft.get("attention_bias")
        or d.get("causal")
        or draft.get("sliding_window")
        or d.get("use_swa")
        or not d.get("use_aux_hidden_state", True)
    ):
        raise ValueError("unsupported block attention contract")
    if draft.get("logit_scale", 1.0) != 1.0 or draft.get("quantization_config"):
        raise ValueError(
            "Reflex selected-head feedback requires an unquantized, unit-scale shared LM head"
        )
    if target["model_type"] != "qwen3" or target.get("attention_bias", False):
        raise ValueError("only Qwen3 targets are supported by this pinned backend")
    if not 0 <= d["mask_token_id"] < target["vocab_size"]:
        raise ValueError("invalid checkpoint MASK id")
    if not all(0 <= i < target["num_hidden_layers"] for i in d["target_layer_ids"]):
        raise ValueError("invalid target auxiliary layer index")
    if not d["target_layer_ids"] or d["target_layer_ids"] != sorted(set(d["target_layer_ids"])):
        raise ValueError("auxiliary layer indices must be unique and ascending")
    manifest = path / "dflash_manifest.json"
    if manifest.is_file():
        if json.loads(manifest.read_text())["target_fingerprint"] != fingerprint(target_path):
            raise ValueError("draft belongs to a different target/tokenizer checkpoint")
    else:
        # Explicitly allow the audited z-lab 8B contract, not arbitrary weights.
        reference = (
            Path(__file__).resolve().parents[1] / "third_party/reference_qwen3_8b_b16_config.json"
        )
        ref = json.loads(reference.read_text())
        for key in ("hidden_size", "vocab_size", "num_target_layers"):
            actual = (
                target.get("num_hidden_layers") if key == "num_target_layers" else target.get(key)
            )
            expected = ref[key]
            if actual != expected:
                raise ValueError(
                    "unmanifested external draft requires the audited Qwen3-8B target contract"
                )
        if d != ref["dflash_config"]:
            raise ValueError("external draft does not match the audited z-lab b16 config")
        for key in (
            "num_hidden_layers",
            "head_dim",
            "intermediate_size",
            "rms_norm_eps",
            "rope_theta",
        ):
            if draft.get(key) != ref.get(key):
                raise ValueError(f"external draft {key} differs from the audited reference")
        tokenizer = json.loads((Path(target_path) / "tokenizer.json").read_text())
        revisions = json.loads((reference.parent / "reference_hf_revisions.json").read_text())
        payload = {
            key: tokenizer[key] for key in ("model", "added_tokens", "normalizer", "pre_tokenizer")
        }
        identity = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if identity != revisions["Qwen/Qwen3-8B"]["tokenizer_contract_sha256"]:
            raise ValueError(
                "external z-lab checkpoint requires the audited Qwen3-8B tokenizer vocabulary"
            )
        used_ids = set(tokenizer.get("model", {}).get("vocab", {}).values())
        used_ids.update(entry["id"] for entry in tokenizer.get("added_tokens", []))
        if d["mask_token_id"] in used_ids or max(used_ids, default=-1) + 1 != d["mask_token_id"]:
            raise ValueError("external draft requires the audited Qwen3 unused MASK row 151669")
    if not any(path.glob("*.safetensors")):
        raise FileNotFoundError(f"no DFlash weights in {path}")
    validate_weight_headers(path, draft, target)
    return draft


def validate_weight_headers(path, draft, target):
    """Inspect safetensors metadata without allocating any model weights."""
    tensors = {}
    for file in Path(path).glob("*.safetensors"):
        with file.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"invalid safetensors header: {file}")
            size = struct.unpack("<Q", prefix)[0]
            if size > 16 * 1024 * 1024:
                raise ValueError(f"unreasonable safetensors header: {file}")
            header = json.loads(stream.read(size))
            payload_size = file.stat().st_size - 8 - size
        for key, value in header.items():
            if key != "__metadata__":
                start, end = value["data_offsets"]
                if not 0 <= start <= end <= payload_size:
                    raise ValueError(f"invalid weight byte offsets: {file}:{key}")
                width = {"BF16": 2, "F16": 2, "F32": 4}.get(value["dtype"])
                if width is None or math.prod(value["shape"]) * width != end - start:
                    raise ValueError(f"invalid DFlash tensor bytes or dtype: {key}")
                normalized = key.removeprefix("model.")
                if normalized in tensors:
                    raise ValueError(f"duplicate checkpoint weight: {key}")
                tensors[normalized] = value
    h = draft["hidden_size"]
    hd = draft.get("head_dim", h // draft["num_attention_heads"])
    q = draft["num_attention_heads"] * hd
    kv = draft["num_key_value_heads"] * hd
    intermediate = draft["intermediate_size"]
    expected = {
        "fc.weight": [h, target["hidden_size"] * len(draft["dflash_config"]["target_layer_ids"])],
        "hidden_norm.weight": [h],
        "norm.weight": [h],
    }
    for layer in range(draft["num_hidden_layers"]):
        prefix = f"layers.{layer}."
        expected.update(
            {
                prefix + name: shape
                for name, shape in {
                    "self_attn.q_proj.weight": [q, h],
                    "self_attn.k_proj.weight": [kv, h],
                    "self_attn.v_proj.weight": [kv, h],
                    "self_attn.o_proj.weight": [h, q],
                    "self_attn.q_norm.weight": [hd],
                    "self_attn.k_norm.weight": [hd],
                    "input_layernorm.weight": [h],
                    "post_attention_layernorm.weight": [h],
                    "mlp.gate_proj.weight": [intermediate, h],
                    "mlp.up_proj.weight": [intermediate, h],
                    "mlp.down_proj.weight": [h, intermediate],
                }.items()
            }
        )
    for key, shape in expected.items():
        if key not in tensors or tensors[key]["shape"] != shape:
            raise ValueError(f"DFlash checkpoint weight {key} must have shape {shape}")
        if tensors[key]["dtype"] not in {"BF16", "F16", "F32"}:
            raise ValueError(f"unsupported checkpoint precision: {key}")
    if any(key in tensors for key in ("lm_head.weight", "embed_tokens.weight")):
        raise ValueError("DFlash must share the target LM head and embedding")
