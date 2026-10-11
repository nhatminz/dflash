"""Strict, stdlib-only scientific comparison contract for the pinned backend."""

MATCHED_BLOCKERS = (
    "Target LoRA: pinned nemo_rl/models/megatron/setup.py constructs Bridge LoRA "
    "on fused linear_qkv/linear_fc1. This shares input factors across Q/K/V and "
    "gate/up, unlike seven independent PEFT rank-64 adapters. Canonical injection, "
    "merged vLLM refit and checkpoint round-trip have not been integrated/validated.",
    "Requested four-rollout target accumulation with independent draft/A steps is "
    "not expressed by the pinned joint Megatron target+draft optimizer. Raising "
    "train_global_batch_size does not implement it. Actual local SpecNaacl calls "
    "target.step after each nonempty filtered rollout, despite ACCUMULATION_STEPS=4.",
    "SpecNaacl excludes constant-reward groups from target only, uses population "
    "reward std without epsilon and retokenizes assistant messages; native NeMo "
    "keeps those samples and uses generated token masks. Their target denominator differs.",
)


def validate_profile(profile):
    if profile not in {"specnaacl_matched", "native_dflash"}:
        raise ValueError("FAIRNESS_PROFILE must be specnaacl_matched or native_dflash")
    if profile == "specnaacl_matched":
        raise ValueError(
            "specnaacl_matched preflight FAILED:\n- "
            + "\n- ".join(MATCHED_BLOCKERS)
            + "\nUse FAIRNESS_PROFILE=native_dflash for an explicitly non-equivalent run."
        )
