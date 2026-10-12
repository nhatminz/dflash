"""Public configuration. Importing this module does not initialize CUDA."""

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ReflexConfig:
    rank: int = 8
    topk: int = 16
    fast_lr: float = 0.01
    accepted_weight: float = 1.0
    rejected_weight: float = 1.0
    projector_lr: float = 1e-4
    projector_weight_decay: float = 0.0
    update_interval: int = 1
    update_stream: bool = False
    proposal_mode: str = "auto"
    dense_implementation: str = "auto"
    pipeline: str = "auto"
    metrics_mode: str = "production"
    profile: bool = False
    diagnostics: bool = False

    def __post_init__(self):
        if not 1 <= self.rank <= 64 or self.topk < 1:
            raise ValueError("rank must be in [1,64]; topk must be positive")
        if (
            min(
                self.fast_lr,
                self.accepted_weight,
                self.rejected_weight,
                self.projector_lr,
                self.projector_weight_decay,
            )
            < 0
        ):
            raise ValueError("learning rates and feedback weights must be nonnegative")
        if not all(
            math.isfinite(x)
            for x in (
                self.fast_lr,
                self.projector_lr,
                self.projector_weight_decay,
                self.accepted_weight,
                self.rejected_weight,
            )
        ):
            raise ValueError("Reflex parameters must be finite")
        if self.projector_lr <= 0:
            raise ValueError("persistent A requires OPD_PROJECTOR_LR > 0")
        if self.update_interval < 1:
            raise ValueError("OPD_UPDATE_INTERVAL must be positive")
        if self.pipeline.startswith("selective") and self.dense_implementation == "gemm":
            raise ValueError("selective pipeline requires fused correction, not GEMM")
        if self.pipeline not in {"auto", "fused", "selective", "selective_masked"}:
            raise ValueError("invalid OPD_PIPELINE")
        if self.metrics_mode not in {"production", "instrumented"}:
            raise ValueError("invalid METRICS_MODE")
        if self.proposal_mode not in {"auto", "sparse", "dense"}:
            raise ValueError("invalid OPD_PROPOSAL_MODE")
        if self.dense_implementation not in {"auto", "fused", "gemm"}:
            raise ValueError("invalid OPD_DENSE_IMPLEMENTATION")

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {}
        for name in cls.__dataclass_fields__:
            key = (
                "OPD_"
                + {"projector_weight_decay": "projector_weight_decay"}.get(name, name).upper()
            )
            if name == "metrics_mode":
                key = "METRICS_MODE"
            raw = os.environ.get(key)
            if raw is not None:
                default = getattr(defaults, name)
                if isinstance(default, bool):
                    if raw.lower() not in {"0", "1", "true", "false"}:
                        raise ValueError(f"{key} must be 0/1/true/false")
                    values[name] = raw.lower() in {"1", "true"}
                else:
                    values[name] = type(default)(raw)
        if os.environ.get("OPD_TRAIN_PROJECTOR", "1") != "1":
            raise ValueError("dflash_reflex requires trainable A (OPD_TRAIN_PROJECTOR=1)")
        return cls(**values)
