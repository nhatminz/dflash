"""CPU-only preflight; invoke using the mcore interpreter before loading vLLM."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dflash_reflex.dataset_helpers import compile_megatron_helpers


if __name__ == "__main__":
    compile_megatron_helpers(
        ROOT / "third_party/nemo-rl/3rdparty/Megatron-Bridge-workspace/"
        "Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets"
    )
