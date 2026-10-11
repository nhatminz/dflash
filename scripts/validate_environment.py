"""Report the real environment; strict mode checks separately locked actors."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def inspect_python(python, backend):
    code = """import importlib.metadata as m, json, platform
import torch
result = dict(python=platform.python_version(), torch=m.version("torch"), triton=m.version("triton"),
              transformers=m.version("transformers"), ray=m.version("ray"), cuda=torch.version.cuda,
              cuda_available=torch.cuda.is_available())
if torch.cuda.is_available():
    result.update(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                  bf16=torch.cuda.is_bf16_supported())
"""
    if backend == "mcore":
        code += """import flash_attn, megatron.core, megatron.bridge, transformer_engine.pytorch
from nemo_rl.models.megatron.draft.dflash import DFlashDraftModel
from nemo_rl.algorithms.loss.loss_functions import BlockDraftLossFn
result.update(flash_attn=m.version("flash-attn"), transformer_engine=m.version("transformer-engine"),
              megatron=True, dflash_model=True)
"""
    if backend == "vllm":
        code += """import vllm
from vllm.v1.spec_decode.dflash import DFlashProposer
from pathlib import Path
result.update(vllm=m.version("vllm"), dflash_proposer=True,
    reflex_patch="reflex.acceptance_buffer" in (Path(vllm.__file__).parent/"v1/sample/rejection_sampler.py").read_text())
"""
    code += "\nprint(json.dumps(result))"
    process = subprocess.run([str(python), "-c", code], capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"{backend} environment failed: {python}\n{process.stderr[-6000:]}")
    return json.loads(process.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-runtime", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--draft")
    parser.add_argument("--data")
    parser.add_argument("--output")
    args = parser.parse_args()
    lock = json.loads((ROOT / "versions.lock").read_text())
    report = dict(expected=lock, sources={}, specnaacl_unchanged=True)
    for name in ("nemo-rl", "vllm", "original-dflash"):
        repo = ROOT / "third_party" / name
        sha = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if sha != lock[name]["sha"]:
            raise RuntimeError(f"unexpected {name} commit: {sha}")
        report["sources"][name] = sha
    original = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())["SpecNaacl_files"]
    changed = [
        file
        for file, digest in original.items()
        if not (ROOT.parent / file).is_file()
        or hashlib.sha256((ROOT.parent / file).read_bytes()).hexdigest() != digest
    ]
    report["specnaacl_unchanged"] = not changed
    report["changed_reference_files"] = changed
    # The reference audit is read-only and is not a runtime dependency.
    if args.model:
        from dflash_reflex.model_contract import fingerprint, validate_checkpoint

        report["target_fingerprint"] = fingerprint(args.model)
        if args.draft:
            config = json.loads((Path(args.draft) / "config.json").read_text())
            block = config["dflash_config"].get("block_size", config.get("block_size"))
            validate_checkpoint(args.model, args.draft, block)
            report["draft_contract"] = "valid"
    if args.data and not Path(args.data).is_file():
        raise FileNotFoundError(args.data)
    if args.require_runtime:
        environments = dict(
            base=sys.executable,
            mcore=os.environ.get("DFLASH_MCORE_PYTHON", ROOT / ".venv-mcore/bin/python"),
            vllm=os.environ.get("DFLASH_VLLM_PYTHON", ROOT / ".venv-vllm/bin/python"),
        )
        report["environments"] = {}
        for backend, python in environments.items():
            state = inspect_python(python, backend)
            for package in ("torch", "triton", "transformers", "ray"):
                if state[package] != lock["runtime"][package]:
                    raise RuntimeError(
                        f"{backend}: {package}={state[package]}, expected {lock['runtime'][package]}"
                    )
            if (
                not state["python"].startswith("3.13.")
                or tuple(map(int, state["python"].split("."))) < (3, 13, 14)
                or state["cuda"] != "13.0"
                or not state.get("bf16")
            ):
                raise RuntimeError(f"incompatible Python/CUDA/BF16: {state}")
            if backend == "vllm" and (
                state["vllm"] != lock["vllm_version"] or not state["reflex_patch"]
            ):
                raise RuntimeError("vLLM version or Reflex patch mismatch")
            if backend == "mcore":
                for package, key in (
                    ("flash_attn", "flash-attn"),
                    ("transformer_engine", "transformer-engine"),
                ):
                    if state[package].lower() != lock["runtime"][key].lower():
                        raise RuntimeError(f"mcore: {package} differs from the NeMo lock")
            report["environments"][backend] = state
    result = json.dumps(report, indent=2)
    print(result)
    if args.output:
        Path(args.output).write_text(result + "\n")


if __name__ == "__main__":
    main()
