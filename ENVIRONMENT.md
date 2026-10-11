# Isolated runtime

Do not install these packages in SpecNaacl's environment. The selected PR has a
newer runtime than many existing SpecNaacl installations. This machine's CUDA
12 / Torch 2.4 test environment is **not** the production runtime.

The authoritative dependency set is `third_party/nemo-rl/uv.lock`, from the
commit in `versions.lock`. It pins Torch 2.11.0+cu130, Triton 3.6.0,
Transformers 5.12.1, Ray 2.56.1 and vLLM 0.25.1. MCore uses the gitlinked
Megatron Bridge and its own gitlinked Megatron-LM; FlashAttention's wheel is
2.8.1+cu13torch2.10cxx11abitrue and TE is 2.15.0+42b8400. Do not substitute
similarly named wheels. The FlashAttention private interfaces are checked by
NVIDIA's DFlash module when imported.

Use Linux, a CUDA 13 development toolkit, Python **>=3.13.14,<3.14**, BF16-capable
GPU and a compatible driver. CUDA 13 GA's documented Linux toolkit driver is
>=580.65.06; see [NVIDIA release notes](https://docs.nvidia.com/cuda/archive/13.0.0/pdf/CUDA_Toolkit_Release_Notes.pdf).
The upstream development Dockerfile currently uses a CUDA 13.2 development base
while the locked PyTorch runtime is cu130. B200 is the intended training GPU.
Building TE/DeepEP requires the development toolkit and system build/RDMA
libraries; bare CUDA runtime images are insufficient.

## Install once on the GPU host

If the GPU host reaches pip/PyPI but cannot reach GitHub, use the staged
wheel/source procedure in [OFFLINE.md](OFFLINE.md) and
`bash scripts/setup_environment.sh --local-packages`. The ordinary command
below fetches remote sources and requires access to those hosts.

```bash
cd dflash
command -v uv
nvidia-smi
nvcc --version
# Set these to the host's installed CUDA development toolkit if needed:
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-100}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
bash scripts/setup_environment.sh
```

The setup script fetches exact source commits, initializes exact recursive
submodule gitlinks, applies Python hooks, then creates three independent locked
environments:

| Environment | Purpose | Upstream extra |
|---|---|---|
| `.venv` | Ray driver, datasets, rewards, reporting | base |
| `.venv-mcore` | DFlash pretraining and Megatron training actor | `mcore` |
| `.venv-vllm` | Inference actor and kernel benchmark | `vllm` |

NeMo explicitly declares the MCore/vLLM extras conflicting. They must not be
installed together. The driver sets each actor's absolute Python executable in
the upstream registry. The vLLM wheel's Python files are patched with a strict
0.25.1 version check; its compiled extensions stay intact. A later `uv sync`
that reinstalls the wheel requires rerunning that patch command. Launchers
never install packages or repair dependencies automatically.

The setup command is provided but has not been executed against CUDA 13 in
this workspace. Its success depends on access to upstream pinned wheels/git
dependencies, the compiler toolchain and their ABI compatibility. These remain
production qualification checks; the local tests do not establish them.

## Validate before reserving a long run

```bash
export PYTHONPATH="$PWD:$PWD/third_party/nemo-rl${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python scripts/validate_environment.py --require-runtime \
  --model /workspace/storage-shared/models/Qwen3-4B \
  --data /workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/train.parquet \
  --output outputs/environment.json
# After pretraining:
.venv/bin/python scripts/validate_environment.py \
  --model /workspace/storage-shared/models/Qwen3-4B \
  --draft outputs/pretrain/qwen3_4b/latest_checkpoint
```

Strict validation launches import probes using each environment, compares
versions, checks BF16/CUDA and imports the actual DFlash model/loss/proposer.
Model validation checks tokenizer/config fingerprints, architecture, block
length, and weight shapes from safetensors headers without loading weights.
Source-only validation (`python3 scripts/validate_environment.py`) needs no
Torch and checks source SHAs plus a read-only SpecNaacl checksum audit when
that reference exists.

## Existing managed environments

You may use already provisioned, matching environments:

```bash
export PYTHON_BIN=/absolute/base/bin/python
export DFLASH_MCORE_PYTHON=/absolute/mcore/bin/python
export DFLASH_VLLM_PYTHON=/absolute/vllm/bin/python
"$DFLASH_VLLM_PYTHON" scripts/patch_upstream.py --vllm-installed
bash pretrain_qwen3_4b.sh
```

Do not set `NEMO_RL_PY_EXECUTABLES_SYSTEM=1`; separate actor environments are
deliberate. All workers must see the same absolute project/output directories.
The supported launch topology is one GPU and one colocated inference owner.
Multi-engine, TP/PP/CP/SP and asynchronous scheduler modes are not qualified.

## Lightweight development validation

The temporary `.venv-test` here inherits Torch 2.4.1/Triton 3.0 from an existing
Python 3.12 environment, with isolated extra packages. It is excluded from Git
and was used only for CPU contracts, the original tiny HF DFlash model, and
actual Triton/rejection-kernel tests. Reproduce these with a separate compatible
Torch environment, `pytest`, Transformers 4.57.1, math-verify, pandas/pyarrow,
OmegaConf/Hydra and matplotlib; do not mistake this for a supported NeMo runtime.

## Final validation and compatibility

The optimization audit retains every pinned production dependency. Development
validation uses `.venv-test` (Python 3.12, Torch 2.4.1/CUDA 12.1, Triton 3.0)
on RTX 3090; it does not certify the production CUDA-13 stack. The new
`reference_qwen3_4b_config.json` records the official Qwen3-4B geometry (H=2560,
V=151936); the benchmark accepts `--model-config` to read your actual local file.

`FAIRNESS_PROFILE=native_dflash` is the default for actual training.
Explicit `specnaacl_matched` fails before loading models because the
pinned joint backend is not cross-framework equivalent. See the final report for native,
strict matched failure, paired measurement, and old-checkpoint commands.

Regenerate and check managed patches after fetching dependencies:

```bash
.venv/bin/python scripts/fetch_sources.py
.venv/bin/python scripts/patch_upstream.py
```

Installed vLLM wheel hooks also changed. Reapply them in the vLLM environment:

```bash
PYTHONPATH="$PWD" .venv-vllm/bin/python scripts/patch_upstream.py --vllm-installed
```

The patcher migrates the exact previously shipped hooks in place, with a
regression test against `tests/reference/vllm_legacy.patch`. It refuses
unrecognized source contexts; do not overwrite unrelated dependency changes. Source snapshots under
`patches/` are regenerated from the pinned checkouts. Runtime sidecars and
projector optimizer state remain compatible; old native loss settings require
`GRPO_OBJECTIVE=original_nemo BETA=0 EPSILON=.2` with the original batch settings.
