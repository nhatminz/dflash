# DFlash + trainable ReflexOPD

Standalone sibling of SpecNaacl. Production entry points use NVIDIA NeMo RL
PR #3712's **actual Megatron DFlash model and BlockDraftLossFn**, with the
compatible vLLM 0.25.1 DFlash proposer and rejection sampler. No training module
imports SpecNaacl. The baseline and Reflex methods share pretraining, online
draft co-training, target policy training, and speculative sampling.

Read [huongdanchay.md](huongdanchay.md) for Vietnamese commands,
[ENVIRONMENT.md](ENVIRONMENT.md) for isolated installation, and
[IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) for integration details
and the limits of validation.

For a B200 host with pip index access but blocked GitHub, transfer the staged
external wheels and pinned source checkouts in `offline_packages/` and
`third_party/`, then use `setup_environment.sh --local-packages`.
See [OFFLINE.md](OFFLINE.md) for the complete transfer and install commands.

**Validation status:** CPU tests and CUDA kernel/tiny DFlash tests have run on
an RTX 3090. The full CUDA 13 NeMo/Megatron/vLLM training path has not run here:
the machine lacks that runtime, B200, and the specified local models/data.
This is an implementation with bounded correctness tests, not a measured
B200 speedup or a claim of completed production qualification.

**B200 rollout stability update:** both training methods now default to
`VLLM_ENFORCE_EAGER=1` for target and draft inference. This is a conservative
bypass for the unqualified graph/refit path implicated by a crash at the
512-token graph boundary, not a confirmed kernel root cause. It can reduce
throughput; compare both methods with the same setting. Reflex now returns
an owned, contiguous draft-token tensor instead of a view of its reusable
top-k workspace. Failed rollouts preserve the original exception without
requesting CUDA feedback/RNG from a failed engine. See
[the investigation and validation](validation/b200_rollout_memory_fix.md).

**Runner integration fix:** `VLLM_USE_V2_MODEL_RUNNER=0` is required for both
methods. The pinned vLLM can automatically choose V2 for dense Qwen models,
which bypasses the project's V1-only Reflex proposal/feedback hooks. Eager mode
does not select V1. Launchers now pin V1, reject V2 overrides, verify the actual
worker class, and refuse multi-token Reflex rollouts with inactive hooks.
Missing verification ratios are `null`/`N/A`, never fabricated zeros. Runs made
with inactive Reflex hooks are not valid Reflex experiments; restart from the
same pretrain checkpoint. See [the AAL investigation](validation/reflex_runner_aal_fix.md).


**Final correctness update:** `FAIRNESS_PROFILE=native_dflash` is the default
for all launchers. Explicit `specnaacl_matched` **fails preflight** on this pinned integration. Seven
independent target LoRA adapters, the requested four-rollout target cadence,
and target-only constant-reward filtering are not equivalent to the joint
Megatron recipe. A successful dry run never conceals these blockers. The local
SpecNaacl source actually steps its target on each nonempty filtered rollout;
`ACCUMULATION_STEPS=4` contributes to its step label, not a four-rollout readiness
condition. The native profile is not fully equivalent to SpecNaacl.

The revised native objective maps `BETA=.04`, `EPSILON=.1` into the real loss,
uses a sequence mean and SpecNaacl's k3 gradient without NeMo's score factor.
Synthetic tests execute the actual patched NeMo loss against the extracted
SpecNaacl loss. Full target fine-tuning and sample/mask differences remain.
For the original native objective or an old checkpoint, use
`GRPO_OBJECTIVE=original_nemo BETA=0 EPSILON=.2`; keep all original run overrides.
Mismatched resume settings are rejected, and checkpoint files are preserved.

`METRICS_MODE=production` is default. Baseline has no GPU acceptance buffer,
Reflex state, or optional forward wrapper. Missing AAL is null. For AAL studies
run **both** methods with `METRICS_MODE=instrumented`; for throughput run both
with production mode. Never compare a profiled baseline with production Reflex.

See [the final report](validation/final_reflex_optimization_report.md),
[fairness matrix](validation/fairness_config_matrix.json), and
[measured complete OPD timings](validation/benchmark_before_after.json).
There is no claim of B200/live-engine/GRPO qualification.

## Sources

| Dependency | Exact audited commit |
|---|---|
| NVIDIA DFlash PR branch | `c18e462adf34739951b3d21773f436eaee324b31` |
| vLLM 0.25.1 | `752a3a504485790a2e8491cacbb35c137339ad34` |
| Original DFlash | `07ebd93db9f472af339b644bb70221ad8428328a` |

Pins live in `versions.lock`; dependencies use the NeMo commit's `uv.lock`.
`scripts/fetch_sources.py` reproduces the checkouts and applies guarded,
idempotent patches. Checkouts are present locally but ignored in the outer Git
repository to avoid accidental gitlinks. A fresh copy must fetch them once.

Primary sources: [issue #3698](https://github.com/NVIDIA-NeMo/RL/issues/3698),
[PR #3712](https://github.com/NVIDIA-NeMo/RL/pull/3712),
[its EAGLE infrastructure dependency #3699](https://github.com/NVIDIA-NeMo/RL/pull/3699),
[PR #4059](https://github.com/NVIDIA-NeMo/RL/pull/4059),
[PR #4060](https://github.com/NVIDIA-NeMo/RL/pull/4060),
[pinned NVIDIA DFlash](https://github.com/MarkXCloud/RL/blob/c18e462adf34739951b3d21773f436eaee324b31/nemo_rl/models/megatron/draft/dflash.py),
[pinned vLLM proposer](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/spec_decode/dflash.py),
[original DFlash](https://github.com/z-lab/dflash).

## Quick start

```bash
cd dflash
# Clear an old shell override to use the native_dflash default:
unset FAIRNESS_PROFILE
DRY_RUN=true bash train_qwen3_4b.sh
# On the CUDA-13/B200 host, install once:
bash scripts/setup_environment.sh
bash pretrain_qwen3_4b.sh
DATASET=simplelr bash train_qwen3_4b.sh
DATASET=simplelr bash train_qwen3_4b_dflash.sh
RESUME=auto DATASET=simplelr bash train_qwen3_4b.sh
```

The SpecNaacl experiment set has six model-specific wrapper families:
Qwen2.5-1.5B/3B/7B/14B-Instruct and Qwen3-1.7B/4B. Each has
`pretrain_<key>.sh`, `train_<key>.sh` (Reflex), and `train_<key>_dflash.sh`
(baseline). The existing Qwen3-8B wrappers remain optional. Model/data paths
are checked locally; nothing downloads a substitute model or dataset.
Each wrapper fixes `MODEL_KEY` to the model named in its filename. `MODEL`
remains an explicit path override; a stale exported `MODEL` takes precedence
over the model config's default. Launchers reject recognized Qwen directory
names belonging to another key before creating run files or checking CUDA.
For custom directory names, the existing architecture/checkpoint validation
still applies; the directory-name guard does not prove model size.
The six-model pretraining default is now physical batch 8 per GPU, with
accumulation 4 for all six models: effective batch 32 on one GPU, as requested.
This changes their previous effective batches and no longer matches SpecNaacl
pretraining defaults; align both projects explicitly for controlled comparisons.
GRPO settings are unchanged. Start a new run when changing these settings;
old checkpoints require their original batch/accumulation to resume.
Qwen2.5 retains its original biased target attention. Its headless DFlash
drafter is independently initialized with bias-free Qwen3-style attention,
the target's hidden/head/GQA dimensions, and shared target embedding/LM head.
MASK 151669 must be an unused embedding row in the actual local tokenizer.
The new family has tiny forward/backward/export tests; full B200 runs remain
unverified. See [six-model commands](huongdanchay.md#chạy-đồng-thời-6-model)
and [the support report](validation/qwen25_support_report.md).
An external z-lab Qwen3-8B-b16 checkpoint is accepted only after architecture,
tokenizer vocabulary, MASK row, block-size and safetensors-header validation.
The official MASK is unused embedding row **151669**, not `<|fim_pad|>`.

## Reflex contract

* `H` is the final normalized MASK-slot state passed into DFlash's shared target
  LM head; `U=H@A`, `z'=z+U@B.T`.
* `A` is FP32, head-aligned QR initialized, persistent and trainable. A named
  `draft_opd_projector` AdamW group receives the accumulated **rollout** gradient
  at successful Megatron optimizer boundaries. It has an independent constant
  scheduler and checkpointed optimizer state. Ordinary draft gradients stay in
  NVIDIA's joint policy/draft optimizer.
* One FP32 `B` belongs to one complete `LLM.generate` cohort. All prompts and
  responses in that call share it. Request completion never resets it. It resets
  at the next cohort boundary, using reusable GPU storage.
* Feedback is accepted prefix plus first rejection, excluding later slots and
  bonus. Counts come from the sampler's decision branches. EOS, remaining token
  limits, padded drafts and discarded requests mask contributions.
* Teacher `p` is the verifier's already processed, temperature/top-p constrained
  probability tensor. The draft KL surrogate is full-vocabulary normalized;
  its top-k union has a tail bucket. As in SpecNaacl, the reported KL includes
  the tail but the operational gradient is union-only `w*(q-p)`.
* A-gradient uses `B_t` **before** the sparse atomic B update. No autograd or
  additional transformer forward runs in verification.
* Proposals remain greedy. Actual verifier draft probability remains one-hot
  (`draft_probs=None`); the KL surrogate is never passed as proposal probability.

The supported GRPO topology is one colocated trainer GPU and one vLLM engine,
TP=PP=CP=1, no sequence packing/SP, synchronous scheduling. Splitting a cohort
across independent engines would change the shared-B algorithm and is rejected.
The projector aggregation primitive also has a two-rank Gloo correctness test;
this does not advertise multi-GPU GRPO support.

## Layout

```text
dflash/
  README.md ENVIRONMENT.md IMPLEMENTATION_REPORT.md huongdanchay.md
  versions.lock SOURCE_MANIFEST.json requirements.txt pyproject.toml
  pretrain_qwen3_{1p7b,4b,8b}.sh
  pretrain_qwen25_{1p5b,3b,7b,14b}.sh
  train_qwen3_{1p7b,4b,8b}{,_dflash}.sh
  train_qwen25_{1p5b,3b,7b,14b}{,_dflash}.sh
  configs/{_shared,qwen3_1p7b,qwen3_4b,qwen3_8b}/
  dflash_reflex/
    pretrain.py train.py nemo_data.py data.py rewards.py model_contract.py
    adapter.py kernels.py feedback.py rollout_state.py vllm_runtime.py
    projector_training.py nemo_bridge.py checkpointing.py metrics.py
    block_training.py config.py
  scripts/
    fetch_sources.py setup_environment.sh patch_upstream.py prepare_run.py
    validate_environment.py prepare_data.py benchmark_rollout.py
    summarize_run.py plot_aal.py plot_training_time.py launch/
  third_party/{nemo-rl,vllm,original-dflash}/
  third_party/reference_*.json
  tests/ validation/ outputs/{pretrain,train,benchmarks}/
```

## Logging and checkpoints

Every DataLoader iteration writes one `logs/rollout_timing.csv` row, with AAL,
acceptance and end-to-end times; JSONL adds upstream training metrics and optional
CUDA profiling. Ratios are formed from cumulative token/round sums on resume.
`summary.json` includes total wall time, rollout time, throughput, actual
successful optimizer-step counts and Reflex totals. Unavailable separate draft
and target timings are explicitly null because NVIDIA runs them jointly.

Pretraining exports a headless HF safetensors checkpoint plus its config,
manifest and initial A; the resumable FP32 master/optimizer/RNG state is separate.
GRPO uses NeMo's atomic `checkpoints/resume/step_*/policy/weights` transaction
for target and attached draft, plus `reflex_rank0.pt`, `dflash_stats_rank0.pt`
and `dflash_driver.pt` sidecars. B is never checkpointed. Empty `target/` and
`draft/` directories in a GRPO run are reserved layout entries; the actual
joint weights are under `resume/`, not duplicated into those directories.

## Tests and bounded profiling

```bash
PYTHONPATH="$PWD:$PWD/third_party/nemo-rl" .venv-mcore/bin/python -m pytest tests -q
PYTHONPATH="$PWD" .venv-vllm/bin/python scripts/benchmark_reflex_hotpath.py \
  --model-config /workspace/storage-shared/models/Qwen3-4B/config.json \
  --requests 1 8 32 64 --repeats 15 \
  --output outputs/benchmarks/b200_hotpath.json \
  --cache outputs/benchmarks/b200_dispatch.json
OPD_DISPATCH_CACHE="$PWD/outputs/benchmarks/b200_dispatch.json" \
  bash train_qwen3_4b.sh
```

Microbenchmark cache keys include GPU, exact row/hidden/vocabulary/rank/top-k
shape and dtype. GPU dispatch reads actual unique active_count. Unmatched shapes
use sparse fused correction and the fused-statistics proposal; there is no automatic tuning sweep. Timings from this utility
exclude transformer forwards and cannot establish end-to-end GRPO speedup.
`OPD_PROFILE=1` adds CUDA event timing, resolved only at the cohort boundary.

`OPD_PIPELINE=auto` selects measured `fused`, `selective` or `selective_masked`
paths by GPU/shape and the last completed cohort's feedback density. An unseen
GPU defaults to the conservative fused path. Calibrate B200 separately; the
RTX 3090 cache is evidence for that GPU only. Selected statistics retain the
original logits until verification (no extra LM-head evaluation or B copy).
This extends logits lifetime: BF16 64 requests x 15 slots x 151936 vocabulary
is approximately 278 MiB (556 MiB if the actual logits are FP32). Teacher ties
follow stable token-ID ordering.

Calibration also distinguishes hidden/logit dtypes. If a model returns FP32
logits for BF16 states, calibrate with `--logits-dtype float32`; a BF16 bucket
is not reused for that path. Explicit dense GEMM feedback reuses its exact
corrected proposal buffer. GEMM remains excluded from automatic selection.

Inspect strict cross-framework compatibility without GPU/model weights:

```bash
python3 scripts/compare_specnaacl_config.py --model qwen3_4b --dataset simplelr \
  --profile specnaacl_matched
# Exit 2 is expected until the reported compatibility blockers are resolved.
DATASET=simplelr FAIRNESS_PROFILE=specnaacl_matched METHOD=dflash_reflex bash train_qwen3_4b.sh
DATASET=simplelr FAIRNESS_PROFILE=specnaacl_matched METHOD=dflash bash train_qwen3_4b_dflash.sh
```
