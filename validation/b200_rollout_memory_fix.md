# B200 rollout CUDA failure investigation — 2026-10-12

The supplied log passed dependency validation, Megatron dataset compilation,
physical/logical GPU selection and Ray memory reporting. During GRPO step 1,
vLLM generated many tokens before a CUDA illegal memory access caused a native
abort. Ray's ActorDiedError is a downstream result. No OOM was established.
The original failing kernel is not identified by the CUDAEvent/TensorImpl
destructor stack.

At failure, the scheduler had 32 requests with 16 scheduled tokens each:
512 tokens, exactly the configured maximum CUDA-graph capture size. Previous
larger decode batches ran outside this limit. This motivates an eager bypass;
it does **not** prove graph replay caused the illegal access.

## Direct source changes

- Default target **and draft** inference to eager for both baseline and Reflex.
  The opt-out is `VLLM_ENFORCE_EAGER=0`. This trades possible throughput for a
  simpler execution path; eager has not yet been qualified on the user's B200.
  The setting is recorded in resolved/launch/run configuration and printed
  before setup. Existing resume configuration checks enforce consistency.
- Return an owned, contiguous argmax vector. Previously `ids[:, 0]` aliased
  a reusable proposal workspace and had stride `topk`. vLLM retains draft
  tensors and copies them on a separate stream; selective feedback can also
  reuse the proposal workspace. Cloning the small token vector removes this
  ownership hazard, including `topk=1` where `contiguous()` alone would alias.
  This is a concrete lifetime fix, not proof it caused this particular crash.
- Validate host request counts, variable draft lengths, output capacity and
  teacher/draft vocabulary/layout before Reflex GPU address arithmetic.
  Variable verification widths near completion remain supported. These checks
  do not introduce GPU-to-host synchronization.
- On `LLM.generate` failure, preserve the original exception and write a CPU
  traceback report if possible. Do not call success-only `finish_rpc`, which
  reads CUDA gradients/counters/RNG and can raise a second exception on a failed
  engine. Partial cohorts never publish feedback. A native abort can still
  prevent Python from recording the engine's original kernel traceback.

No dependency installation, native rebuild, new wheel, or ZIP patch is needed.
No model/data files, SpecNaacl code, or existing checkpoints were modified.

## Validation

Environment: RTX 3090, Python 3.12, Torch 2.4.1, Triton 3.0.0. This is not the
B200's Python 3.13 / Torch 2.11 / Triton 3.6 / vLLM 0.25.1 environment.

84 tests passed with normal asynchronous CUDA execution (no launch blocking):

```bash
.venv-test/bin/python -m pytest -q \
  tests/test_runtime.py tests/test_runtime_alignment.py \
  tests/test_runtime_buffers.py tests/test_generation_failure.py \
  tests/test_launchers.py tests/test_config_fairness.py
```

Coverage includes owned draft outputs for fused/selective pipelines and top-k
1/5; real Qwen vocabulary 151936 with rank 8/top-k 16/block 16 and separate
feedback stream; variable draft lengths; reordered requests; shrinking batches
64 → 54 → 32 → 17 → 4 → 1; original-exception preservation even if reporting
storage is unavailable; and symmetric eager/graph opt-out configuration for all
six model families and both methods.

An additional manual CUDA stress run completed 90 rounds each for fused and
selective pipelines with the same real vocabulary, request compaction and
asynchronous feedback. Both produced finite projector gradients. These tests
exercise Reflex kernels/runtime, not a live vLLM target/draft transformer or
Megatron optimizer. Full GRPO and long-run B200 stability remain unverified.

## B200 run

Update the tracked project source on B200, activate its existing offline
environment and launch a fresh process:

```bash
cd /workspace/storage-shared/nlp/minhpn19/dflash
source offline_packages/activate.sh
CUDA_VISIBLE_DEVICES=2 VLLM_ENFORCE_EAGER=1 bash train_qwen3_1p7b.sh
```

Baseline uses `train_qwen3_1p7b_dflash.sh` with the same eager setting. Keep the
existing model/data/checkpoint overrides if applicable. Batch 8, accumulation
4, checkpoint interval 50 remain defaults. Never compare the new eager run
against an old graph-enabled run as a controlled throughput experiment.

If eager also fails, a separate diagnostic launch with
`CUDA_LAUNCH_BLOCKING=1` can surface the kernel closer to the failing launch;
its timing is not a performance measurement. Do not restart setup or rebuild
libraries in response to this runtime error.
