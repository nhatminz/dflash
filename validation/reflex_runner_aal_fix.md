# Reflex inactive behind auto-selected vLLM V2 — 2026-10-12

The user's iteration 13 has 33,386 generated tokens, 64 responses, 13 successful
target/draft optimizer steps, but zero verification/proposal counters, zero OPD
feedback/updates, A version 0, and no active B rows. These data confirm that the
labelled Reflex run has not exercised its inference adaptation. AAL=0 is also
a reporting bug: the ratio helper returned zero for a zero denominator.

Baseline `N/A` is deliberate in production mode, which does not collect core
speculative acceptance metrics. To compare AAL, use instrumented for both.

## Source-confirmed integration defect

Pinned vLLM `VllmConfig.use_v2_model_runner` automatically chooses V2 for
supported dense Qwen architectures when `VLLM_USE_V2_MODEL_RUNNER` is unset.
DFlash is a supported V2 speculative method. Setting eager does not select V1.
The project's proposal, runner-context and rejection-feedback patches target
only the V1 files. V2 uses separate runner, speculator and sampler modules,
therefore bypassing those hooks while continuing ordinary DFlash training.
The provided JSON does not include the actual worker class, so it proves
inactive adaptation rather than independently identifying the live class.
The automatic V2 path is a concrete sufficient cause found in the pinned code.

The previous eager/owned-buffer tests covered local Reflex components, not
live runner selection. They did not establish that the B200 engine used the
patched proposal and sampler. The missing runner pin was an integration error.

## Changes

- Pin V1 for both methods before importing NeMo/vLLM or launching actors.
  Reject V2 overrides. Use the same backend for a controlled comparison.
- Verify the real worker's runner module at cohort start, before any projector
  loading or adaptation. Print it once for the engine. Reject an unsupported
  vLLM class or import-time disabled hooks.
- Preflight checks all three installed V1 hook sites, rather than only one
  sampler marker. No upstream/native rebuild is involved.
- Count proposal and verification hook calls per cohort and include them,
  along with the actual runner module, in `logs/metrics.jsonl` under
  `rollout_hook_health`. Reject multi-token Reflex cohorts without both hooks
  before publishing feedback or starting policy training on that cohort.
- Record undefined AAL/acceptance ratios as null/N/A. Real zero draft acceptance
  with valid rounds remains acceptance=0 and AAL=1 when each round emits a
  recovered target token.
- Record V1 in the resume runtime contract. Legacy runs with unknown runner
  selection require a fresh run; do not silently mix runner configurations.

## Verification and limits

117 selected tests pass on the local CPU/RTX 3090 environment. Coverage includes
executing the pinned vLLM runner-selection methods extracted by AST for
Qwen2/Qwen3 with eager on/off, explicit runner pinning, rejection of a live V2
class before CUDA work, complete hook-site checking, zero-denominator metrics,
blocked feedback from inactive Reflex, instrumented baseline counting without
an adapter, and the existing real CUDA runtime/compaction tests.

The runner-selection tests stub the supported-feature result; source review
confirmed DFlash is supported with this project's single-GPU settings. These
tests do not instantiate the full CUDA-13 vLLM/Megatron stack. No B200 full-run
success or throughput result is claimed.

## Existing B200 runs

Do not count the supplied run as a Reflex experiment or resume it to mix
inactive and active adaptation. Existing files/checkpoints are preserved.
Restart baseline and Reflex from the same pretrain checkpoint with the same
runner/eager/metrics settings for comparison. No pretraining or dependency
installation needs repeating. The Vietnamese runbook contains the launch and
idempotent installed-Python patch commands.
