# Concurrent standalone Ray startup

The B200 report shows one training job advancing 20 iterations while seven
other one-GPU jobs last printed `Started a local Ray instance` and their GPUs
remained idle. This is not evidence of seven completed model initializations.
The pasted output contains no fatal exception and does not establish the exact
blocking stack or a host PID/CPU/RAM limit. Missing Transformers docstrings and
the pynvml deprecation do not terminate the shown process.

## Source evidence

The pinned NeMo `distributed/virtual_cluster.py:init_ray` first attempts auto
discovery and then starts a local cluster without a CPU or object-store budget,
with dashboard UI enabled. Every independent driver can therefore schedule
against the whole host and create a host-sized initial worker pool.

In [Ray 2.56.1 worker.py](https://github.com/ray-project/ray/blob/ray-2.56.1/python/ray/_private/worker.py),
the local-instance log precedes `connect(...)`. That line alone does not prove
that `ray.init()` returned. In
[Ray 2.56.1 configuration](https://github.com/ray-project/ray/blob/ray-2.56.1/src/ray/common/ray_config_def.h),
GCS client/server and system RPC reply thread pools default to one quarter of
the host CPU limit, independently of Ray's logical CPU budget. First-driver
worker prestarting is enabled. These are concrete scaling hazards for eight
independent clusters, but cannot establish the reported hang's unique cause.

## Direct source changes

- `dflash_reflex/ray_startup.py` uses `address="local"`, a unique short `/tmp`
  directory, bounded logical resources, and no dashboard UI. It never attaches
  to an unrelated cluster. It strips MPI/Slurm launcher variables as NeMo did,
  and removes `RAY_ADDRESS` and the driver CUDA-mask override from actor envs.
- Defaults: 8 logical CPUs, 1 GiB object store. Override with
  `DFLASH_RAY_NUM_CPUS` and `DFLASH_RAY_OBJECT_STORE_GB`. CPU budgets below the
  two-CPU colocated placement group or above process affinity fail early.
  Object spilling remains enabled; these settings do not impose OS CPU quotas.
- Before NeMo/Ray imports, the driver sets default RPC thread counts to 2:
  `RAY_num_server_call_thread`, `RAY_gcs_server_rpc_server_thread_num`,
  `RAY_gcs_server_rpc_client_thread_num`,
  `RAY_core_worker_num_server_call_thread`, and
  `RAY_worker_num_grpc_internal_threads`.
  `RAY_prestart_worker_first_driver=false` removes speculative initial workers;
  needed actors/workers still launch on demand. Explicit user overrides survive.
- `scripts/launch/common.sh` exports unbuffered Python output and defaults
  OMP/MKL/OpenBLAS/NumExpr to one thread, with tokenizer parallelism disabled.
- Each startup phase prints PID, GPU mask, and elapsed time. Every 120 seconds
  in a phase, faulthandler dumps waiting Python stacks without aborting. Timers
  are cancelled on phase exit, including failures.
- `<run>/logs/ray_startup.json` records PID, GPU mask, options, Ray temp directory,
  connection address and resources when ready. It never records the full env.

Settings apply equally to baseline and Reflex. Batch, accumulation, AAL,
checkpoint cadence, objective and model weights are unchanged. These changes
require source updates only, not a wheel transfer/rebuild, vendor patch, or
another pretrain. Restart stalled drivers individually; a healthy run can keep
training with its already-loaded source. Do not use global `ray stop` to restart
one run on a shared host.

## Validation and limits

`tests/test_ray_startup.py`, `tests/test_launchers.py`, and
`tests/test_vllm_runner_contract.py`: **93 passed**, 6.87 seconds.
The new tests include eight concurrent OS driver processes with Ray's native
API mocked, distinct socket directories, inherited GPU-mask preservation,
rejection of bad budgets, cleanup on failed startup, trace timer cancellation,
configuration before first NeMo import, and launcher environment settings.
Worker startup, configuration fairness, and environment validation regressions:
**37 passed** in 0.55 seconds. Total across the two checks: **130 passed**.
Ruff, shell syntax checks, and `git diff --check` also pass.
This validates the project's isolation/configuration contract, not live Ray
registration, actual CPU quotas, or B200 training. The development environment
does not contain the locked Ray 2.56.1 runtime. A live eight-job B200 startup
remains to be verified; new phase and stack logs distinguish a remaining hang.
