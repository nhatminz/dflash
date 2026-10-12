"""Apply narrowly scoped hooks to the exact vendored commits, idempotently.

Also supports --vllm-installed to patch the wheel's Python files without
rebuilding its C++/CUDA extensions. The wheel version MUST be 0.25.1.
"""

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace(path, before, after, count=1):
    source = path.read_text()
    if after in source:
        return
    found = source.count(before)
    if found != count:
        raise RuntimeError(f"upstream API drift: {path}: expected {count} matches, found {found}")
    path.write_text(source.replace(before, after, count))


def migrate_legacy_vllm(root):
    """Migrate only the exact previously shipped hooks; preserve other source."""
    for relative in (
        "v1/spec_decode/llm_base_proposer.py",
        "v1/worker/gpu_model_runner.py",
        "v1/sample/rejection_sampler.py",
    ):
        path = root / relative
        source = path.read_text()
        source = source.replace(
            "        from dflash_reflex.vllm_runtime import current\n        reflex = current()\n",
            "        reflex = current() if HOOKS_ENABLED else None\n",
        )
        source = source.replace(
            "    from dflash_reflex.vllm_runtime import current\n    reflex = current()\n",
            "    reflex = current() if HOOKS_ENABLED else None\n",
        )
        source = source.replace(
            "\n            from dflash_reflex.vllm_runtime import profiled_forward\n"
            '            ret_hidden_states = profiled_forward("dflash_forward_ms", self.model, **model_kwargs)',
            "\n            ret_hidden_states = self.model(**model_kwargs)",
        )
        source = source.replace(
            "        from dflash_reflex.vllm_runtime import profiled_forward\n"
            '        return profiled_forward("target_forward_ms", self.model,\n',
            "        return self.model(\n",
        )
        if source != path.read_text():
            path.write_text(source)


def patch_vllm(root):
    migrate_legacy_vllm(root)
    for relative in (
        "v1/spec_decode/llm_base_proposer.py",
        "v1/worker/gpu_model_runner.py",
        "v1/sample/rejection_sampler.py",
    ):
        replace(
            root / relative,
            "import torch\n",
            "import torch\nfrom functools import partial\n"
            "from dflash_reflex.vllm_runtime import HOOKS_ENABLED, PROFILE_ENABLED, current, profiled_forward\n",
        )
    proposer = root / "v1/spec_decode/llm_base_proposer.py"
    replace(
        proposer,
        "\n            ret_hidden_states = self.model(**model_kwargs)",
        '\n            ret_hidden_states = (profiled_forward("dflash_forward_ms", self.model, **model_kwargs)\n'
        "                                 if PROFILE_ENABLED else self.model(**model_kwargs))",
        count=1,
    )
    replace(
        proposer,
        '        """Greedy-sample draft tokens from hidden states."""\n',
        '        """Greedy-sample draft tokens from hidden states."""\n'
        "        reflex = current() if HOOKS_ENABLED else None\n"
        '        if self.method == "dflash" and reflex is not None and reflex.enabled:\n'
        "            return reflex.propose(self.model, hidden_states)\n",
    )
    runner = root / "v1/worker/gpu_model_runner.py"
    replace(
        runner,
        "        return self.model(\n            input_ids=input_ids,\n",
        '        return (partial(profiled_forward, "target_forward_ms", self.model)\n'
        "                if PROFILE_ENABLED else self.model)(\n"
        "            input_ids=input_ids,\n",
    )
    replace(
        runner,
        "        sampling_metadata = self.input_batch.sampling_metadata\n"
        "        # Update output token ids",
        "        sampling_metadata = self.input_batch.sampling_metadata\n"
        "        reflex = current() if HOOKS_ENABLED else None\n"
        "        if reflex is not None:\n"
        "            reflex.context(self)\n"
        "        # Update output token ids",
    )
    sampler = root / "v1/sample/rejection_sampler.py"
    replace(
        sampler,
        "    # Create output buffer.\n",
        "    reflex = current() if HOOKS_ENABLED else None\n"
        "    if reflex is not None and synthetic_mode:\n"
        '        raise ValueError("Reflex requires actual rejection sampling")\n'
        "    feedback_accept = reflex.acceptance_buffer(batch_size) if reflex is not None else None\n"
        "    # Create output buffer.\n",
    )
    replace(
        sampler,
        "            SYNTHETIC_MODE=synthetic_mode,\n        )\n"
        "        if sampling_metadata.all_greedy:\n            return output_token_ids",
        "            SYNTHETIC_MODE=synthetic_mode,\n"
        "            accepted_prefix_ptr=feedback_accept,\n        )\n"
        "        if sampling_metadata.all_greedy:\n"
        "            if reflex is not None:\n"
        "                reflex.verified(target_logits, feedback_accept,\n"
        '                                type("BlockMetadata", (), dict(num_draft_tokens=num_draft_tokens, draft_token_ids=draft_token_ids))(),\n'
        "                                output_token_ids, greedy=True)\n"
        "            return output_token_ids",
    )
    replace(
        sampler,
        "        NO_DRAFT_PROBS=draft_probs is None,\n"
        "        SYNTHETIC_MODE=synthetic_mode,\n    )\n    return output_token_ids",
        "        NO_DRAFT_PROBS=draft_probs is None,\n"
        "        SYNTHETIC_MODE=synthetic_mode,\n"
        "        accepted_prefix_ptr=feedback_accept,\n    )\n"
        "    if reflex is not None:\n"
        "        if not sampling_metadata.all_random:\n"
        '            raise ValueError("This cohort must have a uniform target sampling mode")\n'
        "        reflex.verified(target_probs, feedback_accept,\n"
        '                        type("BlockMetadata", (), dict(num_draft_tokens=num_draft_tokens, draft_token_ids=draft_token_ids))(),\n'
        "                        output_token_ids)\n"
        "    return output_token_ids",
    )
    # Both kernels write the prefix from the decision branch. No output equality.
    replace(
        sampler,
        "    SYNTHETIC_MODE: tl.constexpr,\n):\n    req_idx = tl.program_id(0)",
        "    SYNTHETIC_MODE: tl.constexpr,\n"
        "    accepted_prefix_ptr=None,\n):\n    req_idx = tl.program_id(0)",
        count=2,
    )
    replace(
        sampler,
        "    rejected = False\n    for pos in range(num_draft_tokens):",
        "    rejected = False\n    accepted_count = 0\n    for pos in range(num_draft_tokens):",
        count=2,
    )
    replace(
        sampler,
        "                rejected = draft_token_id != target_argmax_id\n            tl.store(",
        "                rejected = draft_token_id != target_argmax_id\n"
        "            accepted_count += (not rejected).to(tl.int32)\n"
        "            tl.store(",
    )
    replace(
        sampler,
        "            if accepted:\n                token_id = draft_token_id",
        "            if accepted:\n                accepted_count += 1\n                token_id = draft_token_id",
    )
    replace(
        sampler,
        "    if not rejected:\n        # If all tokens are accepted, append the bonus token.",
        "    if accepted_prefix_ptr is not None:\n"
        "        tl.store(accepted_prefix_ptr + req_idx, accepted_count)\n"
        "    if not rejected:\n        # If all tokens are accepted, append the bonus token.",
        count=2,
    )


def patch_nemo(root):
    memory_tracker = root / "utils/memory_tracker.py"
    replace(
        memory_tracker,
        "from ray.scripts.scripts import memory_summary\n",
        "from ray import get_runtime_context\n"
        "from ray.scripts.scripts import memory_summary\n",
    )
    replace(
        memory_tracker,
        "        ray_memory_summary = memory_summary(stats_only=True, num_entries=5)\n",
        "        # Independent training jobs can own separate local Ray clusters.\n"
        "        # Query this driver's connected cluster instead of auto-discovery.\n"
        "        ray_memory_summary = memory_summary(\n"
        "            address=get_runtime_context().gcs_address,\n"
        "            stats_only=True, num_entries=5,\n"
        "        )\n",
    )
    replace(
        root / "models/megatron/setup.py",
        "    megatron_cfg.dist.external_gpu_device_mapping = True\n"
        "    initialize_megatron(\n",
        "    megatron_cfg.dist.external_gpu_device_mapping = True\n"
        "    from dflash_reflex.dataset_helpers import configure_megatron_dataset_builder\n"
        "    configure_megatron_dataset_builder()\n"
        "    initialize_megatron(\n",
    )
    worker = root / "models/generation/vllm/vllm_worker.py"
    replace(
        worker,
        "        from nemo_rl.distributed.numa_utils import bind_to_gpu_numa\n",
        "        # Placement-group slot zero is shared by independent drivers.\n"
        "        # Lease a port window before vLLM caches environment variables.\n"
        "        if bundle_indices is not None and len(bundle_indices) == 1:\n"
        "            from dflash_reflex.worker_runtime import acquire_vllm_port\n"
        "            self._dflash_vllm_port_lease = acquire_vllm_port()\n"
        '            os.environ["VLLM_PORT"] = str(self._dflash_vllm_port_lease.port)\n'
        '            print(f"DFlash vLLM port window: {self._dflash_vllm_port_lease.port}", flush=True)\n'
        "\n"
        "        from nemo_rl.distributed.numa_utils import bind_to_gpu_numa\n",
    )
    replace(
        worker,
        "        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)",
        "        from dflash_reflex.nemo_bridge import generate_cohort\n"
        "        outputs = generate_cohort(self.llm, prompts, sampling_params, use_tqdm=use_tqdm)",
    )
    replace(
        worker,
        "            seed=seed,\n",
        '            seed=int(os.environ["TRAIN_SUBSET_SEED"]),\n',
    )
    worker = root / "models/policy/workers/megatron_policy_worker.py"
    replace(
        worker,
        "        gpu_ids = ray.get_gpu_ids()\n"
        "        local_rank = int(gpu_ids[0])\n",
        "        gpu_ids = ray.get_gpu_ids()\n"
        "        from dflash_reflex.worker_runtime import policy_cuda_device\n"
        "        local_rank = policy_cuda_device(gpu_ids, os.environ.get(\"CUDA_VISIBLE_DEVICES\"))\n",
    )
    replace(
        worker,
        "        torch.cuda.set_device(local_rank)\n",
        '        print(f"DFlash policy GPU: physical={gpu_ids[0]}, cuda={local_rank}", flush=True)\n'
        "        torch.cuda.set_device(local_rank)\n",
    )
    replace(
        worker,
        "        # local_rank (== ray.get_gpu_ids()[0]) is the physical GPU index that\n"
        "        # keys the affinity file. Pass it explicitly: CUDA_VISIBLE_DEVICES lists\n"
        "        # all node devices here (RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1,\n"
        "        # set by configure_worker), so it can't identify this worker's GPU.\n"
        "        bind_to_gpu_numa(local_rank)\n",
        "        # NUMA affinity uses the physical GPU ID; CUDA uses its masked ordinal.\n"
        "        bind_to_gpu_numa(int(gpu_ids[0]))\n",
    )
    replace(
        worker,
        "        self.draft_model = model_and_optimizer_state.draft_model\n",
        "        self.draft_model = model_and_optimizer_state.draft_model\n"
        "        from dflash_reflex.nemo_bridge import initialize_trainer\n"
        "        self.reflex_trainer = initialize_trainer(self, pretrained_path, init_optimizer)\n",
    )
    replace(
        worker,
        "        self._setup_colocated_cuda_graph_managers()\n",
        "        self._setup_colocated_cuda_graph_managers()\n"
        "        from dflash_reflex.nemo_bridge import restore_trainer_rng\n"
        "        restore_trainer_rng(self, pretrained_path, init_optimizer)\n",
    )
    replace(
        worker,
        "                # Update parameters.\n                if not eval_mode:",
        "                # Update parameters.\n                if not eval_mode:\n"
        "                    from dflash_reflex.nemo_bridge import before_optimizer_step\n"
        "                    before_optimizer_step(self)",
    )
    replace(
        worker,
        "                    # Megatron-LM PR #4116 replaced",
        "                    from dflash_reflex.nemo_bridge import after_optimizer_step\n"
        "                    after_optimizer_step(self, update_successful)\n"
        "                    # Megatron-LM PR #4116 replaced",
    )
    replace(
        worker,
        "        original_save_path = self.mcore_state.cfg.checkpoint.save\n",
        "        from dflash_reflex.nemo_bridge import save_trainer\n"
        "        save_trainer(self, weights_path)\n"
        "        original_save_path = self.mcore_state.cfg.checkpoint.save\n",
    )
    # Reuse the mathematical reward and format coefficient from SpecNaacl.
    env = root / "environments/math_environment.py"
    replace(
        env,
        '                    if kwargs.get("math_verify_impl") == "dapo_math_verify":',
        '                    if kwargs.get("math_verify_impl") == "specnaacl":\n'
        "                        from dflash_reflex.rewards import accuracy_reward_func, format_reward_func\n"
        "                        ret_score = accuracy_reward_func([response], [ground_truth])[0] + 0.2 * format_reward_func([response])[0]\n"
        "                        extracted_answer = None\n"
        '                    elif kwargs.get("math_verify_impl") == "dapo_math_verify":',
    )
    loss = root / "algorithms/loss/loss_functions.py"
    replace(
        loss,
        "    disable_ppo_ratio: bool = False\n",
        "    disable_ppo_ratio: bool = False\n"
        "    # Preserve the k3 gradient used by SpecNaacl (no score factor).\n"
        "    specnaacl_kl_gradient: bool = False\n",
    )
    replace(
        loss,
        "        self.force_on_policy_ratio = cfg.force_on_policy_ratio",
        "        self.specnaacl_kl_gradient = cfg.specnaacl_kl_gradient\n"
        "        self.force_on_policy_ratio = cfg.force_on_policy_ratio",
    )
    replace(
        loss,
        "            # On-policy KL approximation\n            # KL samples come from the optimized policy, so the KL loss must include\n            # the score-function gradient through the sampling probability; see\n            # https://arxiv.org/abs/2506.09477v1. In the non-IS case,\n            # exp(x - x.detach()) has forward value 1 while preserving that gradient.\n            if self.use_on_policy_kl_approximation:\n                # See: docs/guides/grpo.md#on-policy-kl-approximation\n                kl_importance_weights = torch.exp(\n                    curr_logprobs_unfiltered - generation_logprobs\n                )\n            else:\n                kl_importance_weights = torch.exp(\n                    curr_logprobs_unfiltered - curr_logprobs_unfiltered.detach()\n                )\n            kl_importance_weights = torch.nan_to_num(\n                kl_importance_weights, nan=0.0, posinf=0.0, neginf=0.0\n            )\n\n            # Compute KL loss\n            kl = self.reference_policy_kl_penalty * calculate_kl(\n                logprobs=curr_logprobs_unfiltered,\n                logprobs_reference=reference_policy_logprobs,\n                kl_type=self.reference_policy_kl_type,\n                input_clamp_value=self.kl_input_clamp_value,\n                output_clamp_value=self.kl_output_clamp_value,\n                importance_sampling_weights=kl_importance_weights,\n            )\n\n",
        "            if self.specnaacl_kl_gradient:\n                from dflash_reflex.grpo_loss import specnaacl_kl_terms\n                kl = self.reference_policy_kl_penalty * specnaacl_kl_terms(\n                    curr_logprobs_unfiltered, reference_policy_logprobs)\n            else:\n                # On-policy KL approximation\n                # KL samples come from the optimized policy, so the KL loss must include\n                # the score-function gradient through the sampling probability; see\n                # https://arxiv.org/abs/2506.09477v1. In the non-IS case,\n                # exp(x - x.detach()) has forward value 1 while preserving that gradient.\n                if self.use_on_policy_kl_approximation:\n                    # See: docs/guides/grpo.md#on-policy-kl-approximation\n                    kl_importance_weights = torch.exp(\n                        curr_logprobs_unfiltered - generation_logprobs\n                    )\n                else:\n                    kl_importance_weights = torch.exp(\n                        curr_logprobs_unfiltered - curr_logprobs_unfiltered.detach()\n                    )\n                kl_importance_weights = torch.nan_to_num(\n                    kl_importance_weights, nan=0.0, posinf=0.0, neginf=0.0\n                )\n\n                # Compute KL loss\n                kl = self.reference_policy_kl_penalty * calculate_kl(\n                    logprobs=curr_logprobs_unfiltered,\n                    logprobs_reference=reference_policy_logprobs,\n                    kl_type=self.reference_policy_kl_type,\n                    input_clamp_value=self.kl_input_clamp_value,\n                    output_clamp_value=self.kl_output_clamp_value,\n                    importance_sampling_weights=kl_importance_weights,\n                )\n\n",
    )

    grpo = root / "algorithms/grpo.py"
    previous_record = "            record_iteration(total_steps + 1, timing_metrics, metrics)\n"
    revised_record = "            record_iteration(total_steps + 1, timing_metrics, dict(metrics, reward=float(rewards.float().mean())))\n"
    if previous_record in grpo.read_text():
        grpo.write_text(grpo.read_text().replace(previous_record, revised_record))
    replace(
        grpo,
        "        for batch in wrapped_dataloader:\n            refit_metrics:",
        "        from dflash_reflex.nemo_bridge import tracked_iterations\n"
        "        for batch in tracked_iterations(wrapped_dataloader, current_epoch, total_steps):\n"
        "            refit_metrics:",
    )
    replace(
        grpo,
        "            # Reset the batch and set dynamic_sampling_num_gen_batches to 0\n",
        "            from dflash_reflex.nemo_bridge import record_iteration\n"
        "            record_iteration(total_steps + 1, timing_metrics, dict(metrics, reward=float(rewards.float().mean())))\n"
        "            # Reset the batch and set dynamic_sampling_num_gen_batches to 0\n",
    )
    # The synchronous path commits its sidecar after final per-step timing.
    replace(
        grpo,
        "                        checkpointer.begin_finalization(\n"
        "                            checkpoint_path,\n"
        "                            wait_fn=policy.finalize_async_save,\n"
        "                        )",
        "                        from dflash_reflex.nemo_bridge import defer_driver_checkpoint\n"
        "                        defer_driver_checkpoint(checkpointer, checkpoint_path, policy.finalize_async_save)",
        count=2,
    )


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group()
    group.add_argument("--vllm-installed", action="store_true")
    group.add_argument("--nemo-only", action="store_true",
                       help="Patch editable NeMo source only; no installs or CUDA builds")
    a = p.parse_args()
    if a.nemo_only:
        patch_nemo(ROOT / "third_party/nemo-rl/nemo_rl")
    elif a.vllm_installed:
        if importlib.metadata.version("vllm") != "0.25.1":
            raise RuntimeError("only vLLM 0.25.1 has been source-audited")
        patch_vllm(Path(importlib.util.find_spec("vllm").origin).parent)
    else:
        patch_vllm(ROOT / "third_party/vllm/vllm")
        patch_nemo(ROOT / "third_party/nemo-rl/nemo_rl")


if __name__ == "__main__":
    main()
