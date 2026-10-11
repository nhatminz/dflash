"""Launch NVIDIA's actual DFlash co-training GRPO implementation."""

import argparse
import json
import os
from pathlib import Path

from .model_contract import validate_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    a = p.parse_args()
    # Shape/tokenizer/path checks precede importing Ray, Megatron, or vLLM.
    validate_checkpoint(
        os.environ["MODEL"], os.environ["DRAFT_CHECKPOINT"], int(os.environ["DFLASH_BLOCK_SIZE"])
    )
    from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup, shutdown_environments
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.utils import setup_response_data

    # Upstream's vLLM and MCore extras conflict. Use separately locked actor
    # environments, prepared once by scripts/setup_environment.sh.
    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
    from omegaconf import OmegaConf

    from .nemo_bridge import configure_driver, finalize_driver

    ACTOR_ENVIRONMENT_REGISTRY[
        "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
    ] = os.environ["DFLASH_VLLM_PYTHON"]
    ACTOR_ENVIRONMENT_REGISTRY[
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
    ] = os.environ["DFLASH_MCORE_PYTHON"]
    register_omegaconf_resolvers()
    master = MasterConfig(**OmegaConf.to_container(load_config(a.config), resolve=True))
    root = Path(os.environ["DFLASH_RUN_DIR"])
    (root / "run_config.json").write_text(
        json.dumps(master.model_dump(), indent=2, default=str) + "\n"
    )
    init_ray()
    tokenizer = get_tokenizer(master.policy["tokenizer"])
    master.policy["generation"] = configure_generation_config(
        master.policy["generation"], tokenizer, has_refit_draft_weights=True, trains_mtp=False
    )
    dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(
        tokenizer, master.data, master.env
    )
    (
        policy,
        generation,
        gym,
        cluster,
        loader,
        val_loader,
        loss_fn,
        logger,
        checkpointer,
        state,
        config,
        teacher_groups,
        aliases,
    ) = setup(master, tokenizer, dataset, val_dataset)
    configure_driver(checkpointer.get_latest_checkpoint_path())
    try:
        grpo_train(
            policy,
            generation,
            loader,
            val_loader,
            tokenizer,
            loss_fn,
            task_to_env,
            val_task_to_env,
            logger,
            checkpointer,
            state,
            config,
        )
    finally:
        checkpointer.shutdown()
        shutdown_environments(task_to_env, val_task_to_env)
        generation.shutdown()
        finalize_driver()


if __name__ == "__main__":
    main()
