import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dflash_reflex.config import ReflexConfig
from dflash_reflex.projector_training import ProjectorTrainer


def worker(rank, init_file, result_dir):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    trainer = ProjectorTrainer(torch.ones(4, 2), ReflexConfig(rank=2))
    trainer.accumulate(
        dict(
            group_id=f"owner{rank}",
            version=0,
            grad_sum=torch.full((4, 2), float(rank + 1)),
            grad_weight=torch.tensor(float(2 * rank + 1)),
        )
    )
    trainer.step()
    torch.save(
        dict(A=trainer.A.detach(), norm=trainer.last_grad_norm), f"{result_dir}/rank{rank}.pt"
    )
    dist.destroy_process_group()


def test_global_gradient_numerator_and_weight(tmp_path):
    mp.spawn(worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    results = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in range(2)]
    torch.testing.assert_close(results[0]["A"], results[1]["A"], rtol=0, atol=0)
    expected = torch.full((4, 2), 0.75).norm().item()  # (1+2)/(1+3)
    assert abs(results[0]["norm"] - expected) < 1e-7
