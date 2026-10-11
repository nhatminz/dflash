"""Atomic state and RNG checkpoints, including uncommitted gradients."""

import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def rng_state():
    numpy = np.random.get_state()
    return dict(
        python=random.getstate(),
        # Tensors and primitives keep boundary packets weights_only-loadable.
        numpy=(numpy[0], torch.from_numpy(numpy[1].astype(np.int64)), *numpy[2:]),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    )


def restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    values = numpy[1]
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy().astype(np.uint32)
    np.random.set_state((numpy[0], values, *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def capture_training(model, optimizer, scheduler, **extra):
    return dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        rng=rng_state(),
        gradients={
            n: p.grad.detach().cpu() for n, p in model.named_parameters() if p.grad is not None
        },
        **extra,
    )


def restore_training(state, model, optimizer, scheduler):
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    for name, p in model.named_parameters():
        g = state["gradients"].get(name)
        p.grad = None if g is None else g.to(p.device)
    restore_rng(state["rng"])


def atomic_link(target, link):
    target, link = Path(target).resolve(), Path(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp = link.with_name(link.name + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(target)
    tmp.replace(link)
