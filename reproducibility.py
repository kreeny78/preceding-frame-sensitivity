#!/usr/bin/env python3
"""Seeding, random-number state capture, and deterministic data workers.
"""

from __future__ import annotations

import os
import random
from typing import Dict, Optional

import numpy as np
import torch

DEFAULT_SEED = 42


def seed_everything(seed: int, deterministic_algorithms: bool = False) -> int:
    """Seed every generator this codebase draws from.

    Call before constructing datasets, transforms, samplers, DataLoaders,
    the model, or the optimizer. Returns the seed so callers can record it.

    `deterministic_algorithms` additionally requests deterministic CUDA
    kernels. It is off by default because several convolution backward
    kernels have no deterministic implementation and would raise; the
    smoke and contract jobs turn it on where the shapes permit it.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {seed!r}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def worker_init_fn(worker_id: int) -> None:
    """Seed a DataLoader worker from the base seed torch assigned it."""
    base = torch.initial_seed() % (2**31 - 1)
    seed = (base + worker_id) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)


def loader_generator(seed: int) -> torch.Generator:
    """A generator for DataLoader shuffling, seeded independently."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def capture_rng_state(sampler_epoch: Optional[int] = None) -> Dict[str, object]:
    """Snapshot every generator, for storage in a checkpoint."""
    state: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "sampler_epoch": sampler_epoch,
    }
    return state


def restore_rng_state(state: Dict[str, object]) -> Optional[int]:
    """Put every generator back. Returns the stored sampler epoch."""
    if not isinstance(state, dict):
        raise TypeError(
            "RNG state must be the mapping written by capture_rng_state; a "
            "resumed run cannot reproduce the batch order without it."
        )
    for key in ("python", "numpy", "torch_cpu"):
        if key not in state:
            raise KeyError(
                f"RNG state is missing {key!r}. Resume is refused rather "
                f"than continued from a different random stream."
            )
    random.setstate(_as_tuple(state["python"]))
    np.random.set_state(_as_numpy_state(state["numpy"]))
    torch.set_rng_state(_as_byte_tensor(state["torch_cpu"]))
    cuda_states = state.get("torch_cuda") or []
    if torch.cuda.is_available() and cuda_states:
        device_count = torch.cuda.device_count()
        if len(cuda_states) != device_count:
            raise RuntimeError(
                f"Checkpoint holds {len(cuda_states)} CUDA generator states "
                f"but this machine exposes {device_count} devices. Resume on "
                f"a matching device count."
            )
        torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in cuda_states])
    epoch = state.get("sampler_epoch")
    return None if epoch is None else int(epoch)


def _as_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(_as_tuple(v) if isinstance(v, list) else v for v in value)
    return value


def _as_numpy_state(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        head = value[0]
        keys = np.asarray(value[1], dtype=np.uint32)
        return (head, keys, int(value[2]), int(value[3]), float(value[4]))
    return value


def _as_byte_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.cpu().to(torch.uint8)
    return torch.tensor(list(value), dtype=torch.uint8)


def set_epoch_recursively(obj, epoch: int) -> int:
    """Propagate the epoch to every object exposing `set_epoch`.

    Datasets that redraw their sampling grid each epoch and samplers that
    reshuffle both need this. Returns how many objects were reached so a
    caller can assert the propagation happened rather than assume it.
    """
    seen: set = set()
    count = 0

    def walk(node) -> None:
        nonlocal count
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        setter = getattr(node, "set_epoch", None)
        if callable(setter):
            setter(epoch)
            count += 1
        for attribute in ("dataset", "datasets", "sampler", "batch_sampler"):
            child = getattr(node, attribute, None)
            if isinstance(child, (list, tuple)):
                for item in child:
                    walk(item)
            else:
                walk(child)

    walk(obj)
    return count


__all__ = [
    "DEFAULT_SEED",
    "seed_everything",
    "worker_init_fn",
    "loader_generator",
    "capture_rng_state",
    "restore_rng_state",
    "set_epoch_recursively",
]
