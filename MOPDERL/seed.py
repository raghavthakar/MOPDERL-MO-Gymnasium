"""Deterministic seeding utilities.

This module centralizes all seeding and deterministic settings.

Notes on determinism:
- True bitwise determinism can be difficult on GPU depending on the ops used.
- We enable PyTorch deterministic algorithms when possible and disable CuDNN
  benchmarking.
- We also seed Python, NumPy, Torch (CPU+CUDA), and Gymnasium spaces.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class SeedInfo:
    seed: int
    pythonhashseed: str
    cublas_workspace_config: Optional[str]


def seed_everything(
    seed: int,
    *,
    deterministic_torch: bool = True,
    set_env_vars: bool = True,
) -> SeedInfo:
    """Seed Python/NumPy (and later Torch) for maximum determinism.

    IMPORTANT: For best results, call this *before importing torch* in the main
    entrypoint so that env vars like CUBLAS_WORKSPACE_CONFIG are honored.
    """

    if set_env_vars:
        # Makes hashing deterministic (affects dict/set iteration order)
        os.environ["PYTHONHASHSEED"] = str(seed)

        # Required for deterministic CuBLAS on many CUDA setups.
        # Must be set before torch is imported.
        # Valid values are typically ":16:8" or ":4096:8".
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    return SeedInfo(
        seed=seed,
        pythonhashseed=os.environ.get("PYTHONHASHSEED", ""),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )


def seed_torch(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed torch and set deterministic flags."""
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        # CuDNN
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Global deterministic algorithms
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older torch versions may not support it
            pass


def seed_env(env: Any, seed: int) -> None:
    """Seed Gymnasium/mo-gymnasium environment and spaces (best-effort)."""
    try:
        env.reset(seed=seed)
    except TypeError:
        # Older APIs might not accept seed kwarg
        env.reset()

    # Space seeding helps reproducibility for any sampling operations.
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    try:
        env.observation_space.seed(seed)
    except Exception:
        pass
