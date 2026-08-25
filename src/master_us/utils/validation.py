"""Determinism and cross-cutting assertions.

`set_determinism` is called at the top of every training entry point. Without
it, seed-to-seed comparison is meaningless — and seed dispersion is the
threshold this project uses to decide whether any result is real.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_determinism(seed: int) -> None:
    """Pin every RNG this project touches.

    Torch is imported lazily so that data-layer code and tests can call this
    without pulling in a multi-hundred-megabyte dependency.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def assert_aligned(
    left_index: object,
    right_index: object,
    left_name: str = "left",
    right_name: str = "right",
) -> None:
    """Hard-fail on index mismatch. Called at every join in the data layer.

    A silent misalignment produces plausible numbers, which is worse than a
    crash — the whole point of failing loudly.
    """
    import pandas as pd

    li = pd.Index(left_index)
    ri = pd.Index(right_index)
    if not li.equals(ri):
        only_left = li.difference(ri)
        only_right = ri.difference(li)
        raise ValueError(
            f"index mismatch between {left_name} ({len(li)}) and {right_name} ({len(ri)}): "
            f"{len(only_left)} only in {left_name}, {len(only_right)} only in {right_name}"
        )
