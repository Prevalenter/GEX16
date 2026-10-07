from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def smooth_gx16_command(
    desired_rad: Sequence[float],
    previous_rad: Sequence[float],
    lower_rad: Sequence[float],
    upper_rad: Sequence[float],
    alpha: float,
) -> np.ndarray:
    """Validate, clamp, and EMA-smooth one GX16 command in URDF joint order."""
    desired = np.asarray(desired_rad, dtype=np.float64)
    previous = np.asarray(previous_rad, dtype=np.float64)
    lower = np.asarray(lower_rad, dtype=np.float64)
    upper = np.asarray(upper_rad, dtype=np.float64)
    for name, value in (
        ("desired_rad", desired),
        ("previous_rad", previous),
        ("lower_rad", lower),
        ("upper_rad", upper),
    ):
        if value.shape != (16,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain 16 finite values")
    if np.any(lower > upper):
        raise ValueError("GX16 lower joint limits exceed upper limits")
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    limited_target = np.clip(desired, lower, upper)
    limited_previous = np.clip(previous, lower, upper)
    command = alpha * limited_target + (1.0 - alpha) * limited_previous
    return np.clip(command, lower, upper)
