"""This module defines the utils class functions for Orbita."""

from typing import Any

import numpy as np


def to_position(internal_pos: float) -> Any:
    """Convert an internal angluar value in radians to a value in degrees."""
    return round(np.rad2deg(internal_pos), 2)


def to_internal_position(pos: float) -> Any:
    """Convert an angluar value in degrees to a value in radians.

    It is necessary to convert the value from degrees to radians because the
    server expect values in radians.
    """
    try:
        return np.deg2rad(pos)
    except TypeError:
        raise TypeError(f"Excepted one of: int, float, got {type(pos).__name__}")
