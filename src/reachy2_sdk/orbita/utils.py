"""This module defines the utils class functions for Orbita."""

from typing import Any

import numpy as np
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue, UInt32Value
from reachy2_sdk_api.component_pb2 import PIDGains


def to_position(internal_pos: float) -> float:
    """Convert an internal angluar value in radians to a value in degrees."""
    return float(round(np.rad2deg(internal_pos), 2))


def to_internal_position(pos: float) -> Any:
    """Convert an angular value in degrees to a value in radians.

    It is necessary to convert the value from degrees to radians because the
    server expect values in radians.
    """
    try:
        return np.deg2rad(pos)
    except TypeError:
        raise TypeError(f"Excepted one of: int, float, got {type(pos).__name__}")


def unwrapped_pid_value(value: Any) -> Any:
    """Unwrap the internal pid value from gRPC protobuf to a Python value."""
    return (value.p.value, value.i.value, value.d.value)


def wrapped_proto_value(value: bool | float | int) -> Any:
    """Wrap the simple Python value to the corresponding gRPC one."""
    if isinstance(value, bool):
        return BoolValue(value=value)
    if isinstance(value, float):
        return FloatValue(value=value)
    if isinstance(value, UInt32Value):
        return UInt32Value(value=value)
    return value


def wrapped_pid_value(value: Any) -> Any:
    return PIDGains(
        p=FloatValue(value=value[0]),
        i=FloatValue(value=value[1]),
        d=FloatValue(value=value[2]),
    )
