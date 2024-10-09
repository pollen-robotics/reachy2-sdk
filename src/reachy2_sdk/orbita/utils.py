"""This module defines the utils class functions for Orbita."""

from typing import Any

import numpy as np
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue, UInt32Value
from reachy2_sdk_api.component_pb2 import PIDGains


def to_position(internal_pos: float) -> float:
    """Convert an internal angular value in radians to a value in degrees.

    Args:
        internal_pos: The internal angular value in radians.

    Returns:
        The corresponding angular value in degrees.
    """
    return float(np.rad2deg(internal_pos))


def to_internal_position(pos: float) -> Any:
    """Convert an angular value in degrees to a value in radians.

    The server expects values in radians, so conversion is necessary.

    Args:
        pos: The angular value in degrees.

    Returns:
        The corresponding value in radians.

    Raises:
        TypeError: If the provided value is not of type int or float.
    """
    try:
        return np.deg2rad(pos)
    except TypeError:
        raise TypeError(f"Excepted one of: int, float, got {type(pos).__name__}")


def unwrapped_pid_value(value: Any) -> Any:
    """Unwrap the internal PID value from a gRPC protobuf object to a Python value.

    Args:
        value: The gRPC protobuf object containing the PID values.

    Returns:
        A tuple representing the unwrapped PID gains (p, i, d).
    """
    return (value.p.value, value.i.value, value.d.value)


def wrapped_proto_value(value: bool | float | int) -> Any:
    """Wrap a simple Python value to the corresponding gRPC protobuf type.

    Args:
        value: The value to be wrapped, which can be a bool, float, or int.

    Returns:
        The corresponding gRPC protobuf object (BoolValue, FloatValue, or UInt32Value).

    Raises:
        TypeError: If the provided value is not a supported type.
    """
    if isinstance(value, bool):
        return BoolValue(value=value)
    if isinstance(value, float):
        return FloatValue(value=value)
    if isinstance(value, UInt32Value):
        return UInt32Value(value=value)
    return value


def wrapped_pid_value(value: Any) -> Any:
    """Wrap a simple Python value to the corresponding gRPC protobuf type.

    Args:
        value: The value to be wrapped, which can be a bool, float, or int.

    Returns:
        The corresponding gRPC protobuf object (BoolValue, FloatValue, or UInt32Value).

    Raises:
        TypeError: If the provided value is not a supported type.
    """
    return PIDGains(
        p=FloatValue(value=value[0]),
        i=FloatValue(value=value[1]),
        d=FloatValue(value=value[2]),
    )
