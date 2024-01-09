"""This module defines the utils class to describe Orbita2d and Orbita3d joints, motors and axis."""
from typing import Any, Dict, List, Tuple

import numpy as np
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.component_pb2 import JointLimits, PIDGains

from .register import Register


def _to_position(internal_pos: float) -> Any:
    """Convert an internal angluar value in radians to a value in degrees."""
    return round(np.rad2deg(internal_pos), 2)


def _to_internal_position(pos: float) -> Any:
    """Convert an angluar value in degrees to a value in radians.

    It is necessary to convert the value from degrees to radians because the
    server expect values in radians.
    """
    try:
        return np.deg2rad(pos)
    except TypeError:
        raise TypeError(f"Excepted one of: int, float, got {type(pos).__name__}")


class OrbitaJoint2d:
    """The OrbitaJoint2d class represents any Orbita2d joint.

    The OrbitaJoint2d class is used to store the up-to-date state of the joint, especially:
        - its present_position (RO)
        - its goal_position (RW)
    """

    present_position = Register(
        readonly=True,
        type=FloatValue,
        label="present_position",
        conversion=(_to_internal_position, _to_position),
    )
    joint_limits = Register(
        readonly=True,
        type=JointLimits,
        label="joint_limits",
        conversion=(_to_internal_position, _to_position),
    )
    goal_position = Register(
        readonly=False,
        type=FloatValue,
        label="goal_position",
        conversion=(_to_internal_position, _to_position),
    )

    def __init__(self, initial_state: Dict[str, Any], axis_type: str, actuator: Any) -> None:
        self._actuator = actuator
        self.axis_type = axis_type
        self._state = initial_state

        self._register_needing_sync: List[str] = []

    def __repr__(self) -> str:
        return f'<OrbitaJoint2d axis_type="{self.axis_type}" present_position={self.present_position} goal_position={self.goal_position} >'  # noqa: E501


class OrbitaJoint3d:
    """The OrbitaJoint3d class represents any Orbita3d joint.

    The OrbitaJoint3d class is used to store the up-to-date state of the joint, especially:
        - its present position (RO)
        - its goal position (RW)
    """

    present_position = Register(
        readonly=True,
        type=float,
        label="present_position",
        conversion=(_to_internal_position, _to_position),
    )
    joint_limits = Register(
        readonly=True,
        type=JointLimits,
        label="joint_limits",
        conversion=(_to_internal_position, _to_position),
    )
    goal_position = Register(
        readonly=False,
        type=float,
        label="goal_position",
        conversion=(_to_internal_position, _to_position),
    )

    def __init__(self, initial_state: Dict[str, Any], axis_type: str, actuator: Any) -> None:
        """Initialize the joint with its initial state and its axis type (either roll, pitch or yaw)."""
        self._actuator = actuator
        self.axis_type = axis_type
        self._state = initial_state

        for field in dir(self):
            value = getattr(self, field)
            if isinstance(value, Register):
                value.label = field

        self._register_needing_sync: List[str] = []

    def __repr__(self) -> str:
        """Return a clean representation of an Orbita 3d joint."""
        return f'<OrbitaJoint3d axis_type="{self.axis_type}" present_position={self.present_position} goal_position={self.goal_position} >'  # noqa: E501


class OrbitaMotor:
    """The OrbitaMotor class represents any Orbita3d or Orbita2d motor.

    The OrbitaMotor class is used to store the up-to-date state of the motor, especially:
        - its temperature (RO)
        - its compliancy (RO)
        - its speed limit (RW)
        - its torque limit (RW)
        - its pid (RW)
    """

    temperature = Register(readonly=True, type=FloatValue, label="temperature")
    speed_limit = Register(
        readonly=False,
        type=FloatValue,
        label="speed_limit",
        conversion=(_to_internal_position, _to_position),
    )
    torque_limit = Register(readonly=False, type=FloatValue, label="torque_limit", lower_limit=0, upper_limit=100)
    compliant = Register(readonly=True, type=BoolValue, label="compliant")

    pid = Register(readonly=False, type=PIDGains, label="pid")

    def __init__(self, initial_state: Dict[str, Any], actuator: Any) -> None:
        """Initialize the motor with its initial state."""
        self._actuator = actuator

        self._state = initial_state

        self._tmp_fields: Dict[str, float | None] = {}
        self._tmp_pid: Tuple[float, float, float]

        for field in dir(self):
            value = getattr(self, field)
            if isinstance(value, Register):
                value.label = field
                if not value.readonly:
                    self._tmp_fields[field] = None

        self._register_needing_sync: List[str] = []


class OrbitaAxis:
    """The OrbitaAxis class represents any Orbita3d or Orbita2d axis.

    The OrbitaAxis class is used to store the up-to-date state of the axis, especially:
        - its present speed (RO)
        - its present load (RO)
    """

    present_speed = Register(
        readonly=True,
        type=FloatValue,
        label="present_speed",
        conversion=(_to_internal_position, _to_position),
    )
    present_load = Register(readonly=True, type=FloatValue, label="present_load")

    def __init__(self, initial_state: Dict[str, float]) -> None:
        """Initialize the axis with its initial state."""
        self._state = initial_state

        for field in dir(self):
            value = getattr(self, field)
            if isinstance(value, Register):
                value.label = field
