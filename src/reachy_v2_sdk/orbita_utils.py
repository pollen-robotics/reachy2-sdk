from typing import Any, Dict, List, Tuple

from google.protobuf.wrappers_pb2 import FloatValue

from .register import Register

from reachy_sdk_api_v2.component_pb2 import PIDGains
from google.protobuf.wrappers_pb2 import BoolValue

import numpy as np


def _to_position(internal_pos: float) -> Any:
    return round(np.rad2deg(internal_pos), 2)


def _to_internal_position(pos: float) -> Any:
    try:
        return np.deg2rad(pos)
    except TypeError:
        raise TypeError(f"Excepted one of: int, float, got {type(pos).__name__}")


class OrbitaJoint2D:
    present_position = Register(
        readonly=True, type=FloatValue, label="present_position", conversion=(_to_internal_position, _to_position)
    )
    goal_position = Register(
        readonly=False, type=FloatValue, label="goal_position", conversion=(_to_internal_position, _to_position)
    )

    def __init__(self, initial_state: Dict[str, float], axis_type: str, actuator: Any) -> None:
        self._actuator = actuator
        self.axis_type = axis_type
        self._state = initial_state

        self._register_needing_sync: List[str] = []

    def __repr__(self) -> str:
        return f'<OrbitaJoint2D axis_type="{self.axis_type}" present_position={self.present_position} goal_position={self.goal_position} >'  # noqa: E501


class OrbitaJoint3D:
    present_position = Register(
        readonly=True, type=float, label="present_position", conversion=(_to_internal_position, _to_position)
    )
    goal_position = Register(
        readonly=False, type=float, label="goal_position", conversion=(_to_internal_position, _to_position)
    )

    def __init__(self, initial_state: Dict[str, float], axis_type: str, actuator: Any) -> None:
        self._actuator = actuator
        self.axis_type = axis_type
        self._state = initial_state

        self._register_needing_sync: List[str] = []

    def __repr__(self) -> str:
        return f'<OrbitaJoint3D axis_type="{self.axis_type}" present_position={self.present_position} goal_position={self.goal_position} >'  # noqa: E501


class OrbitaMotor:
    temperature = Register(readonly=True, type=FloatValue, label="temperature")
    speed_limit = Register(
        readonly=False, type=FloatValue, label="speed_limit", conversion=(_to_internal_position, _to_position)
    )
    torque_limit = Register(readonly=False, type=FloatValue, label="torque_limit")
    compliant = Register(readonly=True, type=BoolValue, label="compliant")

    pid = Register(readonly=False, type=PIDGains, label="pid")

    def __init__(self, initial_state: Dict[str, Any], actuator: Any) -> None:
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
    present_speed = Register(
        readonly=True, type=FloatValue, label="present_speed", conversion=(_to_internal_position, _to_position)
    )
    present_load = Register(readonly=True, type=FloatValue, label="present_load")

    def __init__(self, initial_state: Dict[str, float]) -> None:
        self._state = initial_state

        for field in dir(self):
            value = getattr(self, field)
            if isinstance(value, Register):
                value.label = field
