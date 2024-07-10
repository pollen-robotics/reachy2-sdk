"""This module describes Orbita2d and Orbita3d motors."""
from typing import Any, Dict, List, Tuple

from google.protobuf.wrappers_pb2 import FloatValue
from reachy2_sdk_api.component_pb2 import PIDGains

from .utils import (
    to_position,
    unwrapped_pid_value,
    unwrapped_proto_value,
)


class OrbitaMotor:
    """The OrbitaMotor class represents any Orbita3d or Orbita2d motor.

    The OrbitaMotor class is used to store the up-to-date state of the motor, especially:
        - its temperature (RO)
        - its compliancy (RO)
        - its speed limit (RW)
        - its torque limit (RW)
        - its pid (RW)
    """

    def __init__(self, initial_state: Dict[str, Any], actuator: Any) -> None:
        """Initialize the motor with its initial state."""
        self._actuator = actuator

        self._state = initial_state

        self._tmp_fields: Dict[str, float | None] = {}
        self._tmp_state: Dict[str, float | None] = initial_state.copy()
        self._tmp_pid: Tuple[float, float, float]

        self._temperature = unwrapped_proto_value(initial_state["temperature"])
        self._speed_limit = unwrapped_proto_value(initial_state["speed_limit"])
        self._torque_limit = unwrapped_proto_value(initial_state["torque_limit"])
        self._compliant = unwrapped_proto_value(initial_state["compliant"])

        self._pid = unwrapped_pid_value(initial_state["pid"])

        self._register_needing_sync: List[str] = []

    @property
    def speed_limit(self) -> float:
        return to_position(self._speed_limit)

    @property
    def temperature(self) -> float:
        return float(self._temperature)

    @property
    def torque_limit(self) -> float:
        return float(self._torque_limit)

    @property
    def compliant(self) -> float:
        return float(self._compliant)

    @property
    def pid(self) -> PIDGains:
        return self._pid

    def _update_with(self, new_state: Dict[str, FloatValue]) -> None:
        self._temperature = unwrapped_proto_value(new_state["temperature"])
        self._speed_limit = unwrapped_proto_value(new_state["speed_limit"])
        self._torque_limit = unwrapped_proto_value(new_state["torque_limit"])
        self._compliant = unwrapped_proto_value(new_state["compliant"])

        self._pid = unwrapped_pid_value(new_state["pid"])
