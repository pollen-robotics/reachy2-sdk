"""This module describes Orbita2d and Orbita3d motors."""
from typing import Any, Dict, List, Tuple

import numpy as np
from google.protobuf.wrappers_pb2 import FloatValue
from reachy2_sdk_api.component_pb2 import PIDGains

from .utils import unwrapped_pid_value


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

        self._update_with(initial_state)

        self._register_needing_sync: List[str] = []

    @property
    def speed_limit(self) -> float:
        return float(np.round(self._speed_limit, 3))

    @property
    def temperature(self) -> float:
        return float(self._temperature)

    @property
    def torque_limit(self) -> float:
        return float(np.round(self._torque_limit, 3))

    @property
    def compliant(self) -> float:
        return float(self._compliant)

    @property
    def pid(self) -> PIDGains:
        return self._pid

    def _update_with(self, new_state: Dict[str, FloatValue]) -> None:
        self._temperature = new_state["temperature"].value
        self._speed_limit = new_state["speed_limit"].value * 100
        self._torque_limit = new_state["torque_limit"].value * 100
        self._compliant = new_state["compliant"].value

        self._pid = unwrapped_pid_value(new_state["pid"])
