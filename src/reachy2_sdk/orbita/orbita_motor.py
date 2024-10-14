"""Reachy OrbitaMotor module.

Handles all specific methods to OrbitaMotor.
"""

from typing import Any, Dict

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
        """Initialize the motor with its initial state.

        Args:
            initial_state: A dictionary containing the initial state values for the motor. The keys should include
                "temperature", "speed_limit", "torque_limit", "compliant", and "pid", with corresponding
                FloatValue objects as values.
            actuator: The actuator to which the motor belongs.
        """
        self._actuator = actuator
        self._update_with(initial_state)

    @property
    def speed_limit(self) -> float:
        """Get the speed limit of the motor, as a percentage of the max allowed speed, rounded to three decimal places."""
        return float(np.round(self._speed_limit, 3))

    @property
    def temperature(self) -> float:
        """Get the current temperature of the motor in Celsius degrees."""
        return float(self._temperature)

    @property
    def torque_limit(self) -> float:
        """Get the torque limit of the axis, as a percentage of the max allowed speed, rounded to three decimal places."""
        return float(np.round(self._torque_limit, 3))

    @property
    def compliant(self) -> float:
        """Get the compliance status of the motor."""
        return float(self._compliant)

    @property
    def pid(self) -> PIDGains:
        """Get the PID gains of the motor."""
        return self._pid

    def _update_with(self, new_state: Dict[str, FloatValue]) -> None:
        """Update the state of the motor with new values from the provided state dictionary.

        Args:
            new_state: A dictionary containing the new state values for the axis. The keys should include
                "temperature", "speed_limit", "torque_limit", "compliant", and "pid", with corresponding
                FloatValue objects as values.
        """
        self._temperature = new_state["temperature"].value
        self._speed_limit = new_state["speed_limit"].value * 100  # received value in [0, 1]
        self._torque_limit = new_state["torque_limit"].value * 100  # received value in [0, 1]
        self._compliant = new_state["compliant"].value
        self._pid = unwrapped_pid_value(new_state["pid"])
