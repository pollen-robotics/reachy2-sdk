"""This module describes Orbita2d and Orbita3d motors."""
from typing import Any, Dict, List, Tuple

from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.component_pb2 import PIDGains

from ..motors.register import Register
from .utils import to_internal_position, to_position


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
        conversion=(to_internal_position, to_position),
    )
    torque_limit = Register(readonly=False, type=FloatValue, label="torque_limit")
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
