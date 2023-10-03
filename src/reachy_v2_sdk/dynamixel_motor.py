from grpc import Channel

from typing import Any, Dict

from .register import Register
from .orbita_utils import PID

# from reachy_sdk_api_v2.component_pb2 import ComponentId
from reachy_sdk_api_v2.dynamixel_motor_pb2_grpc import DynamixelMotorServiceStub


class DynamixelMotor:
    compliant = Register(readonly=False, label="compliant")
    present_position = Register(readonly=True, label="present_position")
    present_speed = Register(readonly=True, label="present_speed")
    present_load = Register(readonly=True, label="present_load")
    temperature = Register(readonly=True, label="temperature")
    goal_position = Register(readonly=False, label="goal_position")
    speed_limit = Register(readonly=False, label="speed_limit")
    torque_limit = Register(readonly=False, label="torque_limit")

    def __init__(self, name: str, grpc_channel: Channel):
        self.name = name
        self._stub = DynamixelMotorServiceStub(grpc_channel)

        self.compliant = False
        self.pid = PID(p=0.0, i=0.0, d=0.0)

        self._state: Dict[str, Any] = {}

        for field in dir(self):
            value = getattr(self, field)
            if isinstance(value, Register):
                value.label = field

    def __getitem__(self, field: str) -> Any:
        return self._state[field]

    def __setitem__(self, field: str, value: float) -> None:
        self._state[field] = value
