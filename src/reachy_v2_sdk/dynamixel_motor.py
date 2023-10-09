from grpc import Channel

from typing import Any, Dict

from .register import Register

from google.protobuf.wrappers_pb2 import FloatValue

from reachy_sdk_api_v2.dynamixel_motor_pb2_grpc import DynamixelMotorServiceStub
from reachy_sdk_api_v2.dynamixel_motor_pb2 import DynamixelMotorState


class DynamixelMotor:
    compliant = Register(readonly=False, label="compliant")
    present_position = Register(readonly=True, label="present_position")
    present_speed = Register(readonly=True, label="present_speed")
    present_load = Register(readonly=True, label="present_load")
    temperature = Register(readonly=True, label="temperature")
    goal_position = Register(readonly=False, label="goal_position")
    speed_limit = Register(readonly=False, label="speed_limit")
    torque_limit = Register(readonly=False, label="torque_limit")

    def __init__(self, uid: int, name: str, initial_state: DynamixelMotorState, grpc_channel: Channel):
        self.id = uid
        self.name = name
        self._stub = DynamixelMotorServiceStub(grpc_channel)

        self._state: Dict[str, Any] = {}

        self._update_with(initial_state)

    def __getitem__(self, field: str) -> Any:
        return self._state[field]

    def __setitem__(self, field: str, value: float) -> None:
        self._state[field] = value

    def _update_with(self, new_state: DynamixelMotorState) -> None:
        for field, value in new_state.ListFields():
            self._state[field.name] = value

    def set_position(self, goal_position: float, duration: float) -> None:
        self._stub.SetPosition(id=self.id, goal_position=goal_position, duration=FloatValue(value=duration))
