"""
This module defines the DynamixelMotor class.

This class is used to represent a single Dynamixel motor on the robot.
It is used to send commands to the motor and to read its state.
"""
import asyncio
from typing import Any, Dict, List

from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from grpc import Channel
from reachy2_sdk_api.component_pb2 import ComponentId
from reachy2_sdk_api.dynamixel_motor_pb2 import (
    DynamixelMotorCommand,
    DynamixelMotorState,
)
from reachy2_sdk_api.dynamixel_motor_pb2_grpc import DynamixelMotorServiceStub

from .register import Register


class DynamixelMotor:
    """DynamixelMotor class.

    This class is used to represent a single Dynamixel motor on the robot.
    It is used to send commands to the motor and to read its state.

    The DynamixelMotor class is used in parts using Dynamixel motors (hand and head).
    """

    compliant = Register(readonly=False, type=BoolValue, label="compliant")
    present_position = Register(readonly=True, type=FloatValue, label="present_position")
    present_speed = Register(readonly=True, type=FloatValue, label="present_speed")
    present_load = Register(readonly=True, type=FloatValue, label="present_load")
    temperature = Register(readonly=True, type=FloatValue, label="temperature")
    goal_position = Register(readonly=False, type=FloatValue, label="goal_position")
    speed_limit = Register(readonly=False, type=FloatValue, label="speed_limit")
    torque_limit = Register(readonly=False, type=FloatValue, label="torque_limit")

    def __init__(
        self,
        uid: int,
        name: str,
        initial_state: DynamixelMotorState,
        grpc_channel: Channel,
    ):
        self.id = uid
        self.name = name
        self._stub = DynamixelMotorServiceStub(grpc_channel)

        self._state: Dict[str, Any] = {}

        self._register_needing_sync: List[str] = []

        self._update_with(initial_state)

    def __repr__(self) -> str:
        return f"<DynamixelMotor compliant={self.compliant} present_position={self.present_position} goal_position={self.goal_position} >"  # noqa: E501

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in ["goal_position", "speed_limit", "torque_limit", "compliant"]:
            self._state[__name] = __value

            async def set_in_loop() -> None:
                self._register_needing_sync.append(__name)
                self._need_sync.set()

            fut = asyncio.run_coroutine_threadsafe(set_in_loop(), self._loop)
            fut.result()
        super().__setattr__(__name, __value)

    def __getitem__(self, field: str) -> Any:
        return self._state[field]

    # def __setitem__(self, field: str, value: float) -> None:
    #     self._state[field] = value

    def _update_with(self, new_state: DynamixelMotorState) -> None:
        for field, value in new_state.ListFields():
            self._state[field.name] = value

    def set_position(self, goal_position: float, duration: float) -> None:
        """Set the goal position of the motor. The movement will be done in the given duration (in seconds)."""
        self._stub.SetPosition(id=self.id, goal_position=goal_position, duration=FloatValue(value=duration))

    def _setup_sync_loop(self) -> None:
        """Set up the async synchronisation loop.

        The setup is done separately, as the async Event should be created in the same EventLoop than it will be used.

        The _need_sync Event is used to inform the robot that some data need to be pushed to the real robot.
        The _register_needing_sync stores a list of the register that need to be synced.
        """
        self._need_sync = asyncio.Event()
        self._loop = asyncio.get_running_loop()

    def _pop_command(self) -> DynamixelMotorCommand:
        """Create a gRPC command from the registers that need to be synced."""
        values = {
            "id": ComponentId(id=self.id),
        }

        set_reg = set(self._register_needing_sync)

        for reg in set_reg:
            if reg == "compliant":
                values["compliant"] = BoolValue(value=self._state["compliant"])
            else:
                values[reg] = FloatValue(value=self._state[reg])
        command = DynamixelMotorCommand(**values)

        self._register_needing_sync.clear()
        self._need_sync.clear()

        return command
