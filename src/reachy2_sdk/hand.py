import grpc
import numpy as np

from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.hand_pb2 import Hand as Hand_proto
from reachy2_sdk_api.hand_pb2 import HandPosition, HandPositionRequest, HandState, ParallelGripperPosition
from reachy2_sdk_api.part_pb2 import PartId
from reachy2_sdk_api.hand_pb2_grpc import HandServiceStub

from typing import Any


class Hand:
    def __init__(self, hand_msg: Hand_proto, initial_state: HandState, grpc_channel: grpc.Channel) -> None:
        """Set up the arm with its kinematics."""
        self._hand_stub = HandServiceStub(grpc_channel)
        self.type = "gripper"
        self.part_id = PartId(id=hand_msg.part_id.id)

    def __repr__(self) -> str:
        return ""

    def open(self, percentage: float = 100) -> None:
        if not 0.0 <= percentage <= 100.0:
            raise ValueError(f"Percentage should be between 0 and 100, not {percentage}")

        if percentage == 100.0:
            self._hand_stub.OpenHand(self.part_id)
        else:
            self._hand_stub.SetHandPosition(
                HandPositionRequest(
                    id=self.part_id,
                    position=HandPosition(parallel_gripper=ParallelGripperPosition(position=percentage/100.0))
                )
            )

    def close(self, percentage: float = 100) -> None:
        if not 0.0 <= percentage <= 100.0:
            raise ValueError(f"Percentage should be between 0 and 100, not {percentage}")

        if percentage == 100.0:
            self._hand_stub.CloseHand(self.part_id)
        else:
            self._hand_stub.SetHandPosition(
                HandPositionRequest(
                    id=self.part_id,
                    position=HandPosition(parallel_gripper=ParallelGripperPosition(position=(100-percentage)/100.0))
                )
            )

    def turn_on(self) -> None:
        self._hand_stub.TurnOn(self.part_id)

    def turn_off(self) -> None:
        self._hand_stub.TurnOff(self.part_id)

    def _update_with(self, new_state: HandState) -> None:
        for field, value in new_state.ListFields():
            if isinstance(value, FloatValue):
                setattr(self, field.name, value.value)
            if isinstance(value, BoolValue):
                setattr(self, field.name, value.value)
            if isinstance(value, HandPosition):
                setattr(self, field.name, value.parallel_gripper.position)
