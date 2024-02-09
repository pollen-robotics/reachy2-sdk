"""Reachy Hand module.


Handles all specific method to a Hand:
    - turn_on / turn_off
    - open / close
"""
import grpc
import numpy as np
from reachy2_sdk_api.hand_pb2 import Hand as Hand_proto
from reachy2_sdk_api.hand_pb2 import (
    HandPosition,
    HandPositionRequest,
    HandState,
    ParallelGripperPosition,
)
from reachy2_sdk_api.hand_pb2_grpc import HandServiceStub
from reachy2_sdk_api.part_pb2 import PartId


class Hand:
    def __init__(self, hand_msg: Hand_proto, initial_state: HandState, grpc_channel: grpc.Channel) -> None:
        """Set up the hand."""
        self._hand_stub = HandServiceStub(grpc_channel)

        self._setup_hand(hand_msg, initial_state)

    def __repr__(self) -> str:
        """Clean representation of a Hand."""
        return f"Hand with part_id {self._part_id} and opening of {self.opening}%"

    def _setup_hand(self, hand_msg: Hand_proto, initial_state: HandState) -> None:
        """Set up the hand.

        It will create the hand and set its initial state.
        """
        self._part_id = PartId(id=hand_msg.part_id.id)
        self._present_position: float = round(np.rad2deg(initial_state.present_position.parallel_gripper.position), 1)
        self._goal_position: float = round(np.rad2deg(initial_state.goal_position.parallel_gripper.position), 1)
        self._opening: float = initial_state.opening.value
        self._compliant: bool = initial_state.compliant.value

    @property
    def opening(self) -> float:
        """Return the opening of the hand in percentage."""
        return round(self._opening * 100, 2)

    def set_opening(self, percentage: float) -> None:
        """Set an opening value for the hand

        Args:
            percentage (float): Percentage of the opening.
        """
        if not 0.0 <= percentage <= 100.0:
            raise ValueError(f"Percentage should be between 0 and 100, not {percentage}")
        if self._compliant:
            raise RuntimeError("Gripper is off. Opening value not sent.")

        self._hand_stub.SetHandPosition(
            HandPositionRequest(
                id=self._part_id,
                position=HandPosition(parallel_gripper=ParallelGripperPosition(position=percentage / 100.0)),
            )
        )

    def open(self) -> None:
        """Open the hand."""
        if self._compliant:
            raise RuntimeError("Gripper is off. Open request not sent.")
        self._hand_stub.OpenHand(self._part_id)

    def close(self) -> None:
        """Close the hand."""
        if self._compliant:
            raise RuntimeError("Gripper is off. Close request not sent.")
        self._hand_stub.CloseHand(self._part_id)

    def turn_on(self) -> None:
        """Turn all motors of the hand on.

        All hand's motors will then be stiff.
        """
        self._hand_stub.TurnOn(self._part_id)

    def turn_off(self) -> None:
        """Turn all motors of the hand off.

        All hand's motors will then be compliant.
        """
        self._hand_stub.TurnOff(self._part_id)

    def is_on(self) -> bool:
        """Get compliancy of the hand"""
        return not self._compliant

    def _update_with(self, new_state: HandState) -> None:
        """Update the hand with a newly received (partial) state received from the gRPC server."""
        self._present_position = new_state.present_position.parallel_gripper.position
        self._goal_position = new_state.goal_position.parallel_gripper.position
        self._opening = new_state.opening.value
        self._compliant = new_state.compliant.value
