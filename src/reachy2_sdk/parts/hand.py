"""Reachy Hand module.


Handles all specific method to a Hand:
    - turn_on / turn_off
    - open / close
"""
from collections import deque
from typing import Deque, Optional

import grpc
import numpy as np
from google.protobuf.wrappers_pb2 import FloatValue
from reachy2_sdk_api.hand_pb2 import Hand as Hand_proto
from reachy2_sdk_api.hand_pb2 import (
    HandPosition,
    HandPositionRequest,
    HandState,
    HandStatus,
    ParallelGripperPosition,
)
from reachy2_sdk_api.hand_pb2_grpc import HandServiceStub

from ..orbita.utils import to_internal_position, to_position
from .part import Part


class Hand(Part):
    def __init__(
        self,
        hand_msg: Hand_proto,
        initial_state: HandState,
        grpc_channel: grpc.Channel,
    ) -> None:
        """Set up the hand."""
        super().__init__(hand_msg, grpc_channel, HandServiceStub(grpc_channel))
        self._hand_stub = HandServiceStub(grpc_channel)

        self._is_moving = False
        self._last_present_positions_queue_size = 10
        self._last_present_positions: Deque[float] = deque(maxlen=self._last_present_positions_queue_size)

        self._setup_hand(hand_msg, initial_state)

    def __repr__(self) -> str:
        """Clean representation of a Hand."""
        return f"<Hand on={self.is_on()} opening={self.opening} >"

    def _setup_hand(self, hand_msg: Hand_proto, initial_state: HandState) -> None:
        """Set up the hand.

        It will create the hand and set its initial state.
        """
        self._present_position: float = initial_state.present_position.parallel_gripper.position.value
        self._goal_position: float = initial_state.goal_position.parallel_gripper.position.value
        self._opening: float = initial_state.opening.value
        self._compliant: bool = initial_state.compliant.value
        self._last_present_positions.append(self._present_position)

        self._outgoing_goal_positions: Optional[float] = None

    def _set_speed_limits(self, value: int) -> None:
        return super()._set_speed_limits(value)

    @property
    def opening(self) -> float:
        """Return the opening of the hand in percentage."""
        return round(self._opening * 100, 2)

    @property
    def present_position(self) -> float:
        return to_position(self._present_position)

    @property
    def goal_position(self) -> float:
        return to_position(self._goal_position)

    @goal_position.setter
    def goal_position(self, value: float | int) -> None:
        if isinstance(value, float) | isinstance(value, int):
            self._outgoing_goal_positions = to_internal_position(value)
        else:
            raise TypeError("goal_position must be a float or int")

    def is_on(self) -> bool:
        """Get compliancy of the hand"""
        return not self._compliant

    def is_off(self) -> bool:
        """Get compliancy of the hand"""
        return self._compliant

    def is_moving(self) -> bool:
        """Get state of gripper movement"""
        return self._is_moving

    def _check_hand_movement(self, present_position: float) -> None:
        if (
            len(self._last_present_positions) >= self._last_present_positions_queue_size
            and np.isclose(present_position, self._last_present_positions[-1], np.deg2rad(0.1))
            and np.isclose(present_position, self._last_present_positions[-2], np.deg2rad(0.1))
        ):
            self._is_moving = False
            self._last_present_positions.clear()
        self._last_present_positions.append(present_position)

    def get_current_opening(self) -> float:
        return self.opening

    def open(self) -> None:
        """Open the hand."""
        if self._compliant:
            raise RuntimeError("Gripper is off. Open request not sent.")
        self._hand_stub.OpenHand(self._part_id)
        self._is_moving = True

    def close(self) -> None:
        """Close the hand."""
        if self._compliant:
            raise RuntimeError("Gripper is off. Close request not sent.")
        self._hand_stub.CloseHand(self._part_id)
        self._is_moving = True

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
                position=HandPosition(
                    parallel_gripper=ParallelGripperPosition(opening_percentage=FloatValue(value=percentage / 100.0))
                ),
            )
        )
        self._is_moving = True

    def send_goal_positions(self) -> None:
        if self.is_off():
            self._logger.warning(f"{self._part_id.name} is off. Command not sent.")
            return
        if self._outgoing_goal_positions is not None:
            self._hand_stub.SetHandPosition(
                HandPositionRequest(
                    id=self._part_id,
                    position=HandPosition(
                        parallel_gripper=ParallelGripperPosition(position=FloatValue(value=self._outgoing_goal_positions))
                    ),
                )
            )
            self._outgoing_goal_positions = None
            self._is_moving = True

    def _update_with(self, new_state: HandState) -> None:
        """Update the hand with a newly received (partial) state received from the gRPC server."""
        self._present_position = new_state.present_position.parallel_gripper.position.value
        self._goal_position = new_state.goal_position.parallel_gripper.position.value
        self._opening = new_state.opening.value
        self._compliant = new_state.compliant.value
        if self._is_moving:
            self._check_hand_movement(present_position=self._present_position)

    def _update_audit_status(self, new_status: HandStatus) -> None:
        pass
