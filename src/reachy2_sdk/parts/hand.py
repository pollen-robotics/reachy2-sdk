"""Reachy Hand module.

Handles all specific method to a Hand.
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
    """Class for controlling the Reachy's hand.

    The `Hand` class provides methods to control the gripper of Reachy, including opening and closing
    the hand, setting the goal position, and checking the hand's state. It also manages the hand's
    compliance status (whether it is stiff or free).

    Attributes:
        opening: The opening of the hand as a percentage (0-100), rounded to two decimal places.
        present_position: The current position of the hand in degrees.
        goal_position: The target goal position of the hand in degrees.
    """

    def __init__(
        self,
        hand_msg: Hand_proto,
        initial_state: HandState,
        grpc_channel: grpc.Channel,
    ) -> None:
        """Initialize the Hand component.

        Sets up the necessary attributes and configuration for the hand, including the gRPC
        stub and initial state.

        Args:
            hand_msg: The Hand_proto object containing the configuration details for the hand.
            initial_state: The initial state of the hand, represented as a HandState object.
            grpc_channel: The gRPC channel used to communicate with the hand's gRPC service.
        """
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
        """Set up the hand with the given initial state.

        This method initializes the hand's present position, goal position, opening, and compliance status.

        Args:
            hand_msg: A Hand_proto object representing the hand's configuration.
            initial_state: A HandState object representing the initial state of the hand.
        """
        self._present_position: float = initial_state.present_position.parallel_gripper.position.value
        self._goal_position: float = initial_state.goal_position.parallel_gripper.position.value
        self._opening: float = initial_state.opening.value
        self._compliant: bool = initial_state.compliant.value
        self._last_present_positions.append(self._present_position)

        self._outgoing_goal_positions: Optional[float] = None

    def _set_speed_limits(self, value: int) -> None:
        """Set the speed limits for the hand.

        Args:
            value: The speed limit value to be set, as a percentage (0-100) of the maximum allowed speed,
                represented as an integer.
        """
        return super()._set_speed_limits(value)

    @property
    def opening(self) -> float:
        """Get the opening of the hand as a percentage.

        Returns:
            The hand opening as a percentage (0-100), rounded to two decimal places.
        """
        return round(self._opening * 100, 2)

    @property
    def present_position(self) -> float:
        """Get the current position of the hand.

        Returns:
            The present position of the hand in degrees.
        """
        return to_position(self._present_position)

    @property
    def goal_position(self) -> float:
        """Get the goal position of the hand.

        Returns:
            The goal position of the hand in degrees.
        """
        return to_position(self._goal_position)

    @goal_position.setter
    def goal_position(self, value: float | int) -> None:
        """Set the goal position for the hand.

        Args:
            value: The goal position to set, specified as a float or int.

        Raises:
            TypeError: If the provided value is not a float or int.
        """
        if isinstance(value, float) | isinstance(value, int):
            self._outgoing_goal_positions = to_internal_position(value)
        else:
            raise TypeError("goal_position must be a float or int")

    def is_on(self) -> bool:
        """Check if the hand is stiff.

        Returns:
            `True` if the hand is on (not compliant), `False` otherwise.
        """
        return not self._compliant

    def is_off(self) -> bool:
        """Check if the hand is compliant.

        Returns:
            `True` if the hand is off (compliant), `False` otherwise.
        """
        return self._compliant

    def is_moving(self) -> bool:
        """Check if the hand is currently moving.

        Returns:
            `True` if the gripper is moving, `False` otherwise.
        """
        return self._is_moving

    def _check_hand_movement(self, present_position: float) -> None:
        """Check if the hand is still moving based on the present position.

        This method updates the movement status by comparing the current position to the last few positions.
        If the position has not changed significantly, the hand is considered to have stopped moving.

        Args:
            present_position: The current position of the hand.
        """
        if (
            len(self._last_present_positions) >= self._last_present_positions_queue_size
            and np.isclose(present_position, self._last_present_positions[-1], np.deg2rad(0.1))
            and np.isclose(present_position, self._last_present_positions[-2], np.deg2rad(0.1))
        ):
            self._is_moving = False
            self._last_present_positions.clear()
        self._last_present_positions.append(present_position)

    def get_current_opening(self) -> float:
        """Get the current opening of the hand.

        Returns:
            The current opening of the hand as a percentage (0-100).
        """
        return self.opening

    def open(self) -> None:
        """Open the hand.

        Raises:
            RuntimeError: If the gripper is off and the open request cannot be sent.
        """
        if self._compliant:
            raise RuntimeError("Gripper is off. Open request not sent.")
        self._hand_stub.OpenHand(self._part_id)
        self._is_moving = True

    def close(self) -> None:
        """Close the hand.

        Raises:
            RuntimeError: If the gripper is off and the close request cannot be sent.
        """
        if self._compliant:
            raise RuntimeError("Gripper is off. Close request not sent.")
        self._hand_stub.CloseHand(self._part_id)
        self._is_moving = True

    def set_opening(self, percentage: float) -> None:
        """Set the opening value for the hand.

        Args:
            percentage: The desired opening percentage of the hand, ranging from 0 to 100.

        Raises:
            ValueError: If the percentage is not between 0 and 100.
            RuntimeError: If the gripper is off and the opening value cannot be set.
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
        """Send the goal position to the hand actuator.

        If any goal position has been specified to the gripper, sends them to the robot.
        If the hand is off, the command is not sent.
        """
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
        """Update the hand with a newly received (partial) state from the gRPC server.

        This method updates the present position, goal position, opening, and compliance status.
        It also checks if the hand is still moving based on the new state.

        Args:
            new_state: A HandState object representing the new state of the hand.
        """
        self._present_position = new_state.present_position.parallel_gripper.position.value
        self._goal_position = new_state.goal_position.parallel_gripper.position.value
        self._opening = new_state.opening.value
        self._compliant = new_state.compliant.value
        if self._is_moving:
            self._check_hand_movement(present_position=self._present_position)

    def _update_audit_status(self, new_status: HandStatus) -> None:
        """Update the audit status with the new status received from the gRPC server.

        Args:
            new_status: A HandStatus object representing the new status of the hand.
        """
        pass
