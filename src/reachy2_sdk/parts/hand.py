"""Reachy Hand module.

Handles all specific method to a Hand.
"""

from collections import deque
from typing import Any, Deque, List, Optional

import grpc
import numpy as np
from google.protobuf.wrappers_pb2 import FloatValue
from reachy2_sdk_api.goto_pb2 import GoToId, GoToRequest, JointsGoal
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.hand_pb2 import Hand as Hand_proto
from reachy2_sdk_api.hand_pb2 import (
    HandPosition,
    HandPositionRequest,
    HandState,
    HandStatus,
    ParallelGripperPosition,
)
from reachy2_sdk_api.hand_pb2_grpc import HandJointGoal, HandServiceStub

from ..orbita.utils import to_internal_position, to_position
from ..utils.utils import get_grpc_interpolation_mode
from .goto_based_part import IGoToBasedPart
from .part import Part


class Hand(Part, IGoToBasedPart):
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
        goto_stub: GoToServiceStub,
    ) -> None:
        """Initialize the Hand component.

        Sets up the necessary attributes and configuration for the hand, including the gRPC
        stub and initial state.

        Args:
            hand_msg: The Hand_proto object containing the configuration details for the hand.
            initial_state: The initial state of the hand, represented as a HandState object.
            grpc_channel: The gRPC channel used to communicate with the hand's gRPC service.
            goto_stub: The gRPC stub for controlling goto movements.
        """
        super().__init__(hand_msg, grpc_channel, HandServiceStub(grpc_channel))
        IGoToBasedPart.__init__(self, self._part_id, goto_stub)
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

    def _check_goto_parameters(self, target: Any, duration: Optional[float] = 0, q0: Optional[List[float]] = None) -> None:
        """Check the validity of the parameters for the `goto` method.

        Args:
            duration: The time in seconds for the movement to be completed.
            target: The target position, either a list of joint positions or a 4x4 pose matrix.
            q0: An optional initial joint configuration for inverse kinematics. Defaults to None.

        Raises:
            TypeError: If the target is not a float or a int.
            ValueError: If the duration is set to 0.
        """
        if not (isinstance(target, float) or isinstance(target, int)):
            raise TypeError(f"Invalid target: must be either a float or a int, got {type(target)} instead.")

        elif duration == 0:
            raise ValueError("duration cannot be set to 0.")

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

    def send_goal_positions(self, check_positions: bool = True) -> None:
        """Send the goal position to the hand actuator.

        If any goal position has been specified to the gripper, sends them to the robot.
        If the hand is off, the command is not sent.

        Args :
            check_positions: A boolean indicating whether to check the positions after sending the command.
                Defaults to True.
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

    def goto_posture(
        self,
        common_posture: str = "default",
        duration: float = 2,
        wait: bool = False,
        wait_for_goto_end: bool = True,
        interpolation_mode: str = "minimum_jerk",
    ) -> GoToId:
        """Send all joints to standard positions with optional parameters for duration, waiting, and interpolation mode.

        Args:
            common_posture: The standard positions to which all joints will be sent.
                It can be 'default' or 'elbow_90'. Defaults to 'default'.
            duration: The time duration in seconds for the robot to move to the specified posture.
                Defaults to 2.
            wait: Determines whether the program should wait for the movement to finish before
                returning. If set to `True`, the program waits for the movement to complete before continuing
                execution. Defaults to `False`.
            wait_for_goto_end: Specifies whether commands will be sent to a part immediately or
                only after all previous commands in the queue have been executed. If set to `False`, the program
                will cancel all executing moves and queues. Defaults to `True`.
            interpolation_mode: The type of interpolation used when moving the arm's joints.
                Can be 'minimum_jerk' or 'linear'. Defaults to 'minimum_jerk'.

        Returns:
            A unique GoToId identifier for this specific movement.
        """
        if not wait_for_goto_end:
            self.cancel_all_goto()
        if self.is_on():
            return self.goto(0.0, duration, wait, interpolation_mode=interpolation_mode)
        else:
            self._logger.warning(f"{self._part_id.name} is off. No command sent.")
        return GoToId(id=-1)

    def goto(
        self,
        target: float | int,
        duration: float = 2,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
    ) -> GoToId:
        """Move the hand to a specified goal position.

        Args:
            target: The target position. It can either be a float or int.
            duration: The time in seconds for the movement to be completed. Defaults to 2.
            wait: If True, the function waits until the movement is completed before returning.
                    Defaults to False.
            interpolation_mode: The interpolation method to be used. It can be either "minimum_jerk"
                    or "linear". Defaults to "minimum_jerk".
            degrees: If True, the joint values in the `target` argument are treated as degrees.
                    Defaults to True.

        Returns:
            GoToId: The unique GoToId identifier for the movement command.
        """
        self._check_goto_parameters(target, duration)

        if self.is_off():
            self._logger.warning(f"{self._part_id.name} is off. Goto not sent.")
            return GoToId(id=-1)

        if degrees:
            target = np.deg2rad(target)

        request = GoToRequest(
            joints_goal=JointsGoal(
                hand_joint_goal=HandJointGoal(
                    goal_request=HandPositionRequest(
                        id=self._part_id,
                        position=HandPosition(
                            parallel_gripper=ParallelGripperPosition(position=FloatValue(value=self._outgoing_goal_positions))
                        ),
                    ),
                    duration=FloatValue(value=duration),
                )
            ),
            interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
        )

        response = self._goto_stub.GoToJoints(request)

        if response.id == -1:
            self._logger.error(f"Position {target} was not reachable. No command sent.")
        elif wait:
            self._wait_goto(response, duration)
        return response

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
        pass  # pragma: no cover
