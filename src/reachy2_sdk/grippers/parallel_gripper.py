"""Reachy ParallelGripper module.

Handles all specific methods to a ParallelGripper.
"""

from typing import Any, List, Optional

import grpc
import numpy as np
from google.protobuf.wrappers_pb2 import FloatValue
from reachy2_sdk_api.goto_pb2 import GoToId, GoToRequest, JointsGoal
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.hand_pb2 import Hand as Hand_proto
from reachy2_sdk_api.hand_pb2 import (
    HandJointGoal,
    HandPosition,
    HandPositionRequest,
    HandState,
    HandStatus,
    ParallelGripperPosition,
)
from reachy2_sdk_api.hand_pb2_grpc import HandServiceStub

from ..parts.hand import Hand
from ..utils.utils import get_grpc_interpolation_mode
from .gripper_joint import GripperJoint


class ParallelGripper(Hand):
    """Class for controlling the Reachy's parallel gripper.

    The `ParallelGripper` class provides methods to control the gripper of Reachy, including opening and closing
    the hand, setting the goal position, and checking the hand's state. It also manages the hand's
    compliance status (whether it is stiff or free). It implements all specific behaviors for the parallel gripper.

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
        """Initialize the ParallelGripper component.

        Sets up the necessary attributes and configuration for the hand, including the gRPC
        stub and initial state.

        Args:
            hand_msg: The Hand_proto object containing the configuration details for the hand.
            initial_state: The initial state of the hand, represented as a HandState object.
            grpc_channel: The gRPC channel used to communicate with the hand's gRPC service.
            goto_stub: The gRPC stub for controlling goto movements.
        """
        super().__init__(hand_msg, grpc_channel, goto_stub)
        self._stub = HandServiceStub(grpc_channel)

        self._joints = {"finger": GripperJoint(initial_state)}

    def __repr__(self) -> str:
        """Clean representation of a ParallelGripper."""
        s = "\n\t".join([str(joint) for joint in self._joints.values()])
        return f"""<ParallelGripper on={self.is_on()} joints=\n\t{
            s
        }\n>"""

    @property
    def opening(self) -> float:
        """Get the opening of the parallel gripper only joint as a percentage.

        Returns:
            The hand opening as a percentage (0-100), rounded to two decimal places.
        """
        return float(self._joints["finger"].opening)

    @property
    def present_position(self) -> float:
        """Get the current position of the parallel gripper only joint.

        Returns:
            The present position of the hand in degrees.
        """
        return float(self._joints["finger"].present_position)

    @property
    def goal_position(self) -> float:
        """Get the goal position of the parallel gripper only joint.

        Returns:
            The goal position of the hand in degrees.
        """
        return float(self._joints["finger"].goal_position)

    @goal_position.setter
    def goal_position(self, value: float | int) -> None:
        """Set the goal position for the parallel gripper only joint.

        Args:
            value: The goal position to set, specified as a float or int.

        Raises:
            TypeError: If the provided value is not a float or int.
        """
        self._joints["finger"].goal_position = value

    def _check_goto_parameters(self, target: Any, duration: Optional[float] = 0, q0: Optional[List[float]] = None) -> None:
        """Check the validity of the parameters for the `goto` method.

        Args:
            duration: The time in seconds for the movement to be completed.
            target: The target position, either a float or int.
            q0: An optional initial joint configuration for inverse kinematics (not used for the hand). Defaults to None.

        Raises:
            TypeError: If the target is not a float or a int.
            ValueError: If the duration is set to 0.
        """
        if not (isinstance(target, float) or isinstance(target, int)):
            raise TypeError(f"Invalid target: must be either a float or a int, got {type(target)} instead.")

        elif duration == 0:
            raise ValueError("duration cannot be set to 0.")

    def get_current_opening(self) -> float:
        """Get the current opening of the parallel gripper only joint.

        Returns:
            The current opening of the hand as a percentage (0-100).
        """
        return self.opening

    def set_opening(self, percentage: float) -> None:
        """Set the opening value for the parallel gripper only joint.

        Args:
            percentage: The desired opening percentage of the hand, ranging from 0 to 100.

        Raises:
            ValueError: If the percentage is not between 0 and 100.
            RuntimeError: If the gripper is off and the opening value cannot be set.
        """
        if not 0.0 <= percentage <= 100.0:
            raise ValueError(f"Percentage should be between 0 and 100, not {percentage}")
        if self.is_off():
            raise RuntimeError("Gripper is off. Opening value not sent.")

        self._stub.SetHandPosition(
            HandPositionRequest(
                id=self._part_id,
                position=HandPosition(
                    parallel_gripper=ParallelGripperPosition(opening_percentage=FloatValue(value=percentage / 100.0))
                ),
            )
        )
        self._joints["finger"]._is_moving = True

    def _get_goal_positions_message(self) -> Optional[HandPositionRequest]:
        """Get the HandPositionRequest message to send the goal positions to the actuator."""
        if self._joints["finger"]._outgoing_goal_positions is not None:
            if self.is_off():
                self._logger.warning(f"{self._part_id.name} is off. Command not sent.")
                return None
            command = HandPositionRequest(
                id=self._part_id,
                position=HandPosition(
                    parallel_gripper=ParallelGripperPosition(
                        position=FloatValue(value=self._joints["finger"]._outgoing_goal_positions)
                    )
                ),
            )
            self._joints["finger"]._is_moving = True
            return command
        return None

    def goto_posture(
        self,
        common_posture: str = "default",
        duration: float = 2,
        wait: bool = False,
        wait_for_goto_end: bool = True,
        interpolation_mode: str = "minimum_jerk",
    ) -> GoToId:
        """Send the gripper to default open posture with optional parameters for duration, waiting, and interpolation mode.

        Args:
            common_posture: The standard posture. It can be 'default' or 'elbow_90'. Defaults to 'default'.
                Modifying the posture has no effect on the hand.
            duration: The time duration in seconds for the robot to move to the specified posture.
                Defaults to 2.
            wait: Determines whether the program should wait for the movement to finish before
                returning. If set to `True`, the program waits for the movement to complete before continuing
                execution. Defaults to `False`.
            wait_for_goto_end: Specifies whether commands will be sent to a part immediately or
                only after all previous commands in the queue have been executed. If set to `False`, the program
                will cancel all executing moves and queues. Defaults to `True`.
            interpolation_mode: The type of interpolation used when moving the gripper.
                Can be 'minimum_jerk' or 'linear'. Defaults to 'minimum_jerk'.

        Returns:
            A unique GoToId identifier for this specific movement.
        """
        if not wait_for_goto_end:
            self.cancel_all_goto()
        if self.is_on():
            return self.goto(100.0, duration, wait, percentage=True, interpolation_mode=interpolation_mode)
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
        percentage: float = False,
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
            percentage: If True, the target value is treated as a percentage of opening. Defaults to False.

        Returns:
            GoToId: The unique GoToId identifier for the movement command.
        """
        self._check_goto_parameters(target, duration)

        if self.is_off():
            self._logger.warning(f"{self._part_id.name} is off. Goto not sent.")
            return GoToId(id=-1)

        if degrees and not percentage:
            target = np.deg2rad(target)

        if percentage:
            parallel_gripper_target = ParallelGripperPosition(opening_percentage=FloatValue(value=target / 100.0))
        else:
            parallel_gripper_target = ParallelGripperPosition(position=FloatValue(value=target))

        request = GoToRequest(
            joints_goal=JointsGoal(
                hand_joint_goal=HandJointGoal(
                    goal_request=HandPositionRequest(
                        id=self._part_id,
                        position=HandPosition(parallel_gripper=parallel_gripper_target),
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
        self._joints["finger"]._is_moving = True
        if interpolation_mode == "minimum_jerk":
            self._nb_steps_to_ignore = 10
        return response

    def _update_with(self, new_state: HandState) -> None:
        """Update the hand with a newly received (partial) state from the gRPC server.

        This method updates the present position, goal position, opening, and compliance status.
        It also checks if the hand is still moving based on the new state.

        Args:
            new_state: A HandState object representing the new state of the hand.
        """
        self._joints["finger"]._update_with(new_state)

    def _update_audit_status(self, new_status: HandStatus) -> None:
        """Update the audit status with the new status received from the gRPC server.

        Args:
            new_status: A HandStatus object representing the new status of the hand.
        """
        pass  # pragma: no cover

    @property
    def status(self) -> Optional[str]:
        """Get the current audit status of the actuator.

        Returns:
            The audit status as a string, representing the latest error or status
            message, or `None` if there is no error.
        """
        pass
