"""Reachy Head module.

Handles all specific methods to a Head.
"""

from typing import List

import grpc
import numpy as np
from google.protobuf.wrappers_pb2 import FloatValue
from pyquaternion import Quaternion as pyQuat
from reachy2_sdk_api.goto_pb2 import (
    CartesianGoal,
    CustomJointGoal,
    GoToId,
    GoToRequest,
    JointsGoal,
)
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.head_pb2 import CustomNeckJoints
from reachy2_sdk_api.head_pb2 import Head as Head_proto
from reachy2_sdk_api.head_pb2 import (
    HeadState,
    HeadStatus,
    NeckCartesianGoal,
    NeckJointGoal,
    NeckJoints,
    NeckOrientation,
)
from reachy2_sdk_api.head_pb2_grpc import HeadServiceStub
from reachy2_sdk_api.kinematics_pb2 import ExtEulerAngles, Point, Quaternion, Rotation3d

from ..orbita.orbita3d import Orbita3d
from ..utils.utils import get_grpc_interpolation_mode
from .goto_based_part import IGoToBasedPart
from .joints_based_part import JointsBasedPart


class Head(JointsBasedPart, IGoToBasedPart):
    """Head class for controlling the head of Reachy.

    The `Head` class manages the neck actuator and provides utilities for controlling the orientation
    of the head, such as moving to a specific posture or looking at a Cartesian point in Reachy's
    coordinate system.

    Attributes:
        neck: An instance of `Orbita3d` representing the neck actuator of the head.
    """

    def __init__(
        self,
        head_msg: Head_proto,
        initial_state: HeadState,
        grpc_channel: grpc.Channel,
        goto_stub: GoToServiceStub,
    ) -> None:
        """Initialize the Head component with its actuators.

        Sets up the necessary attributes and configuration for the head, including the gRPC
        stubs and initial state.

        Args:
            head_msg: The Head_proto object containing the configuration details for the head.
            initial_state: The initial state of the head, represented as a HeadState object.
            grpc_channel: The gRPC channel used to communicate with the head's gRPC service.
            goto_stub: The GoToServiceStub used to handle goto-based movements for the head.
        """
        JointsBasedPart.__init__(self, head_msg, grpc_channel, HeadServiceStub(grpc_channel))
        IGoToBasedPart.__init__(self, self, goto_stub)

        self._setup_head(head_msg, initial_state)
        self._actuators = {
            "neck": self.neck,
        }

    def _setup_head(self, head: Head_proto, initial_state: HeadState) -> None:
        """Set up the head with its actuators.

        This method initializes the neck and antenna actuators for the head and sets their initial state.

        Args:
            head: A Head_proto object containing the configuration details for the head.
            initial_state: A HeadState object representing the initial state of the head's actuators.
        """
        description = head.description
        self._neck = Orbita3d(
            uid=description.neck.id.id,
            name=description.neck.id.name,
            initial_state=initial_state.neck_state,
            grpc_channel=self._grpc_channel,
            part=self,
            joints_position_order=[NeckJoints.ROLL, NeckJoints.PITCH, NeckJoints.YAW],
        )

    def __repr__(self) -> str:
        """Clean representation of an Head."""
        s = "\n\t".join([act_name + ": " + str(actuator) for act_name, actuator in self._actuators.items()])
        return f"""<Head on={self.is_on()} actuators=\n\t{
            s
        }\n>"""

    @property
    def neck(self) -> Orbita3d:
        """Get the neck actuator of the head."""
        return self._neck

    def get_orientation(self) -> pyQuat:
        """Get the current orientation of the head.

        Returns:
            The orientation of the head as a quaternion (w, x, y, z).
        """
        quat = self._stub.GetOrientation(self._part_id).q
        return pyQuat(w=quat.w, x=quat.x, y=quat.y, z=quat.z)

    def get_joints_positions(self) -> List[float]:
        """Return the current joint positions of the neck.

        Returns:
            A list of the current neck joint positions in the order [roll, pitch, yaw].
        """
        roll = self.neck._joints["roll"].present_position
        pitch = self.neck._joints["pitch"].present_position
        yaw = self.neck._joints["yaw"].present_position
        return [roll, pitch, yaw]

    def look_at(
        self, x: float, y: float, z: float, duration: float = 2.0, wait: bool = False, interpolation_mode: str = "minimum_jerk"
    ) -> GoToId:
        """Compute and send a neck position to look at a specified point in Reachy's Cartesian space (torso frame).

        The (x, y, z) coordinates are expressed in meters, where x is forward, y is left, and z is upward.

        Args:
            x: The x-coordinate of the target point.
            y: The y-coordinate of the target point.
            z: The z-coordinate of the target point.
            duration: The time in seconds for the neck to reach the target point. Defaults to 2.0.
            wait: Whether to wait for the movement to complete before returning. Defaults to False.
            interpolation_mode: The interpolation mode for the movement, either "minimum_jerk" or "linear".
                Defaults to "minimum_jerk".

        Returns:
            The unique GoToId associated with the movement command.

        Raises:
            ValueError: If the duration is set to 0.
        """
        if duration == 0:
            raise ValueError("duration cannot be set to 0.")
        if not self.neck.is_on():
            self._logger.warning("head.neck is off. No command sent.")
            return GoToId(id=-1)

        request = GoToRequest(
            cartesian_goal=CartesianGoal(
                neck_cartesian_goal=NeckCartesianGoal(
                    id=self._part_id,
                    point=Point(x=x, y=y, z=z),
                    duration=FloatValue(value=duration),
                )
            ),
            interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
        )
        response = self._goto_stub.GoToCartesian(request)
        if response.id == -1:
            self._logger.error(f"Position {x}, {y}, {z} was not reachable. No command sent.")
        elif wait:
            self._wait_goto(response)
        return response

    def goto_joints(
        self,
        positions: List[float],
        duration: float = 2.0,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
    ) -> GoToId:
        """Send the neck to a specified roll, pitch, and yaw position.

        The rotation is applied in the order: roll, pitch, yaw.

        Args:
            positions: A list of three float values representing the target roll, pitch, and yaw angles.
            duration: The time in seconds for the neck to reach the target position. Defaults to 2.0.
            wait: Whether to wait for the movement to complete before returning. Defaults to False.
            interpolation_mode: The interpolation mode for the movement, either "minimum_jerk" or "linear".
                Defaults to "minimum_jerk".
            degrees: Whether the provided positions are in degrees. If True, positions will be converted to radians.
                Defaults to True.

        Returns:
            The unique GoToId associated with the movement command.

        Raises:
            ValueError: If the duration is set to 0.
        """
        if duration == 0:
            raise ValueError("duration cannot be set to 0.")
        if not self.neck.is_on():
            self._logger.warning("head.neck is off. No command sent.")
            return GoToId(id=-1)

        if degrees:
            positions = np.deg2rad(positions).tolist()
        request = GoToRequest(
            joints_goal=JointsGoal(
                neck_joint_goal=NeckJointGoal(
                    id=self._part_id,
                    joints_goal=NeckOrientation(
                        rotation=Rotation3d(
                            rpy=ExtEulerAngles(
                                roll=FloatValue(value=positions[0]),
                                pitch=FloatValue(value=positions[1]),
                                yaw=FloatValue(value=positions[2]),
                            )
                        )
                    ),
                    duration=FloatValue(value=duration),
                )
            ),
            interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
        )
        response = self._goto_stub.GoToJoints(request)
        if response.id == -1:
            self._logger.error(f"Position {positions} was not reachable. No command sent.")
        elif wait:
            self._wait_goto(response)
        return response

    def _goto_single_joint(
        self,
        neck_joint: int,
        goal_position: float,
        duration: float = 2,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
    ) -> GoToId:
        """Move a single joint of the neck to a specified goal position.

        Args:
            neck_joint: The index of the neck joint to move (0 for roll, 1 for pitch, 2 for yaw).
            goal_position: The target position for the joint.
            duration: The time in seconds for the joint to reach the goal position. Defaults to 2.
            wait: Whether to wait for the movement to complete before returning. Defaults to False.
            interpolation_mode: The interpolation mode for the movement, either "minimum_jerk" or "linear".
                Defaults to "minimum_jerk".
            degrees: Whether the goal position is provided in degrees. If True, the position will be converted to radians.
                Defaults to True.

        Returns:
            The GoToId associated with the movement command.

        Raises:
            ValueError: If the duration is set to 0.
        """
        if duration == 0:
            raise ValueError("duration cannot be set to 0.")
        if degrees:
            goal_position = np.deg2rad(goal_position)
        request = GoToRequest(
            joints_goal=JointsGoal(
                custom_joint_goal=CustomJointGoal(
                    id=self._part_id,
                    neck_joints=CustomNeckJoints(joints=[neck_joint]),
                    joints_goals=[FloatValue(value=goal_position)],
                    duration=FloatValue(value=duration),
                )
            ),
            interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
        )
        response = self._goto_stub.GoToJoints(request)
        if response.id == -1:
            self._logger.error(f"Position {goal_position} was not reachable. No command sent.")
        elif wait:
            self._wait_goto(response)
        return response

    def goto_quat(
        self, q: pyQuat, duration: float = 2.0, wait: bool = False, interpolation_mode: str = "minimum_jerk"
    ) -> GoToId:
        """Send the neck to the orientation specified by a quaternion.

        Args:
            q: The target orientation as a quaternion (w, x, y, z).
            duration: The time in seconds for the neck to reach the target orientation. Defaults to 2.0.
            wait: Whether to wait for the movement to complete before returning. Defaults to False.
            interpolation_mode: The interpolation mode for the movement, either "minimum_jerk" or "linear".
                Defaults to "minimum_jerk".

        Returns:
            The GoToId associated with the movement command.

        Raises:
            ValueError: If the duration is set to 0.
        """
        if duration == 0:
            raise ValueError("duration cannot be set to 0.")
        if not self.neck.is_on():
            self._logger.warning("head.neck is off. No command sent.")
            return GoToId(id=-1)

        request = GoToRequest(
            joints_goal=JointsGoal(
                neck_joint_goal=NeckJointGoal(
                    id=self._part_id,
                    joints_goal=NeckOrientation(rotation=Rotation3d(q=Quaternion(w=q.w, x=q.x, y=q.y, z=q.z))),
                    duration=FloatValue(value=duration),
                )
            ),
            interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
        )
        response = self._goto_stub.GoToJoints(request)
        if response.id == -1:
            self._logger.error(f"Orientation {q} was not reachable. No command sent.")
        elif wait:
            self._wait_goto(response)
        return response

    def send_goal_positions(self) -> None:
        """Send goal positions to the head's joints.

        If goal positions have been specified for any joint of the head, sends them to the robot.
        """
        if self.is_off():
            self._logger.warning(f"{self._part_id.name} is off. Command not sent.")
            return
        for actuator in self._actuators.values():
            actuator.send_goal_positions()

    def goto_posture(
        self,
        duration: float = 2,
        wait: bool = False,
        wait_for_goto_end: bool = True,
        interpolation_mode: str = "minimum_jerk",
    ) -> GoToId:
        """Send all neck joints to standard positions within the specified duration.

        The default posture sets the neck joints to [0, -10, 0] (roll, pitch, yaw).

        Args:
            duration: The time in seconds for the neck to reach the target posture. Defaults to 2.
            wait: Whether to wait for the movement to complete before returning. Defaults to False.
            wait_for_goto_end: Whether to wait for all previous goto commands to finish before executing
                the current command. If False, it cancels all ongoing commands. Defaults to True.
            interpolation_mode: The interpolation mode for the movement, either "minimum_jerk" or "linear".
                Defaults to "minimum_jerk".

        Returns:
            The unique GoToId associated with the movement command.

        Raises:
            ValueError: If the neck is off and the command cannot be sent.
        """
        if not wait_for_goto_end:
            self.cancel_all_goto()
        if self.neck.is_on():
            return self.goto_joints([0, -10, 0], duration, wait, interpolation_mode)
        else:
            self._logger.warning("head.neck is off. No command sent.")
        return GoToId(id=-1)

    def _update_with(self, new_state: HeadState) -> None:
        """Update the head with a newly received (partial) state from the gRPC server.

        Args:
            new_state: A HeadState object representing the new state of the head's actuators.
        """
        self.neck._update_with(new_state.neck_state)

    def _update_audit_status(self, new_status: HeadStatus) -> None:
        """Update the audit status of the neck with the new status from the gRPC server.

        Args:
            new_status: A HeadStatus object representing the new status of the neck.
        """
        self.neck._update_audit_status(new_status.neck_status)
