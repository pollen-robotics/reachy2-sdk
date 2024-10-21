"""Reachy Head module.

Handles all specific methods to a Head.
"""

from typing import Any, List, overload

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
from ..utils.utils import get_grpc_interpolation_mode, quaternion_from_euler
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

    def get_current_orientation(self) -> pyQuat:
        """Get the current orientation of the head.

        Returns:
            The orientation of the head as a quaternion (w, x, y, z).
        """
        quat = self._stub.GetOrientation(self._part_id).q
        return pyQuat(w=quat.w, x=quat.x, y=quat.y, z=quat.z)

    def get_current_positions(self, degrees: bool = True) -> List[float]:
        """Return the current joint positions of the neck.

        Returns:
            A list of the current neck joint positions in the order [roll, pitch, yaw].
        """
        roll = self.neck._joints["roll"].present_position
        pitch = self.neck._joints["pitch"].present_position
        yaw = self.neck._joints["yaw"].present_position
        if degrees:
            return [roll, pitch, yaw]
        return [np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)]

    @overload
    def goto(
        self,
        target: List[float],
        duration: float = 2.0,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
    ) -> GoToId:
        ...

    @overload
    def goto(
        self,
        target: pyQuat,
        duration: float = 2.0,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
    ) -> GoToId:
        ...

    def goto(
        self,
        target: Any,
        duration: float = 2.0,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
    ) -> GoToId:
        """Send the neck to a specified orientation.

        This method moves the neck either to a given roll-pitch-yaw (RPY) position or to a quaternion orientation.

        Args:
            target (Any): The desired orientation for the neck. Can either be:
                - A list of three floats [roll, pitch, yaw] representing the RPY orientation (in degrees if `degrees=True`).
                - A pyQuat object representing a quaternion.
            duration (float, optional): Time in seconds for the movement. Defaults to 2.0.
            wait (bool, optional): Whether to wait for the movement to complete before returning. Defaults to False.
            interpolation_mode (str, optional): The type of interpolation to be used for the movement.
                                                Can be "minimum_jerk" or other modes. Defaults to "minimum_jerk".
            degrees (bool, optional): Specifies if the RPY values in `target` are in degrees. Defaults to True.

        Raises:
            ValueError: If the `duration` is set to 0, or if the input type for `target` is invalid.

        Returns:
            GoToId: The unique identifier for the movement command.
        """
        if duration == 0:
            raise ValueError("duration cannot be set to 0.")

        if not self.neck.is_on():
            self._logger.warning("head.neck is off. No command sent.")
            return GoToId(id=-1)

        if isinstance(target, list) and len(target) == 3:
            if degrees:
                target = np.deg2rad(target).tolist()
            joints_goal = NeckOrientation(
                rotation=Rotation3d(
                    rpy=ExtEulerAngles(
                        roll=FloatValue(value=target[0]),
                        pitch=FloatValue(value=target[1]),
                        yaw=FloatValue(value=target[2]),
                    )
                )
            )
        elif isinstance(target, pyQuat):
            joints_goal = NeckOrientation(rotation=Rotation3d(q=Quaternion(w=target.w, x=target.x, y=target.y, z=target.z)))
        else:
            raise ValueError("Invalid input type for orientation. Must be either a list of 3 floats or a pyQuat.")

        request = GoToRequest(
            joints_goal=JointsGoal(
                neck_joint_goal=NeckJointGoal(
                    id=self._part_id,
                    joints_goal=joints_goal,
                    duration=FloatValue(value=duration),
                )
            ),
            interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
        )

        response = self._goto_stub.GoToJoints(request)

        if response.id == -1:
            if isinstance(target, list):
                self._logger.error(f"Position {target} was not reachable. No command sent.")
            elif isinstance(target, pyQuat):
                self._logger.error(f"Orientation {target} was not reachable. No command sent.")
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

    def look_at(
        self, x: float, y: float, z: float, duration: float = 2.0, wait: bool = False, interpolation_mode: str = "minimum_jerk"
    ) -> GoToId:
        """Compute and send a neck position to look at a specified point in Reachy's Cartesian space (torso frame).

        The (x, y, z) coordinates are expressed in meters, where x is forward, y is left, and z is upward.

        Args:
            x: The x-coordinate of the target point.
            y: The y-coordinate of the target point.
            z: The z-coordinate of the target point.
            duration: The time in seconds for the head to look at the point. Defaults to 2.0.
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

    def rotate_by(
        self,
        roll: float = 0,
        pitch: float = 0,
        yaw: float = 0,
        duration: float = 2,
        wait: bool = False,
        degrees: bool = True,
        frame: str = "robot",
        interpolation_mode: str = "minimum_jerk",
    ) -> GoToId:
        """Rotate the neck by the specified angles.

        Args:
            roll: The angle in degrees to rotate around the x-axis (roll). Defaults to 0.
            pitch: The angle in degrees to rotate around the y-axis (pitch). Defaults to 0.
            yaw: The angle in degrees to rotate around the z-axis (yaw). Defaults to 0.
            duration: The time in seconds for the neck to reach the target posture. Defaults to 2.
            wait: Whether to wait for the movement to complete before returning. Defaults to False.
            degrees: Whether the angles are provided in degrees. If True, the angles will be converted to radians.
                Defaults to True.
            frame: The frame of reference for the rotation. Can be either "robot" or "head". Defaults to "robot".
            interpolation_mode: The interpolation mode for the movement, either "minimum_jerk" or "linear".
                Defaults to "minimum_jerk".
        Raises:
            ValueError: If the frame is not "robot" or "head".
            ValueError: If the duration is set to 0.
            ValueError: If the interpolation mode is not "minimum_jerk" or "linear".
        """
        if duration == 0:
            raise ValueError("duration cannot be set to 0.")
        if interpolation_mode not in ["minimum_jerk", "linear"]:
            raise ValueError(f"Unknown interpolation mode {interpolation_mode}! Should be 'minimum_jerk' or 'linear'")
        if frame not in ["robot", "head"]:
            raise ValueError(f"Unknown frame {frame}! Should be 'robot' or 'head'")

        if not degrees:
            roll, pitch, yaw = np.rad2deg([roll, pitch, yaw])

        current_quaternion = self.get_current_orientation()
        additional_quaternion = quaternion_from_euler(roll, pitch, yaw, degrees=True)
        if frame == "head":
            target = current_quaternion * additional_quaternion
        elif frame == "robot":
            target = additional_quaternion * current_quaternion

        joints_goal = NeckOrientation(rotation=Rotation3d(q=Quaternion(w=target.w, x=target.x, y=target.y, z=target.z)))

        request = GoToRequest(
            joints_goal=JointsGoal(
                neck_joint_goal=NeckJointGoal(
                    id=self._part_id,
                    joints_goal=joints_goal,
                    duration=FloatValue(value=duration),
                )
            ),
            interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
        )

        response = self._goto_stub.GoToJoints(request)

        if response.id == -1:
            self._logger.error(f"Orientation {target} was not reachable. No command sent.")
        elif wait:
            self._wait_goto(response)
        return response

    def goto_posture(
        self,
        duration: float = 2,
        common_posture: str = "default",
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
            return self.goto([0, -10, 0], duration, wait, interpolation_mode)
        else:
            self._logger.warning("Head is off. No command sent.")
        return GoToId(id=-1)

    def send_goal_positions(self) -> None:
        """Send goal positions to the head's joints.

        If goal positions have been specified for any joint of the head, sends them to the robot.
        """
        if self.is_off():
            self._logger.warning(f"{self._part_id.name} is off. Command not sent.")
            return
        for actuator in self._actuators.values():
            actuator.send_goal_positions()

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
