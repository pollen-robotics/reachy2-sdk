"""Reachy Antenna module.

Handles all specific methods to Antennas.
"""

from typing import Any, List, Optional

import numpy as np
from google.protobuf.wrappers_pb2 import FloatValue
from grpc import Channel
from reachy2_sdk_api.component_pb2 import ComponentId
from reachy2_sdk_api.dynamixel_motor_pb2 import DynamixelMotor as DynamixelMotor_proto
from reachy2_sdk_api.dynamixel_motor_pb2 import DynamixelMotorState
from reachy2_sdk_api.goto_pb2 import GoToId, GoToRequest, JointsGoal
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.head_pb2 import AntennaJointGoal

from ..dynamixel.dynamixel_motor import DynamixelMotor
from ..parts.part import Part
from ..utils.utils import get_grpc_interpolation_mode
from .goto_based_component import IGoToBasedComponent


class Antenna(DynamixelMotor, IGoToBasedComponent):
    """The Antenna class represents any antenna of the robot's head."""

    def __init__(
        self,
        uid: int,
        name: str,
        initial_state: DynamixelMotorState,
        grpc_channel: Channel,
        goto_stub: GoToServiceStub,
        part: Part,
    ):
        """Initialize the Antenna with its initial state and configuration.

        Args:
            uid: The unique identifier of the component.
            name: The name of the joint.
            initial_state: A dictionary containing the initial state of the joint, with
                each entry representing a specific parameter of the joint (e.g., present position).
            grpc_channel: The gRPC channel used to communicate with the DynamixelMotor service.
            goto_stub: The gRPC stub for controlling goto movements.
            part: The part to which this joint belongs.
        """
        super().__init__(uid, name, initial_state, grpc_channel, part)
        IGoToBasedComponent.__init__(self, ComponentId(id=uid, name=name), goto_stub)
        self._goto_stub = goto_stub

    def _check_goto_parameters(self, target: Any, duration: Optional[float], q0: Optional[List[float]] = None) -> None:
        """Check the validity of the parameters for the `goto` method.

        Args:
            duration: The time in seconds for the movement to be completed.
            target: The target position, either a list of joint positions or a quaternion.

        Raises:
            TypeError: If the target is not a list or a quaternion.
            ValueError: If the target list has a length other than 3.
            ValueError: If the duration is set to 0.
        """
        if not (isinstance(target, float) or isinstance(target, int)):
            raise TypeError(f"Antenna's goto target must be either a float or int, got {type(target)}.")

        elif duration == 0:
            raise ValueError("duration cannot be set to 0.")

    def goto_posture(
        self,
        common_posture: str = "default",
        duration: float = 2,
        wait: bool = False,
        wait_for_goto_end: bool = True,
        interpolation_mode: str = "minimum_jerk",
    ) -> GoToId:
        """Send the antenna to standard positions within the specified duration.

        The default posture sets the antenna is 0.0.

        Args:
            common_posture: The standard positions to which all joints will be sent.
                It can be 'default' or 'elbow_90'. Defaults to 'default'.
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
        if self.is_on():
            return self.goto(0, duration, wait, interpolation_mode)
        else:
            self._logger.warning(f"{self._name} is off. No command sent.")
        return GoToId(id=-1)

    def goto(
        self,
        target: float,
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
            TypeError : If the input type for `target` is invalid
            ValueError: If the `duration` is set to 0.

        Returns:
            GoToId: The unique identifier for the movement command.
        """
        if not self.is_on():
            self._logger.warning(f"head.{self._name} is off. No command sent.")
            return GoToId(id=-1)

        self._check_goto_parameters(target, duration)

        if degrees:
            target = np.deg2rad(target)

        request = GoToRequest(
            joints_goal=JointsGoal(
                antenna_joint_goal=AntennaJointGoal(
                    id=self._part._part_id,
                    antenna=DynamixelMotor_proto(
                        id=ComponentId(id=self._id, name=self._name),
                    ),
                    joint_goal=FloatValue(value=target),
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
