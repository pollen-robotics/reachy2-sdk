"""Reachy Arm module.

Handles all specific methods to an Arm (left and/or right).
"""

import time
from typing import Any, Dict, List, Optional, overload

import grpc
import numpy as np
import numpy.typing as npt
from google.protobuf.wrappers_pb2 import FloatValue
from reachy2_sdk_api.arm_pb2 import Arm as Arm_proto
from reachy2_sdk_api.arm_pb2 import (  # ArmLimits,; ArmTemperatures,
    ArmCartesianGoal,
    ArmComponentsCommands,
    ArmEndEffector,
    ArmFKRequest,
    ArmIKRequest,
    ArmJointGoal,
    ArmJoints,
    ArmState,
    ArmStatus,
    CustomArmJoints,
)
from reachy2_sdk_api.arm_pb2_grpc import ArmServiceStub
from reachy2_sdk_api.goto_pb2 import (
    CartesianGoal,
    CustomJointGoal,
    EllipticalGoToParameters,
    GoToId,
    GoToRequest,
    JointsGoal,
)
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.hand_pb2 import Hand as HandState
from reachy2_sdk_api.hand_pb2 import Hand as Hand_proto
from reachy2_sdk_api.hand_pb2 import HandType
from reachy2_sdk_api.kinematics_pb2 import Matrix4x4

from ..grippers.parallel_gripper import ParallelGripper
from ..orbita.orbita2d import Orbita2d
from ..orbita.orbita3d import Orbita3d
from ..utils.utils import (
    arm_position_to_list,
    get_grpc_arc_direction,
    get_grpc_interpolation_mode,
    get_grpc_interpolation_space,
    list_to_arm_position,
    matrix_from_euler_angles,
    recompose_matrix,
    rotate_in_self,
    translate_in_self,
)
from .goto_based_part import IGoToBasedPart
from .hand import Hand
from .joints_based_part import JointsBasedPart


class Arm(JointsBasedPart, IGoToBasedPart):
    """Reachy Arm module.

    Handles specific functionalities for the arm (left and/or right), including:
    - Forward and inverse kinematics
    - Goto functions for movement
    - Turning the arm on and off
    - Cartesian interpolation for movements

    Attributes:
        shoulder (Orbita2d): The shoulder actuator of the arm.
        elbow (Orbita2d): The elbow actuator of the arm.
        wrist (Orbita3d): The wrist actuator of the arm.
        gripper (Optional[Hand]): The gripper of the arm, if initialized.
    """

    def __init__(
        self,
        arm_msg: Arm_proto,
        initial_state: ArmState,
        grpc_channel: grpc.Channel,
        goto_stub: GoToServiceStub,
    ) -> None:
        """Initialize an Arm instance.

        This constructor sets up the arm's gRPC communication and initializes its actuators
        (shoulder, elbow, and wrist). Optionally, a gripper can also be configured.

        Args:
            arm_msg: The protobuf message containing the arm's configuration details.
            initial_state: The initial state of the arm's actuators.
            grpc_channel: The gRPC channel used for communication with the arm's server.
            goto_stub: The gRPC stub for controlling goto movements.
        """
        JointsBasedPart.__init__(self, arm_msg, grpc_channel, ArmServiceStub(grpc_channel))
        IGoToBasedPart.__init__(self, self._part_id, goto_stub)

        self._setup_arm(arm_msg, initial_state)
        self._gripper: Optional[Hand] = None

        self._actuators: Dict[str, Orbita2d | Orbita3d | Hand] = {}
        self._actuators["shoulder"] = self.shoulder
        self._actuators["elbow"] = self.elbow
        self._actuators["wrist"] = self.wrist

    def _setup_arm(self, arm: Arm_proto, initial_state: ArmState) -> None:
        """Initialize the arm's actuators (shoulder, elbow, and wrist) based on the arm's description and initial state.

        Args:
            arm: The arm description used to set up the actuators, including the shoulder,
                elbow, and wrist. The method creates instances of `Orbita2d` for the shoulder and
                elbow, and an instance of `Orbita3d` for the wrist.
            initial_state: The initial state of the arm's actuators, containing the starting
                positions or states of the shoulder, elbow, and wrist. This information is used to
                initialize the corresponding actuators.
        """
        description = arm.description
        self._shoulder = Orbita2d(
            uid=description.shoulder.id.id,
            name=description.shoulder.id.name,
            axis1=description.shoulder.axis_1,
            axis2=description.shoulder.axis_2,
            initial_state=initial_state.shoulder_state,
            grpc_channel=self._grpc_channel,
            part=self,
            joints_position_order=[ArmJoints.SHOULDER_PITCH, ArmJoints.SHOULDER_ROLL],
        )
        self._elbow = Orbita2d(
            uid=description.elbow.id.id,
            name=description.elbow.id.name,
            axis1=description.elbow.axis_1,
            axis2=description.elbow.axis_2,
            initial_state=initial_state.elbow_state,
            grpc_channel=self._grpc_channel,
            part=self,
            joints_position_order=[ArmJoints.ELBOW_YAW, ArmJoints.ELBOW_PITCH],
        )
        self._wrist = Orbita3d(
            uid=description.wrist.id.id,
            name=description.wrist.id.name,
            initial_state=initial_state.wrist_state,
            grpc_channel=self._grpc_channel,
            part=self,
            joints_position_order=[ArmJoints.WRIST_ROLL, ArmJoints.WRIST_PITCH, ArmJoints.WRIST_YAW],
        )

    def _init_hand(self, hand: Hand_proto, hand_initial_state: HandState) -> None:
        if hand.type == HandType.PARALLEL_GRIPPER:
            self._gripper = ParallelGripper(hand, hand_initial_state, self._grpc_channel, self._goto_stub)
            self._actuators["gripper"] = self._gripper

    @property
    def shoulder(self) -> Orbita2d:
        """Get the shoulder actuator of the arm."""
        return self._shoulder

    @property
    def elbow(self) -> Orbita2d:
        """Get the elbow actuator of the arm."""
        return self._elbow

    @property
    def wrist(self) -> Orbita3d:
        """Get the wrist actuator of the arm."""
        return self._wrist

    @property
    def gripper(self) -> Optional[Hand]:
        """Get the gripper of the arm, or None if not set."""
        return self._gripper

    def __repr__(self) -> str:
        """Clean representation of an Arm."""
        s = "\n\t".join([act_name + ": " + str(actuator) for act_name, actuator in self._actuators.items()])
        return f"""<Arm on={self.is_on()} actuators=\n\t{
            s
        }\n>"""

    def turn_on(self) -> bool:
        """Turn on all motors of the part, making all arm motors stiff.

        If a gripper is present, it will also be turned on.

        Returns:
            'True' if all motors are on, 'False' otherwise.
        """
        if self._gripper:
            self._gripper._turn_on()
            if not self._gripper.is_on():
                return False
        super().turn_on()
        return self.is_on()

    def turn_off(self) -> bool:
        """Turn off all motors of the part, making all arm motors compliant.

        If a gripper is present, it will also be turned off.

        Returns:
            'True' if all motors are off, 'False' otherwise.
        """
        if self._gripper:
            self._gripper._turn_off()
            if not self._gripper.is_off():
                return False
        super().turn_off()
        return self.is_off()

    def turn_off_smoothly(self) -> bool:
        """Gradually reduce the torque limit of all motors over 3 seconds before turning them off.

        This function decreases the torque limit in steps until the motors are turned off.
        It then restores the torque limit to its original value.

        Returns:
            'True' if all motors are off, 'False' otherwise.
        """
        torque_limit_low = 35
        torque_limit_high = 100
        duration = 3

        self.set_torque_limits(torque_limit_low)
        self.goto_posture(duration=duration, wait_for_goto_end=False)

        countingTime = 0
        while countingTime < duration:
            time.sleep(1)
            torque_limit_low -= 10
            self.set_torque_limits(torque_limit_low)
            countingTime += 1

        super().turn_off()
        self.set_torque_limits(torque_limit_high)
        return self.is_off()

    def _turn_on(self) -> bool:
        """Turn on all motors of the part.

        This will make all arm motors stiff. If a gripper is present, it will also be turned on.

        Returns:
            'True' if all motors are on, 'False' otherwise.
        """
        if self._gripper:
            self._gripper._turn_on()
            if not self._gripper.is_on():
                return False
        super()._turn_on()
        return self.is_on()

    def _turn_off(self) -> bool:
        """Turn off all motors of the part.

        This will make all arm motors compliant. If a gripper is present, it will also be turned off.

        Returns:
            True' if all motors are off, 'False' otherwise.
        """
        if self._gripper:
            self._gripper._turn_off()
            if not self._gripper.is_off():
                return False
        super()._turn_off()
        return self.is_off()

    def is_on(self, check_gripper: bool = True) -> bool:
        """Check if all actuators of the arm are stiff.

        Returns:
            `True` if all actuators of the arm are stiff, `False` otherwise.
        """
        if not check_gripper:
            for actuator in [self._actuators[act] for act in self._actuators.keys() if act not in ["gripper"]]:
                if not actuator.is_on():
                    return False
            return True
        return super().is_on()

    def is_off(self, check_gripper: bool = True) -> bool:
        """Check if all actuators of the arm are compliant.

        Returns:
            `True` if all actuators of the arm are compliant, `False` otherwise.
        """
        if not check_gripper:
            for actuator in [self._actuators[act] for act in self._actuators.keys() if act not in ["gripper"]]:
                if not actuator.is_off():
                    return False
            return True
        return super().is_off()

    def get_current_positions(self, degrees: bool = True) -> List[float]:
        """Return the current joint positions of the arm, either in degrees or radians.

        Args:
            degrees: Specifies whether the joint positions should be returned in degrees.
                If set to `True`, the positions are returned in degrees; otherwise, they are returned in radians.
                Defaults to `True`.

        Returns:
            A list of float values representing the current joint positions of the arm in the
            following order: [shoulder_pitch, shoulder_roll, elbow_yaw, elbow_pitch, wrist_roll, wrist_pitch,
            wrist_yaw].
        """
        response = self._stub.GetJointPosition(self._part_id)
        positions: List[float] = arm_position_to_list(response, degrees)
        return positions

    def forward_kinematics(
        self, joints_positions: Optional[List[float]] = None, degrees: bool = True
    ) -> npt.NDArray[np.float64]:
        """Compute the forward kinematics of the arm and return a 4x4 pose matrix.

        The pose matrix is expressed in Reachy coordinate system.

        Args:
            joints_positions: A list of float values representing the positions of the joints
                in the arm. If not provided, the current robot joints positions are used. Defaults to None.
            degrees: Indicates whether the joint positions are in degrees or radians.
                If `True`, the positions are in degrees; if `False`, in radians. Defaults to True.

        Returns:
            A 4x4 pose matrix as a NumPy array, expressed in Reachy coordinate system.

        Raises:
            ValueError: If `joints_positions` is provided and its length is not 7.
            ValueError: If no solution is found for the given joint positions.
        """
        req_params = {
            "id": self._part_id,
        }
        if joints_positions is None:
            present_joints_positions = [
                joint.present_position for orbita in self._actuators.values() for joint in orbita._joints.values()
            ]
            req_params["position"] = list_to_arm_position(present_joints_positions, degrees)

        else:
            if len(joints_positions) != 7:
                raise ValueError(f"joints_positions should be length 7 (got {len(joints_positions)} instead)!")
            req_params["position"] = list_to_arm_position(joints_positions, degrees)
        req = ArmFKRequest(**req_params)
        resp = self._stub.ComputeArmFK(req)
        if not resp.success:
            raise ValueError(f"No solution found for the given joints ({joints_positions})!")

        return np.array(resp.end_effector.pose.data).reshape((4, 4))

    def inverse_kinematics(
        self,
        target: npt.NDArray[np.float64],
        q0: Optional[List[float]] = None,
        degrees: bool = True,
    ) -> List[float]:
        """Compute a joint configuration to reach a specified target pose for the arm end-effector.

        Args:
            target: A 4x4 homogeneous pose matrix representing the target pose in
                Reachy coordinate system, provided as a NumPy array.
            q0: An optional initial joint configuration for the arm. If provided, the
                algorithm will use it as a starting point for finding a solution. Defaults to None.
            degrees: Indicates whether the returned joint angles should be in degrees or radians.
                If `True`, angles are in degrees; if `False`, in radians. Defaults to True.
            round: Number of decimal places to round the computed joint angles to before
                returning. If None, no rounding is performed. Defaults to None.

        Returns:
            A list of joint angles representing the solution to reach the target pose, in the following order:
                [shoulder_pitch, shoulder_roll, elbo_yaw, elbow_pitch, wrist.roll, wrist.pitch, wrist.yaw].

        Raises:
            ValueError: If the target shape is not (4, 4).
            ValueError: If the length of `q0` is not 7.
            ValueError: If vectorized kinematics is attempted (unsupported).
            ValueError: If no solution is found for the given target.
        """
        if target.shape != (4, 4):
            raise ValueError("target shape should be (4, 4) (got {target.shape} instead)!")

        if q0 is not None and (len(q0) != 7):
            raise ValueError(f"q0 should be length 7 (got {len(q0)} instead)!")

        if isinstance(q0, np.ndarray) and len(q0.shape) > 1:
            raise ValueError("Vectorized kinematics not supported!")

        req_params = {
            "target": ArmEndEffector(
                pose=Matrix4x4(data=target.flatten().tolist()),
            ),
            "id": self._part_id,
        }

        if q0 is not None:
            req_params["q0"] = list_to_arm_position(q0, degrees)

        else:
            present_joints_positions = [
                joint.present_position for orbita in self._actuators.values() for joint in orbita._joints.values()
            ]
            req_params["q0"] = list_to_arm_position(present_joints_positions, degrees)

        req = ArmIKRequest(**req_params)
        resp = self._stub.ComputeArmIK(req)

        if not resp.success:
            raise ValueError(f"No solution found for the given target ({target})!")

        answer: List[float] = arm_position_to_list(resp.arm_position, degrees)
        return answer

    @overload
    def goto(
        self,
        target: List[float],
        duration: float = 2,
        wait: bool = False,
        interpolation_space: str = "joint_space",
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
        q0: Optional[List[float]] = None,
    ) -> GoToId:
        ...  # pragma: no cover

    @overload
    def goto(
        self,
        target: npt.NDArray[np.float64],
        duration: float = 2,
        wait: bool = False,
        interpolation_space: str = "joint_space",
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
        q0: Optional[List[float]] = None,
        arc_direction: str = "above",
        secondary_radius: Optional[float] = None,
    ) -> GoToId:
        ...  # pragma: no cover

    def goto(
        self,
        target: Any,
        duration: float = 2,
        wait: bool = False,
        interpolation_space: str = "joint_space",
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
        q0: Optional[List[float]] = None,
        arc_direction: str = "above",
        secondary_radius: Optional[float] = None,
    ) -> GoToId:
        """Move the arm to a specified target position, either in joint space or Cartesian space.

        This function allows the arm to move to a specified target using either:
        - A list of 7 joint positions, or
        - A 4x4 pose matrix representing the desired end-effector position.

        The function also supports an optional initial configuration `q0` for
        computing the inverse kinematics solution when the target is in Cartesian space.

        Args:
            target: The target position. It can either be a list of 7 joint values (for joint space)
                    or a 4x4 NumPy array (for Cartesian space).
            duration: The time in seconds for the movement to be completed. Defaults to 2.
            wait: If True, the function waits until the movement is completed before returning.
                    Defaults to False.
            interpolation_space: The space in which the interpolation should be performed. It can
                    be either "joint_space" or "cartesian_space". Defaults to "joint_space".
            interpolation_mode: The interpolation method to be used. It can be either "minimum_jerk",
                    "linear" or "elliptical". Defaults to "minimum_jerk".
            degrees: If True, the joint values in the `target` argument are treated as degrees.
                    Defaults to True.
            q0: An optional list of 7 joint values representing the initial configuration
                    for inverse kinematics. Defaults to None.
            arc_direction: The direction of the arc to be followed during elliptical interpolation.
                    Can be "above", "below", "front", "back", "left" or "right" . Defaults to "above".
            secondary_radius: The secondary radius of the ellipse for elliptical interpolation, in meters.

        Returns:
            GoToId: The unique GoToId identifier for the movement command.

        Raises:
            TypeError: If the `target` is neither a list nor a pose matrix.
            TypeError: If the `q0` is not a list.
            ValueError: If the `target` list has a length other than 7, or the pose matrix is not
                of shape (4, 4).
            ValueError: If the `q0` list has a length other than 7.
            ValueError: If the `duration` is set to 0.
        """
        self._check_goto_parameters(target, duration, q0)

        if self.is_off(check_gripper=False):
            self._logger.warning(f"{self._part_id.name} is off. Goto not sent.")
            return GoToId(id=-1)

        if interpolation_space == "joint_space" and interpolation_mode == "elliptical":
            self._logger.warning("Elliptical interpolation is not supported in joint space. Switching to linear.")
            interpolation_mode = "linear"
        if secondary_radius is not None and secondary_radius > 0.3:
            self._logger.warning("Interpolation secondary_radius was too large, reduced to 0.3")
            secondary_radius = 0.3

        if isinstance(target, list) and len(target) == 7:
            response = self._goto_joints(
                target,
                duration,
                interpolation_space,
                interpolation_mode,
                degrees,
            )
        elif isinstance(target, np.ndarray) and target.shape == (4, 4):
            response = self._goto_from_matrix(
                target, duration, interpolation_space, interpolation_mode, q0, arc_direction, secondary_radius
            )

        if response.id == -1:
            self._logger.error("Target was not reachable. No command sent.")
        elif wait:
            self._wait_goto(response, duration)

        return response

    def _goto_joints(
        self,
        target: List[float],
        duration: float,
        interpolation_space: str,
        interpolation_mode: str,
        degrees: bool,
    ) -> GoToId:
        """Handle movement to a specified position in joint space.

        Args:
            target: A list of 7 joint positions to move the arm to.
            duration: The time in seconds for the movement to be completed.
            interpolation_space: The space in which the interpolation should be performed.
                    Only "joint_space" is supported for joints target.
            interpolation_mode: The interpolation method to be used. Can be "minimum_jerk" or "linear".
            degrees: If True, the joint positions are interpreted as degrees; otherwise, as radians.

        Returns:
            GoToId: A unique identifier for the movement command.
        """
        if isinstance(target, np.ndarray):
            target = target.tolist()
        arm_pos = list_to_arm_position(target, degrees)

        if interpolation_space == "cartesian_space":
            self._logger.warning(
                "cartesian_space interpolation is not supported using joints target. Switching to joint_space interpolation."
            )
            interpolation_space == "joint_space"
        if interpolation_mode == "elliptical":
            self._logger.warning("Elliptical interpolation is not supported in joint space. Switching to linear.")
            interpolation_mode = "linear"

        req_params = {
            "joints_goal": JointsGoal(
                arm_joint_goal=ArmJointGoal(id=self._part_id, joints_goal=arm_pos, duration=FloatValue(value=duration))
            ),
            "interpolation_space": get_grpc_interpolation_space(interpolation_space),
            "interpolation_mode": get_grpc_interpolation_mode(interpolation_mode),
        }

        request = GoToRequest(**req_params)

        return self._goto_stub.GoToJoints(request)

    def _goto_from_matrix(
        self,
        target: npt.NDArray[np.float64],
        duration: float,
        interpolation_space: str,
        interpolation_mode: str,
        q0: Optional[List[float]],
        arc_direction: str,
        secondary_radius: Optional[float],
    ) -> GoToId:
        """Handle movement to a Cartesian target using a 4x4 transformation matrix.

        This function computes and sends a command to move the arm to a Cartesian target specified by a
        4x4 homogeneous transformation matrix. Optionally, an initial joint configuration (`q0`) can be provided
        for the inverse kinematics calculation.

        Args:
            target: A 4x4 NumPy array representing the Cartesian target pose.
            duration: The time in seconds for the movement to be completed.
            interpolation_space: The space in which the interpolation should be performed. Can be "joint_space"
                    or "cartesian_space".
            interpolation_mode: The interpolation method to be used. Can be "minimum_jerk", "linear" or "elliptical".
            q0: An optional list of 7 joint positions representing the initial configuration. Defaults to None.
            arc_direction: The direction of the arc to be followed during elliptical interpolation. Can be "above",
                    "below", "front", "back", "left" or "right".
            secondary_radius: The secondary radius of the ellipse for elliptical interpolation, in meters.

        Returns:
            GoToId: A unique identifier for the movement command.

        Raises:
            ValueError: If the length of `q0` is not 7.
        """
        goal_pose = Matrix4x4(data=target.flatten().tolist())

        req_params = {
            "cartesian_goal": CartesianGoal(
                arm_cartesian_goal=ArmCartesianGoal(
                    id=self._part_id,
                    goal_pose=goal_pose,
                    duration=FloatValue(value=duration),
                    q0=list_to_arm_position(q0) if q0 is not None else None,
                )
            ),
            "interpolation_space": get_grpc_interpolation_space(interpolation_space),
            "interpolation_mode": get_grpc_interpolation_mode(interpolation_mode),
        }

        if interpolation_mode == "elliptical":
            ellipse_params = {
                "arc_direction": get_grpc_arc_direction(arc_direction),
            }
            if secondary_radius is not None:
                ellipse_params["secondary_radius"] = FloatValue(value=secondary_radius)
            elliptical_params = EllipticalGoToParameters(**ellipse_params)
            req_params["elliptical_parameters"] = elliptical_params

        request = GoToRequest(**req_params)

        return self._goto_stub.GoToCartesian(request)

    def _check_goto_parameters(self, target: Any, duration: Optional[float] = 0, q0: Optional[List[float]] = None) -> None:
        """Check the validity of the parameters for the `goto` method.

        Args:
            duration: The time in seconds for the movement to be completed.
            target: The target position, either a list of joint positions or a 4x4 pose matrix.
            q0: An optional initial joint configuration for inverse kinematics. Defaults to None.

        Raises:
            TypeError: If the target is not a list or a NumPy matrix.
            ValueError: If the target list has a length other than 7, or the pose matrix is not of
                shape (4, 4).
            ValueError: If the duration is set to 0.
            ValueError: If the length of `q0` is not 7.
        """
        if not (isinstance(target, list) or isinstance(target, np.ndarray)):
            raise TypeError(f"Invalid target: must be either a list or a np matrix, got {type(target)} instead.")

        elif isinstance(target, list) and not (len(target) == 7):
            raise ValueError(f"The joints list should be of length 7 (got {len(target)} instead).")
        elif isinstance(target, np.ndarray) and not (target.shape == (4, 4)):
            raise ValueError(f"The pose matrix should be of shape (4, 4) (got {target.shape} instead).")

        elif q0 is not None:
            if not isinstance(q0, list):
                raise TypeError("Invalid q0: must be a list.")
            elif len(q0) != 7:
                raise ValueError(f"q0 should be of length 7 (got {len(q0)} instead)!")

        elif duration == 0:
            raise ValueError("duration cannot be set to 0.")

    def _goto_single_joint(
        self,
        arm_joint: int,
        goal_position: float,
        duration: float = 2,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
    ) -> GoToId:
        """Move a single joint of the arm to a specified position.

        The function allows for optional parameters for duration, interpolation mode, and waiting for completion.

        Args:
            arm_joint: The specific joint of the arm to move, identified by an integer value.
            goal_position: The target position for the specified arm joint, given as a float.
                The value can be in radians or degrees, depending on the `degrees` parameter.
            duration: The time duration in seconds for the joint to reach the specified goal
                position. Defaults to 2.
            wait: Determines whether the program should wait for the movement to finish before
                returning. If set to `True`, the program waits for the movement to complete before continuing
                execution. Defaults to `False`.
            interpolation_mode: The type of interpolation to use when moving the arm's joint.
                Can be 'minimum_jerk' or 'linear'. Defaults to 'minimum_jerk'.
            degrees: Specifies whether the joint positions are in degrees. If set to `True`,
                the goal position is interpreted as degrees. Defaults to `True`.

        Returns:
            A unique GoToId identifier corresponding to this specific goto movement.
        """
        if degrees:
            goal_position = np.deg2rad(goal_position)
        request = GoToRequest(
            joints_goal=JointsGoal(
                custom_joint_goal=CustomJointGoal(
                    id=self._part_id,
                    arm_joints=CustomArmJoints(joints=[arm_joint]),
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
            self._wait_goto(response, duration)
        return response

    def goto_posture(
        self,
        common_posture: str = "default",
        duration: float = 2,
        wait: bool = False,
        wait_for_goto_end: bool = True,
        interpolation_mode: str = "minimum_jerk",
        open_gripper: bool = False,
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
            open_gripper: If `True`, the gripper will open, if `False`, it stays in its current position.
                Defaults to `False`.

        Returns:
            A unique GoToId identifier for this specific movement.
        """
        joints = self.get_default_posture_joints(common_posture=common_posture)
        if self._gripper is not None and self._gripper.is_on() and open_gripper:
            self._gripper.open()
        if not wait_for_goto_end:
            self.cancel_all_goto()
        if self.is_on():
            return self.goto(joints, duration, wait, interpolation_mode=interpolation_mode)
        else:
            self._logger.warning(f"{self._part_id.name} is off. No command sent.")
        return GoToId(id=-1)

    def get_default_posture_joints(self, common_posture: str = "default") -> List[float]:
        """Get the list of joint positions for default or elbow_90 poses.

        Args:
            common_posture: The name of the posture to retrieve. Can be "default" or "elbow_90".
                Defaults to "default".

        Returns:
            A list of joint positions in degrees for the specified posture.

        Raises:
            ValueError: If `common_posture` is not "default" or "elbow_90".
        """
        if common_posture not in ["default", "elbow_90"]:
            raise ValueError(f"common_posture {common_posture} not supported! Should be 'default' or 'elbow_90'")
        if common_posture == "elbow_90":
            elbow_pitch = -90
        else:
            elbow_pitch = 0
        if self._part_id.name == "r_arm":
            return [0, 10, -10, elbow_pitch, 0, 0, 0]
        else:
            return [0, -10, 10, elbow_pitch, 0, 0, 0]

    def get_default_posture_matrix(self, common_posture: str = "default") -> npt.NDArray[np.float64]:
        """Get the 4x4 pose matrix in Reachy coordinate system for a default robot posture.

        Args:
            common_posture: The posture to retrieve. Can be "default" or "elbow_90".
                Defaults to "default".

        Returns:
            The 4x4 homogeneous pose matrix for the specified posture in Reachy coordinate system.
        """
        joints = self.get_default_posture_joints(common_posture)
        return self.forward_kinematics(joints)

    def get_translation_by(
        self,
        x: float,
        y: float,
        z: float,
        initial_pose: Optional[npt.NDArray[np.float64]] = None,
        frame: str = "robot",
    ) -> npt.NDArray[np.float64]:
        """Return a 4x4 matrix representing a pose translated by specified x, y, z values.

        The translation is performed in either the robot or gripper coordinate system.

        Args:
            x: Translation along the x-axis in meters (forwards direction) to apply
                to the pose matrix.
            y: Translation along the y-axis in meters (left direction) to apply
                to the pose matrix.
            z: Translation along the z-axis in meters (upwards direction) to apply
                to the pose matrix.
            initial_pose: A 4x4 matrix representing the initial pose of the end-effector in Reachy coordinate system,
                expressed as a NumPy array of type `np.float64`.
                If not provided, the current pose of the arm is used. Defaults to `None`.
            frame: The coordinate system in which the translation should be performed.
                Can be either "robot" or "gripper". Defaults to "robot".

        Returns:
            A 4x4 pose matrix, expressed in Reachy coordinate system,
            translated by the specified x, y, z values from the initial pose.

        Raises:
            ValueError: If the `frame` is not "robot" or "gripper".
        """
        if frame not in ["robot", "gripper"]:
            raise ValueError(f"Unknown frame {frame}! Should be 'robot' or 'gripper'")

        if initial_pose is None:
            initial_pose = self.forward_kinematics()

        pose = initial_pose.copy()

        if frame == "robot":
            pose[0, 3] += x
            pose[1, 3] += y
            pose[2, 3] += z
        elif frame == "gripper":
            pose = translate_in_self(initial_pose, [x, y, z])

        return pose

    def translate_by(
        self,
        x: float,
        y: float,
        z: float,
        duration: float = 2,
        wait: bool = False,
        frame: str = "robot",
        interpolation_space: str = "cartesian_space",
        interpolation_mode: str = "minimum_jerk",
        arc_direction: str = "above",
        secondary_radius: Optional[float] = None,
    ) -> GoToId:
        """Create a translation movement for the arm's end effector.

        The movement is based on the last sent position or the current position.

        Args:
            x: Translation along the x-axis in meters (forwards direction) to apply
                to the pose matrix.
            y: Translation along the y-axis in meters (left direction) to apply
                to the pose matrix.
            z: Translation along the z-axis in meters (vertical direction) to apply
                to the pose matrix.
            duration: Time duration in seconds for the translation movement to be completed.
                Defaults to 2.
            wait: Determines whether the program should wait for the movement to finish before
                returning. If set to `True`, the program waits for the movement to complete before continuing
                execution. Defaults to `False`.
            frame: The coordinate system in which the translation should be performed.
                Can be "robot" or "gripper". Defaults to "robot".
            interpolation_mode: The type of interpolation to be used when moving the arm's
                joints. Can be 'minimum_jerk' or 'linear'. Defaults to 'minimum_jerk'.

        Returns:
            The GoToId of the movement command, created using the `goto_from_matrix` method with the
            translated pose computed in the specified frame.

        Raises:
            ValueError: If the `frame` is not "robot" or "gripper".
        """
        try:
            goto = self.get_goto_queue()[-1]
        except IndexError:
            goto = self.get_goto_playing()

        if goto.id != -1:
            joints_request = self._get_goto_request(goto)
        else:
            joints_request = None

        if joints_request is not None:
            if joints_request.request.target.joints is not None:
                pose = self.forward_kinematics(joints_request.request.target.joints)
            else:
                pose = joints_request.request.target.pose
        else:
            pose = self.forward_kinematics()

        pose = self.get_translation_by(x, y, z, initial_pose=pose, frame=frame)
        return self.goto(
            pose,
            duration=duration,
            wait=wait,
            interpolation_space=interpolation_space,
            interpolation_mode=interpolation_mode,
            arc_direction=arc_direction,
            secondary_radius=secondary_radius,
        )

    def get_rotation_by(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        initial_pose: Optional[npt.NDArray[np.float64]] = None,
        degrees: bool = True,
        frame: str = "robot",
    ) -> npt.NDArray[np.float64]:
        """Calculate a new pose matrix by rotating an initial pose matrix by specified roll, pitch, and yaw angles.

        The rotation is performed in either the robot or gripper coordinate system.

        Args:
            roll: Rotation around the x-axis in the Euler angles representation, specified
                in radians or degrees (based on the `degrees` parameter).
            pitch: Rotation around the y-axis in the Euler angles representation, specified
                in radians or degrees (based on the `degrees` parameter).
            yaw: Rotation around the z-axis in the Euler angles representation, specified
                in radians or degrees (based on the `degrees` parameter).
            initial_pose: A 4x4 matrix representing the initial
                pose of the end-effector, expressed as a NumPy array of type `np.float64`. If not provided,
                the current pose of the arm is used. Defaults to `None`.
            degrees: Specifies whether the rotation angles are provided in degrees. If set to
                `True`, the angles are interpreted as degrees. Defaults to `True`.
            frame: The coordinate system in which the rotation should be performed. Can be
                "robot" or "gripper". Defaults to "robot".

        Returns:
            A 4x4 pose matrix, expressed in the Reachy coordinate system, rotated
            by the specified roll, pitch, and yaw angles from the initial pose, in the specified frame.

        Raises:
            ValueError: If the `frame` is not "robot" or "gripper".
        """
        if frame not in ["robot", "gripper"]:
            raise ValueError(f"Unknown frame {frame}! Should be 'robot' or 'gripper'")

        if initial_pose is None:
            initial_pose = self.forward_kinematics()

        pose = initial_pose.copy()
        rotation = matrix_from_euler_angles(roll, pitch, yaw, degrees=degrees)

        if frame == "robot":
            pose_rotation = np.eye(4)
            pose_rotation[:3, :3] = pose.copy()[:3, :3]
            pose_translation = pose.copy()[:3, 3]
            pose_rotation = (rotation @ pose_rotation).astype(np.float64)
            pose = recompose_matrix(pose_rotation[:3, :3], pose_translation).astype(np.float64)
        elif frame == "gripper":
            pose = rotate_in_self(initial_pose, [roll, pitch, yaw], degrees=degrees).astype(np.float64)

        return pose.astype(np.float64)

    def rotate_by(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        duration: float = 2,
        wait: bool = False,
        degrees: bool = True,
        frame: str = "robot",
        interpolation_mode: str = "minimum_jerk",
    ) -> GoToId:
        """Create a rotation movement for the arm's end effector based on the specified roll, pitch, and yaw angles.

        The rotation is performed in either the robot or gripper frame.

        Args:
            roll: Rotation around the x-axis in the Euler angles representation, specified
                in radians or degrees (based on the `degrees` parameter).
            pitch: Rotation around the y-axis in the Euler angles representation, specified
                in radians or degrees (based on the `degrees` parameter).
            yaw: Rotation around the z-axis in the Euler angles representation, specified
                in radians or degrees (based on the `degrees` parameter).
            duration: Time duration in seconds for the rotation movement to be completed.
                Defaults to 2.
            wait: Determines whether the program should wait for the movement to finish before
                returning. If set to `True`, the program waits for the movement to complete before continuing
                execution. Defaults to `False`.
            degrees: Specifies whether the rotation angles are provided in degrees. If set to
                `True`, the angles are interpreted as degrees. Defaults to `True`.
            frame: The coordinate system in which the rotation should be performed. Can be
                "robot" or "gripper". Defaults to "robot".
            interpolation_mode: The type of interpolation to be used when moving the arm's
                joints. Can be 'minimum_jerk' or 'linear'. Defaults to 'minimum_jerk'.

        Returns:
            The GoToId of the movement command, created by calling the `goto_from_matrix` method with
            the rotated pose computed in the specified frame.

        Raises:
            ValueError: If the `frame` is not "robot" or "gripper".
        """
        if frame not in ["robot", "gripper"]:
            raise ValueError(f"Unknown frame {frame}! Should be 'robot' or 'gripper'")

        try:
            goto = self.get_goto_queue()[-1]
        except IndexError:
            goto = self.get_goto_playing()

        if goto.id != -1:
            joints_request = self._get_goto_request(goto)
        else:
            joints_request = None

        if joints_request is not None:
            if joints_request.request.target.joints is not None:
                pose = self.forward_kinematics(joints_request.request.target.joints)
            else:
                pose = joints_request.request.target.pose
        else:
            pose = self.forward_kinematics()

        pose = self.get_rotation_by(roll, pitch, yaw, initial_pose=pose, degrees=degrees, frame=frame)
        return self.goto(pose, duration=duration, wait=wait, interpolation_mode=interpolation_mode)

    # @property
    # def joints_limits(self) -> ArmLimits:
    #     """Get limits of all the part's joints"""
    #     limits = self._arm_stub.GetJointsLimits(self._part_id)
    #     return limits

    # @property
    # def temperatures(self) -> ArmTemperatures:
    #     """Get temperatures of all the part's motors"""
    #     temperatures = self._arm_stub.GetTemperatures(self._part_id)
    #     return temperatures

    def _get_goal_positions_message(self) -> ArmComponentsCommands:
        """Get the ArmComponentsCommands message to send the goal positions to the actuator."""
        commands = {}
        for actuator_name, actuator in self._actuators.items():
            if actuator_name != "gripper":
                actuator_command = actuator._get_goal_positions_message()
                if actuator_command is not None:
                    commands[f"{actuator_name}_command"] = actuator_command
        return ArmComponentsCommands(**commands)

    def _clean_outgoing_goal_positions(self) -> None:
        """Clean the outgoing goal positions."""
        for actuator in [self._actuators[act] for act in self._actuators.keys() if act not in ["gripper"]]:
            actuator._clean_outgoing_goal_positions()

    def _post_send_goal_positions(self) -> None:
        """Monitor the joint positions to check if they reach the specified goals."""
        for actuator in [self._actuators[act] for act in self._actuators.keys() if act not in ["gripper"]]:
            actuator._post_send_goal_positions()

    def send_goal_positions(self, check_positions: bool = False) -> None:
        """Send goal positions to the arm's joints, including the gripper.

        If goal positions have been specified for any joint of the part, sends them to the robot.

        Args :
            check_positions: A boolean indicating whether to check the positions after sending the command.
                Defaults to True.
        """
        super().send_goal_positions(check_positions)
        if self.gripper is not None:
            self.gripper.send_goal_positions(check_positions)

    def _update_with(self, new_state: ArmState) -> None:
        """Update the arm with a newly received (partial) state from the gRPC server.

        Updating the shoulder, elbow, and wrist states accordingly.

        Args:
            new_state: current state of the arm, including the states of the shoulder, elbow, and wrist.
        """
        self.shoulder._update_with(new_state.shoulder_state)
        self.elbow._update_with(new_state.elbow_state)
        self.wrist._update_with(new_state.wrist_state)

    def _update_audit_status(self, new_status: ArmStatus) -> None:
        """Update the audit status of different components based on a new overall status.

        Args:
            new_status: new status of the shoulder, elbow, and  wrist.
        """
        self.shoulder._update_audit_status(new_status.shoulder_status)
        self.elbow._update_audit_status(new_status.elbow_status)
        self.wrist._update_audit_status(new_status.wrist_status)
