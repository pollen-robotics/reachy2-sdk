"""Reachy Arm module.

Handles all specific methods to an Arm (left and/or right).
"""

import time
from typing import Any, Dict, List, Optional, overload

import grpc
import numpy as np
import numpy.typing as npt
from google.protobuf.wrappers_pb2 import FloatValue
from pyquaternion import Quaternion
from reachy2_sdk_api.arm_pb2 import Arm as Arm_proto
from reachy2_sdk_api.arm_pb2 import (  # ArmLimits,; ArmTemperatures,
    ArmCartesianGoal,
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
    GoToId,
    GoToRequest,
    JointsGoal,
)
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.hand_pb2 import Hand as HandState
from reachy2_sdk_api.hand_pb2 import Hand as Hand_proto
from reachy2_sdk_api.kinematics_pb2 import Matrix4x4

from ..orbita.orbita2d import Orbita2d
from ..orbita.orbita3d import Orbita3d
from ..utils.utils import (
    arm_position_to_list,
    decompose_matrix,
    get_grpc_interpolation_mode,
    get_normal_vector,
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
        IGoToBasedPart.__init__(self, self, goto_stub)

        self._setup_arm(arm_msg, initial_state)
        self._gripper: Optional[Hand] = None

        self._actuators: Dict[str, Orbita2d | Orbita3d] = {}
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
        self._gripper = Hand(hand, hand_initial_state, self._grpc_channel)

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

    def turn_on(self) -> None:
        """Turn on all motors of the part, making all arm motors stiff.

        If a gripper is present, it will also be turned on.
        """
        if self._gripper is not None:
            self._gripper._turn_on()
        super().turn_on()

    def turn_off(self) -> None:
        """Turn off all motors of the part, making all arm motors compliant.

        If a gripper is present, it will also be turned off.
        """
        if self._gripper is not None:
            self._gripper._turn_off()
        super().turn_off()

    def turn_off_smoothly(self) -> None:
        """Gradually reduce the torque limit of all motors over 3 seconds before turning them off.

        This function decreases the torque limit in steps until the motors are turned off.
        It then restores the torque limit to its original value.
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

    def _turn_on(self) -> None:
        """Turn on all motors of the part.

        This will make all arm motors stiff. If a gripper is present, it will also be turned on.
        """
        if self._gripper is not None:
            self._gripper._turn_on()
        super()._turn_on()

    def _turn_off(self) -> None:
        """Turn off all motors of the part.

        This will make all arm motors compliant. If a gripper is present, it will also be turned off.
        """
        if self._gripper is not None:
            self._gripper._turn_off()
        super()._turn_off()

    def is_on(self) -> bool:
        """Check if all actuators of the arm are stiff.

        Returns:
            `True` if all actuators of the arm are stiff, `False` otherwise.
        """
        if not super().is_on():
            return False
        return True

    def is_off(self) -> bool:
        """Check if all actuators of the arm are compliant.

        Returns:
            `True` if all actuators of the arm are compliant, `False` otherwise.
        """
        if not super().is_off():
            return False
        return True

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
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
        q0: Optional[List[float]] = None,
    ) -> GoToId: ...

    @overload
    def goto(
        self,
        target: npt.NDArray[np.float64],
        duration: float = 2,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
        q0: Optional[List[float]] = None,
    ) -> GoToId: ...

    def goto(
        self,
        target: Any,
        duration: float = 2,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
        q0: Optional[List[float]] = None,
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
            interpolation_mode: The interpolation method to be used. It can be either "minimum_jerk"
                    or "linear". Defaults to "minimum_jerk".
            degrees: If True, the joint values in the `target` argument are treated as degrees.
                    Defaults to True.
            q0: An optional list of 7 joint values representing the initial configuration
                    for inverse kinematics. Defaults to None.

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
        self._check_goto_parameters(duration, target, q0)

        if self.is_off():
            self._logger.warning(f"{self._part_id.name} is off. Goto not sent.")
            return GoToId(id=-1)

        if isinstance(target, list) and len(target) == 7:
            response = self._goto_joints(target, duration, interpolation_mode, degrees)
        elif isinstance(target, np.ndarray) and target.shape == (4, 4):
            response = self._goto_from_matrix(target, duration, interpolation_mode, q0)

        if response.id == -1:
            self._logger.error("Target was not reachable. No command sent.")

        if wait:
            self._logger.info(f"Waiting for movement with {response}.")
            while not self._is_goto_finished(response):
                time.sleep(0.1)
            self._logger.info(f"Movement with {response} finished.")

        return response

    def _goto_joints(self, target: List[float], duration: float, interpolation_mode: str, degrees: bool) -> GoToId:
        """Handle movement to a specified position in joint space.

        Args:
            target: A list of 7 joint positions to move the arm to.
            duration: The time in seconds for the movement to be completed.
            interpolation_mode: The interpolation method to be used. Can be "minimum_jerk" or "linear".
            degrees: If True, the joint positions are interpreted as degrees; otherwise, as radians.

        Returns:
            GoToId: A unique identifier for the movement command.
        """
        if isinstance(target, np.ndarray):
            target = target.tolist()
        arm_pos = list_to_arm_position(target, degrees)
        request = GoToRequest(
            joints_goal=JointsGoal(
                arm_joint_goal=ArmJointGoal(id=self._part_id, joints_goal=arm_pos, duration=FloatValue(value=duration))
            ),
            interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
        )
        return self._goto_stub.GoToJoints(request)

    def _goto_from_matrix(
        self, target: npt.NDArray[np.float64], duration: float, interpolation_mode: str, q0: Optional[List[float]]
    ) -> GoToId:
        """Handle movement to a Cartesian target using a 4x4 transformation matrix.

        This function computes and sends a command to move the arm to a Cartesian target specified by a
        4x4 homogeneous transformation matrix. Optionally, an initial joint configuration (`q0`) can be provided
        for the inverse kinematics calculation.

        Args:
            target: A 4x4 NumPy array representing the Cartesian target pose.
            duration: The time in seconds for the movement to be completed.
            interpolation_mode: The interpolation method to be used. Can be "minimum_jerk" or "linear".
            q0: An optional list of 7 joint positions representing the initial configuration. Defaults to None.

        Returns:
            GoToId: A unique identifier for the movement command.

        Raises:
            ValueError: If the length of `q0` is not 7.
        """
        goal_pose = Matrix4x4(data=target.flatten().tolist())
        request = GoToRequest(
            cartesian_goal=CartesianGoal(
                arm_cartesian_goal=ArmCartesianGoal(
                    id=self._part_id,
                    goal_pose=goal_pose,
                    duration=FloatValue(value=duration),
                    q0=list_to_arm_position(q0) if q0 is not None else None,
                )
            ),
            interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
        )
        return self._goto_stub.GoToCartesian(request)

    def _check_goto_parameters(self, duration: float, target: Any, q0: Optional[List[float]] = None) -> None:
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
        if wait:
            self._logger.info(f"Waiting for movement with {response}.")
            while not self._is_goto_finished(response):
                time.sleep(0.1)
            self._logger.info(f"Movement with {response} finished.")
        return response

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
        joints = self.get_default_posture_joints(common_posture=common_posture)
        if common_posture == "default":
            if self._gripper is not None and self._gripper.is_on():
                self._gripper.open()
        if not wait_for_goto_end:
            self.cancel_all_goto()
        if self.is_on():
            return self.goto(joints, duration, wait, interpolation_mode)
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
            return [0, -15, -15, elbow_pitch, 0, 0, 0]
        else:
            return [0, 15, 15, elbow_pitch, 0, 0, 0]

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
        interpolation_mode: str = "minimum_jerk",
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
            joints_request = self._get_goto_joints_request(goto)
        else:
            joints_request = None

        if joints_request is not None:
            pose = self.forward_kinematics(joints_request.goal_positions)
        else:
            pose = self.forward_kinematics()

        pose = self.get_translation_by(x, y, z, initial_pose=pose, frame=frame)
        return self.goto(pose, duration=duration, wait=wait, interpolation_mode=interpolation_mode)

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
            pose_rotation = rotation @ pose_rotation
            pose = recompose_matrix(pose_rotation[:3, :3], pose_translation)
        elif frame == "gripper":
            pose = rotate_in_self(initial_pose, [roll, pitch, yaw], degrees=degrees)

        return pose

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
            joints_request = self._get_goto_joints_request(goto)
        else:
            joints_request = None

        if joints_request is not None:
            pose = self.forward_kinematics(joints_request.goal_positions)
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

    def send_cartesian_interpolation(
        self,
        target: npt.NDArray[np.float64],
        duration: float = 2,
        arc_direction: Optional[str] = None,
        elliptic_radius: Optional[float] = None,
        interpolation_frequency: float = 120,
        precision_distance_xyz: float = 0.003,
    ) -> None:
        """Perform Cartesian interpolation and move the arm towards a target pose.

        The function uses linear or elliptical interpolation for translation to reach or get close
        to the specified target pose.

        Args:
            target: A 4x4 homogeneous pose matrix representing the desired
                position and orientation in the Reachy coordinate system, provided as a NumPy array.
            duration: The expected time in seconds for the arm to reach the target position
                from its current position. Defaults to 2.
            arc_direction: The direction for elliptic interpolation when moving
                the arm towards the target pose. Can be 'above', 'below', 'right', 'left', 'front',
                or 'back'. If not specified, a linear interpolation is computed.
            elliptic_radius: The second radius of the computed ellipse for elliptical
                interpolation. The first radius is the distance between the current pose and the
                target pose. If not specified, a circular interpolation is used.
            interpolation_frequency: The number of intermediate points used to interpolate
                the movement in Cartesian space between the initial and target poses. Defaults to 120.
            precision_distance_xyz: The maximum allowed distance in meters in the XYZ space between
                the current end-effector position and the target position. If the end-effector is
                further than this distance from the target after the movement, the movement is repeated
                until the precision is met. Defaults to 0.003.

        Raises:
            TypeError: If the target is not a NumPy matrix.
            ValueError: If the target shape is not (4, 4).
            ValueError: If the duration is set to 0.
        """
        self.cancel_all_goto()
        if not isinstance(target, np.ndarray):
            raise TypeError(f"target should be a NumPy array (got {type(target)} instead)!")
        if target.shape != (4, 4):
            raise ValueError("target shape should be (4, 4) (got {target.shape} instead)!")
        if duration == 0:
            raise ValueError("duration cannot be set to 0.")
        if self.is_off():
            self._logger.warning(f"{self._part_id.name} is off. Commands not sent.")
            return
        try:
            self.inverse_kinematics(target)
        except ValueError:
            raise ValueError(f"Target pose: \n{target}\n is not reachable!")

        origin_matrix = self.forward_kinematics()
        nb_steps = int(duration * interpolation_frequency)
        time_step = duration / nb_steps

        q1, trans1 = decompose_matrix(origin_matrix)
        q2, trans2 = decompose_matrix(target)

        if arc_direction is None:
            self._send_linear_interpolation(trans1, trans2, q1, q2, nb_steps=nb_steps, time_step=time_step)

        else:
            self._send_elliptical_interpolation(
                trans1,
                trans2,
                q1,
                q2,
                arc_direction=arc_direction,
                secondary_radius=elliptic_radius,
                nb_steps=nb_steps,
                time_step=time_step,
            )

        current_pose = self.forward_kinematics()
        current_precision_distance_xyz = np.linalg.norm(current_pose[:3, 3] - target[:3, 3])
        if current_precision_distance_xyz > precision_distance_xyz:
            request = ArmCartesianGoal(
                id=self._part_id,
                goal_pose=Matrix4x4(data=target.flatten().tolist()),
            )
            self._stub.SendArmCartesianGoal(request)
            time.sleep(time_step)

            current_pose = self.forward_kinematics()
            current_precision_distance_xyz = np.linalg.norm(current_pose[:3, 3] - target[:3, 3])
        self._logger.info(f"l2 xyz distance to goal: {current_precision_distance_xyz}")

    def _send_linear_interpolation(
        self,
        origin_trans: npt.NDArray[np.float64],
        target_trans: npt.NDArray[np.float64],
        origin_rot: Quaternion,
        target_rot: Quaternion,
        nb_steps: int,
        time_step: float,
    ) -> None:
        """Generate linear interpolation between two poses over a specified number of steps.

        The function performs linear interpolation for both translation and rotation between
        the origin and target poses.

        Args:
            origin_trans: The original translation vector in 3D space,
                given as a NumPy array of type `np.float64`, containing the translation components
                along the x, y, and z axes in meters.
            target_trans: The target translation vector in 3D space, given
                as a NumPy array of type `np.float64`, representing the desired final translation in meters.
            origin_rot: The initial rotation quaternion, used as the starting point for
                the rotation interpolation.
            target_rot: The target rotation quaternion for the interpolation.
            nb_steps: The number of steps or intervals for the interpolation process, determining
                how many intermediate points will be calculated between the origin and target poses.
            time_step: The time interval in seconds between each step of the interpolation.
        """
        for t in np.linspace(0, 1, nb_steps):
            # Linear interpolation for translation
            trans_interpolated = (1 - t) * origin_trans + t * target_trans

            # SLERP for rotation interpolation
            q_interpolated = Quaternion.slerp(origin_rot, target_rot, t)
            rot_interpolated = q_interpolated.rotation_matrix

            # Recompose the interpolated matrix
            interpolated_matrix = recompose_matrix(rot_interpolated, trans_interpolated)

            request = ArmCartesianGoal(
                id=self._part_id,
                goal_pose=Matrix4x4(data=interpolated_matrix.flatten().tolist()),
            )
            self._stub.SendArmCartesianGoal(request)
            time.sleep(time_step)

    def _send_elliptical_interpolation(
        self,
        origin_trans: npt.NDArray[np.float64],
        target_trans: npt.NDArray[np.float64],
        origin_rot: Quaternion,
        target_rot: Quaternion,
        arc_direction: str,
        secondary_radius: Optional[float],
        nb_steps: int,
        time_step: float,
    ) -> None:
        """Generate elliptical interpolation between two poses for the arm.

        The function performs elliptical interpolation for both translation and rotation from the
        origin to the target pose.

        Args:
            origin_trans: The initial translation vector of the arm,
                given as a NumPy array of type `np.float64`, containing the x, y, and z coordinates.
            target_trans: The target translation vector that the robot
                end-effector should reach, provided as a NumPy array of type `np.float64` with the x,
                y, and z coordinates.
            origin_rot: The initial orientation (rotation) of the end-effector.
            target_rot: The target orientation (rotation) that the end-effector should reach.
            arc_direction: The direction of the elliptical interpolation, which can be 'above',
                'below', 'right', 'left', 'front', or 'back'.
            secondary_radius: The radius of the secondary axis of the ellipse. If not
                provided, it defaults to the primary radius, which is based on the distance between the
                origin and target poses.
            nb_steps: The number of steps for the interpolation, determining how many
                intermediate poses will be generated between the origin and target.
            time_step: The time interval in seconds between each interpolation step.
        """
        vector_target_origin = target_trans - origin_trans

        center = (origin_trans + target_trans) / 2
        radius = float(np.linalg.norm(vector_target_origin) / 2)

        vector_origin_center = origin_trans - center
        vector_target_center = target_trans - center

        if np.isclose(radius, 0, atol=1e-03):
            self._logger.warning(f"{self._part_id.name} is already at the target pose. No command sent.")
            return
        if secondary_radius is None:
            secondary_radius = radius
        if secondary_radius is not None and secondary_radius > 0.3:
            self._logger.warning("interpolation elliptic_radius was too large, reduced to 0.3")
            secondary_radius = 0.3

        normal = get_normal_vector(vector=vector_target_origin, arc_direction=arc_direction)

        if normal is None:
            self._logger.warning("arc_direction has no solution. Executing linear interpolation instead.")
            self._send_linear_interpolation(
                origin_trans=origin_trans,
                target_trans=target_trans,
                origin_rot=origin_rot,
                target_rot=target_rot,
                nb_steps=nb_steps,
                time_step=time_step,
            )
            return

        cos_angle = np.dot(vector_origin_center, vector_target_center) / (
            np.linalg.norm(vector_origin_center) * np.linalg.norm(vector_target_center)
        )
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        for t in np.linspace(0, 1, nb_steps):
            # Interpolated angles
            theta = t * angle

            # Rotation of origin_vector around the circle center in the plan defined by 'normal'
            q1 = Quaternion(axis=normal, angle=theta)
            rotation_matrix = q1.rotation_matrix

            # Interpolated point in plan
            trans_interpolated = np.dot(rotation_matrix, vector_origin_center)
            # Adjusting the ellipse
            ellipse_interpolated = trans_interpolated * np.array([1, 1, secondary_radius / radius])
            trans_interpolated = ellipse_interpolated + center

            # SLERP for the rotation
            q_interpolated = Quaternion.slerp(origin_rot, target_rot, t)
            rot_interpolated = q_interpolated.rotation_matrix

            # Recompose the interpolated matrix
            interpolated_matrix = recompose_matrix(rot_interpolated, trans_interpolated)

            request = ArmCartesianGoal(
                id=self._part_id,
                goal_pose=Matrix4x4(data=interpolated_matrix.flatten().tolist()),
            )
            self._stub.SendArmCartesianGoal(request)
            time.sleep(time_step)

    def send_goal_positions(self) -> None:
        """Send goal positions to the gripper and actuators if the parts are on.

        The function checks if the gripper and actuators are active before sending the goal positions.
        """
        if self._gripper is not None:
            self._gripper.send_goal_positions()
        if self.is_off():
            self._logger.warning(f"{self._part_id.name} is off. Command not sent.")
            return
        for actuator in self._actuators.values():
            actuator.send_goal_positions()

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
