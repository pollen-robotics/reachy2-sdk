"""Reachy Arm module.

Handles all specific method to an Arm (left and/or right) especially:
- the forward kinematics
- the inverse kinematics
- goto functions
"""

import time
from typing import Dict, List, Optional

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
    """Arm class used for both left/right arms.

    It exposes the kinematics functions for the arm:
    - you can compute the forward and inverse kinematics
    It also exposes movements functions.
    Arm can be turned on and off.
    """

    def __init__(
        self,
        arm_msg: Arm_proto,
        initial_state: ArmState,
        grpc_channel: grpc.Channel,
        goto_stub: GoToServiceStub,
    ) -> None:
        """Define an arm (left or right).

        Connect to the arm's gRPC server stub and set up the arm's actuators.
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
        """
        Initialize the arm's actuators (shoulder, elbow, and wrist) based on the arm's description and initial state.

        Args:
            arm (Arm_proto): The arm description used to set up the actuators, including the shoulder,
                elbow, and wrist. The method creates instances of `Orbita2d` for the shoulder and
                elbow, and an instance of `Orbita3d` for the wrist.
            initial_state (ArmState): The initial state of the arm's actuators, containing the starting
                positions or states of the shoulder, elbow, and wrist. This information is used to
                initialize the corresponding actuators.

        Returns:
            None
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
        """
        The shoulder actuator of the arm.

        Returns:
            Orbita2d: The shoulder actuator.
        """
        return self._shoulder

    @property
    def elbow(self) -> Orbita2d:
        """
        The elbow actuator of the arm.

        Returns:
            Orbita2d: The elbow actuator.
        """
        return self._elbow

    @property
    def wrist(self) -> Orbita3d:
        """
        The wrist actuator of the arm.

        Returns:
            Orbita3d: The wrist actuator.
        """
        return self._wrist

    @property
    def gripper(self) -> Optional[Hand]:
        """
        The gripper of the arm.

        Returns:
            Optional[Hand]: The gripper, or None if not set.
        """
        return self._gripper

    def turn_on(self) -> None:
        """
        The function `turn_on` turns on all motors of the part, making all arm's motors stiff.
        """
        if self._gripper is not None:
            self._gripper._turn_on()
        super().turn_on()

    def turn_off(self) -> None:
        """
        The function `turn_off` turns off all motors of the part, making all arm motors compliant.
        """
        if self._gripper is not None:
            self._gripper._turn_off()
        super().turn_off()

    def _turn_on(self) -> None:
        """Turn all motors of the part on.

        All arm's motors will then be stiff.
        """
        if self._gripper is not None:
            self._gripper._turn_on()
        super()._turn_on()

    def _turn_off(self) -> None:
        """Turn all motors of the part off.

        All arm's motors will then be compliant.
        """
        if self._gripper is not None:
            self._gripper._turn_off()
        super()._turn_off()

    def turn_off_smoothly(self) -> None:
        """
        Gradually reduces the torque limit of all motors for 3 seconds before turning them off.
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

    def is_on(self) -> bool:
        """
        Check if all actuators of the arm are stiff.

        Returns:
          bool: `True` if all actuators of the arm are stiff, `False` otherwise.
        """
        if not super().is_on():
            return False
        return True

    def is_off(self) -> bool:
        """
        Check if all actuators of the arm are compliant.

        Returns:
          bool: `True` if all actuators of the arm are compliant, `False` otherwise.
        """
        if not super().is_off():
            return False
        return True

    def __repr__(self) -> str:
        """Clean representation of an Arm."""
        s = "\n\t".join([act_name + ": " + str(actuator) for act_name, actuator in self._actuators.items()])
        return f"""<Arm on={self.is_on()} actuators=\n\t{
            s
        }\n>"""

    def forward_kinematics(
        self, joints_positions: Optional[List[float]] = None, degrees: bool = True
    ) -> npt.NDArray[np.float64]:
        """
        Compute the forward kinematics of the arm and return a 4x4 pose matrix.
        The pose matrix is expressed in the Reachy coordinate system.

        Args:
            - **joints_positions** (Optional[List[float]]): A list of float values representing the positions of the joints
                in the arm. If not provided, the current robot joints positions are used. Defaults to None.
            - **degrees** (bool): Indicates whether the joint positions are in degrees or radians.
                If `True`, the positions are in degrees; if `False`, in radians. Defaults to True.

        Returns:
            - npt.NDArray[np.float64]: A 4x4 pose matrix as a NumPy array, expressed in Reachy coordinate system.

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
        round: Optional[int] = None,
    ) -> List[float]:
        """
        Compute a joint configuration to reach a specified target pose for the arm end-effector.

        Args:
            - **target** (npt.NDArray[np.float64]): A 4x4 homogeneous pose matrix representing the target pose in
                Reachy coordinate system, provided as a NumPy array.
            - **q0** (Optional[List[float]]): An optional initial joint configuration for the arm. If provided, the
                algorithm will use it as a starting point for finding a solution. Defaults to None.
            - **degrees** (bool): Indicates whether the returned joint angles should be in degrees or radians.
                If `True`, angles are in degrees; if `False`, in radians. Defaults to True.
            - **round** (Optional[int]): Number of decimal places to round the computed joint angles to before
                returning. If None, no rounding is performed. Defaults to None.

        Returns:
            - List[float]: A list of joint angles representing the solution to reach the target pose, in the following order:
                [shoulder.pitch, shoulder.roll, elbow.pitch, elbow.yaw, wrist.roll, wrist.pitch, wrist.yaw].

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
        if round is not None:
            answer = np.round(answer, round).tolist()
        return answer

    def get_default_posture_joints(self, common_posture: str = "default") -> List[float]:
        """
        Returns a list of joint positions for default or elbow_90 poses based on the specified common_posture.

        Args:
          common_posture (str): default robot posture for which you want to retrieve the list of joint positions.
        It can have two possible values: "default" or "elbow_90". Defaults to "default".

        Returns:
          a list of joint positions in degrees.
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
        """
        Returns the 4x4 pose matrix of default robot poses based on the common posture provided.

        Args:
          common_posture (str): default robot posture for which you want to retrieve the 4x4 pose matrix.
        It can have two possible values: "default" or "elbow_90". Defaults to "default".

        Returns:
          the 4x4 homogeneous pose matrix of the specified default robot pose.
        """
        joints = self.get_default_posture_joints(common_posture)
        return self.forward_kinematics(joints)

    def goto_from_matrix(
        self,
        target: npt.NDArray[np.float64],
        duration: float = 2,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        q0: Optional[List[float]] = None,
    ) -> GoToId:
        """
        Moves the arm to a specified matrix target position with optional parameters for duration, interpolation mode,
        and initial joint configuration.

        Args:
          target (npt.NDArray[np.float64]): an homogeneous 4x4 pose matrix expressed in Reachy coordinate system, represented
        as a numpy array. This matrix defines the desired position and orientation that the arm should move to.
          duration (float): time duration in seconds for the movement to be completed. Defaults to 2.
          wait (bool): boolean parameter that determines whether the program should wait for the movement to finish before
        returning. If `wait` is set to `True`, the program will wait for the movement to complete before continuing
        execution. Defaults to False
          interpolation_mode (str): type of interpolation to be used when moving the arm's joints to the given
        position. Can be set to 'minimum_jerk' or 'linear'. Defaults to 'minimum_jerk'.
          q0 (Optional[List[float]]): a list of 7 float values corresponding to joint angles of the arm. If provided, it will
        be used as the starting position for computing the inverse kinematics solution to reach the target pose.
        Defaults to None.

        Returns:
          The function `goto_from_matrix` returns a `GoToId` object.
        """
        if target.shape != (4, 4):
            raise ValueError("target shape should be (4, 4) (got {target.shape} instead)!")
        if q0 is not None and (len(q0) != 7):
            raise ValueError(f"q0 should be length 7 (got {len(q0)} instead)!")
        if duration == 0:
            raise ValueError("duration cannot be set to 0.")
        if self.is_off():
            self._logger.warning(f"{self._part_id.name} is off. Goto not sent.")
            return GoToId(id=-1)

        if q0 is not None:
            q0 = list_to_arm_position(q0)
            request = GoToRequest(
                cartesian_goal=CartesianGoal(
                    arm_cartesian_goal=ArmCartesianGoal(
                        id=self._part_id,
                        goal_pose=Matrix4x4(data=target.flatten().tolist()),
                        duration=FloatValue(value=duration),
                        q0=q0,
                    )
                ),
                interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
            )
        else:
            request = GoToRequest(
                cartesian_goal=CartesianGoal(
                    arm_cartesian_goal=ArmCartesianGoal(
                        id=self._part_id,
                        goal_pose=Matrix4x4(data=target.flatten().tolist()),
                        duration=FloatValue(value=duration),
                    )
                ),
                interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
            )
        response = self._goto_stub.GoToCartesian(request)
        if response.id == -1:
            self._logger.error(f"Target pose:\n {target} \nwas not reachable. No command sent.")
        if wait:
            self._logger.info(f"Waiting for movement with {response}.")
            while not self._is_goto_finished(response):
                time.sleep(0.1)
            self._logger.info(f"Movement with {response} finished.")
        return response

    def send_cartesian_interpolation(
        self,
        target: npt.NDArray[np.float64],
        duration: float = 2,
        arc_direction: Optional[str] = None,
        elliptic_radius: Optional[float] = None,
        interpolation_frequency: float = 120,
        precision_distance_xyz: float = 0.003,
    ) -> None:
        """
        The `send_cartesian_interpolation` function performs cartesian interpolation to move the arm
        towards a target pose or get close to it using linear or circular interpolation for translation.

        Args:
          target (npt.NDArray[np.float64]): an homogeneous 4x4 pose matrix expressed in Reachy coordinate system, represented
        as a numpy array. This matrix defines the desired position and orientation that the arm should move to.
          duration (float): The `duration` parameter represents the expected time in seconds for the arm
        to reach the target position from its current position. Defaults to 2
          arc_direction (Optional[str]): The `arc_direction` parameter is used to specify the direction of elliptic
        interpolation when moving the arm towards a target pose. The `arc_direction` parameter can be set to 'above',
        'below', 'right', 'left', 'front', or 'back'. If no `arc_direction` is specified, a linear interpolation is computed.
          elliptic_radius (Optional[float]): The `elliptic_radius` parameter represents the second radius of the computed
        ellipse when performing elliptical interpolation. The first radius is the distance between the present
        pose and the target pose. If no `elliptic_radius` is specified, a circular interpolation is computed.
          interpolation_frequency (float): The `interpolation_frequency` parameter specifies how many
        intermediate points will be used to interpolate the movement in cartesian space between the
        initial and target poses. Defaults to 120
          precision_distance_xyz (float): The `precision_distance_xyz` parameter represents the maximum allowed distance in the
        XYZ space between the current end-effector position and the target position. If the current end-effector position
        is further than this distance from the target position after the interpolation movement, the movement will be repeated
        until the precision is met.

        Returns:
          The `send_cartesian_interpolation` method returns `None`.
        """
        self.cancel_all_goto()
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
            for t in np.linspace(0, 1, nb_steps):
                # Spamming the goal position to make sure its reached
                request = ArmCartesianGoal(
                    id=self._part_id,
                    goal_pose=Matrix4x4(data=target.flatten().tolist()),
                )
                self._stub.SendArmCartesianGoal(request)
                time.sleep(time_step)

            # Small delay to make sure the present position is correctly read
            time.sleep(0.1)
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
        """
        This function generates linear interpolation between two poses over a specified number of steps with a given time step.

        Args:
          origin_trans (npt.NDArray[np.float64]): The `origin_trans` parameter represents the original
        translation vector in 3D space. It is a NumPy array of type `np.float64` containing the
        translation components along the x, y, and z axes.
          target_trans (npt.NDArray[np.float64]): The `target_trans` parameter represents the target translation that you want
        to interpolate towards. It is a numpy array containing the target translation values in float64 format.
          origin_rot (Quaternion): The `origin_rot` parameter is of type `Quaternion`. It represents the initial rotation
        quaternion that is used as the starting point for the interpolation.
          target_rot (Quaternion): The `target_rot` parameter represents the target rotation Quaternion that you want to
        interpolate towards.
          nb_steps (int): The `nb_steps` parameter represents the number of steps or intervals for the interpolation process.
        It determines how many intermediate points will be calculated between the starting point and the ending point for both
        translation and rotation interpolation.
          time_step (float): The `time_step` parameter represents the time interval in seconds between each step of the
        interpolation process.

        Returns:
          The `_send_linear_interpolation` method returns `None`.
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
        """
        The function `_send_elliptical_interpolation` generates elliptical interpolation between two
        poses for the arm.

        Args:
          origin_trans (npt.NDArray[np.float64]): initial translation vector of the arm you want to interpolate from.
        It is a NumPy array of type `np.float64` that contains the x, y, and z.
          target_trans (npt.NDArray[np.float64]): target translation (position) that the robot end-effector should
        reach during the interpolation process. It is a NumPy array of type `np.float64` that contains the x, y, and z.
          origin_rot (Quaternion): initial orientation (rotation) of the end-effector
          target_rot (Quaternion): target orientation (rotation) that the end effector should reach during
        the interpolation process.
          arc_direction (str): The `arc_direction` parameter is used to specify the direction of the elliptical interpolation.
        The `arc_direction` parameter can be set to 'above', 'below', 'right', 'left', 'front', or 'back'.
          secondary_radius (Optional[float]): radius of the secondary axis of the ellipse being interpolated. If this parameter
        is not provided, it defaults to the value of the primary radius (which is calculated based on the distance between the
        origin pose and the target pose).
          nb_steps (int): number of steps for the interpolation process. It determines how many
        intermediate poses will be generated between the starting pose (origin) and the target pose.
          time_step (float): time interval in seconds between each step of the interpolation process.

        Returns:
          The `_send_elliptical_interpolation` method returns `None`.
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

    def goto_joints(
        self,
        positions: List[float],
        duration: float = 2,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
    ) -> GoToId:
        """
        The `goto_joints` function moves the arm's joints to a specified position using a specified
        duration and interpolation mode, with optional waiting for completion.

        Args:
          positions (List[float]): list of float values representing the desired joint positions of the arm.
        It should contain exactly 7 joint positions for the arm , given in the following order:
        [shoulder_pitch, shoulder_roll, elbow_yaw, elbow_pitch, wrist_roll, wrist_pitch, wrist_yaw].
          duration (float): the time duration in seconds for the arm to reach the desired joint positions. Default is 2
          wait (bool): boolean parameter that determines whether the program should wait for the movement to finish before
        returning. If `wait` is set to `True`, the program will wait for the movement to complete before continuing
        execution. Defaults to False
          interpolation_mode (str): type of interpolation to be used when moving the arm's joints to the given
        position. Can be set to 'minimum_jerk' or 'linear'. Defaults to 'minimum_jerk'.
          degrees (bool): boolean flag that specifies whether the joint positions are provided in degrees or not.
        If `degrees` is set to `True`, it indicates that the joint positions in the `positions` list are in degrees.
        Defaults to True

        Returns:
          The function `goto_joints` returns a `GoToId` object.
        """
        if len(positions) != 7:
            raise ValueError(f"positions should be of length 7 (got {len(positions)} instead)!")
        if duration == 0:
            raise ValueError("duration cannot be set to 0.")
        if self.is_off():
            self._logger.warning(f"{self._part_id.name} is off. Goto not sent.")
            return GoToId(id=-1)

        arm_pos = list_to_arm_position(positions, degrees)
        request = GoToRequest(
            joints_goal=JointsGoal(
                arm_joint_goal=ArmJointGoal(id=self._part_id, joints_goal=arm_pos, duration=FloatValue(value=duration))
            ),
            interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
        )
        response = self._goto_stub.GoToJoints(request)
        if response.id == -1:
            self._logger.error(f"Position {positions} was not reachable. No command sent.")
        if wait:
            self._logger.info(f"Waiting for movement with {response}.")
            while not self._is_goto_finished(response):
                time.sleep(0.1)
            self._logger.info(f"Movement with {response} finished.")
        return response

    def get_translation_by(
        self,
        x: float,
        y: float,
        z: float,
        initial_pose: Optional[npt.NDArray[np.float64]] = None,
        frame: str = "robot",
    ) -> npt.NDArray[np.float64]:
        """
        Returns a 4x4 matrix (as a numpy array) representing a pose translated by specified x, y, z values in either the robot
        or gripper coordinate system.

        Args:
          x (float): translation along the x-axis in meters (forwards direction) that you want to apply to the pose matrix.
          y (float): translation along the y-axis in meters (left direction) that you want to apply to the pose matrix.
          z (float): translation along the z-axis in meters (vertical direction) that you want to apply to the pose matrix.
          initial_pose (Optional[npt.NDArray[np.float64]]): a 4x4 matrix representing the initial pose of the end-effector.
        It is expressed as a numpy array of type `np.float64`. If no initial pose is provided, the current pose of the arm is
        used as the initial pose. Defaults to None
          frame (str): coordinate system in which the translation should be performed. It can be one of: "robot" or
        "gripper". Defaults to "robot".

        Returns:
          a 4x4 pose matrix (as a numpy array) expressed in Reachy coordinate system, translated by the specified x, y, z values
        from the initial pose.
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
        """
        Creates a goto to translate the arm's end effector based on the last goto sent or the current position.

        Args:
          x (float): translation along the x-axis in meters (forwards direction) that you want to apply to the pose matrix.
          y (float): translation along the y-axis in meters (left direction) that you want to apply to the pose matrix.
          z (float): translation along the z-axis in meters (vertical direction) that you want to apply to the pose matrix.
          duration (float): time duration in seconds for the translation movement to be completed. Defaults to 2.
          wait (bool): boolean parameter that determines whether the program should wait for the movement to finish before
        returning. If `wait` is set to `True`, the program will wait for the movement to complete before continuing
        execution. Defaults to False
          frame (str): coordinate system in which the translation should be performed. It can be one of: "robot" or
        "gripper". Defaults to "robot".
          interpolation_mode (str): type of interpolation to be used when moving the arm's joints to the given
        position. Can be set to 'minimum_jerk' or 'linear'. Defaults to 'minimum_jerk'.

        Returns:
          a `GoToId` object, which is created using the `goto_from_matrix` method with the translated pose computed in the
        specified frame.
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
        return self.goto_from_matrix(pose, duration=duration, wait=wait, interpolation_mode=interpolation_mode)

    def get_rotation_by(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        initial_pose: Optional[npt.NDArray[np.float64]] = None,
        degrees: bool = True,
        frame: str = "robot",
    ) -> npt.NDArray[np.float64]:
        """
        Calculates a new pose matrix by rotating an initial pose matrix by specified roll,
        pitch, and yaw angles in either the robot or gripper coordinate system.

        Args:
          roll (float): rotation around the x-axis in the Euler angles representation. It is a float value that specifies the
        rotation angle in radians or degrees (based on the `degrees` parameter).
          pitch (float): rotation around the y-axis in the Euler angles representation. It is a float value that specifies the
        rotation angle in radians or degrees (based on the `degrees` parameter).
          yaw (float): rotation around the z-axis in the Euler angles representation. It is a float value that specifies the
        rotation angle in radians or degrees (based on the `degrees` parameter).
          initial_pose (Optional[npt.NDArray[np.float64]]): a 4x4 matrix representing the initial pose of the end-effector.
        It is expressed as a numpy array of type `np.float64`. If no initial pose is provided, the current pose of the arm is
        used as the initial pose. Defaults to None
          degrees (bool): boolean flag that specifies whether the joint positions are provided in degrees or not.
        If `degrees` is set to `True`, it indicates that the joint positions in the `positions` list are in degrees.
        Defaults to True
          frame (str): coordinate system in which the rotation should be performed. It can be one of: "robot" or
        "gripper". Defaults to "robot".

        Returns:
          a 4x4 pose matrix (as a numpy array) expressed in Reachy coordinate system, rotated by the specified roll, pitch, and
        yaw angles from the initial pose, in the specified frame.
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
        """
        Creates a goto to rotate the arm's end effector based on  the specified roll, pitch, and yaw angles
        in either the robot or gripper frame.

        Args:
          roll (float): rotation around the x-axis in the Euler angles representation. It is a float value that specifies the
        rotation angle in radians or degrees (based on the `degrees` parameter).
          pitch (float): rotation around the y-axis in the Euler angles representation. It is a float value that specifies the
        rotation angle in radians or degrees (based on the `degrees` parameter).
          yaw (float): rotation around the z-axis in the Euler angles representation. It is a float value that specifies the
        rotation angle in radians or degrees (based on the `degrees` parameter).
          duration (float): time duration in seconds for the translation movement to be completed. Defaults to 2.
          wait (bool): boolean parameter that determines whether the program should wait for the movement to finish before
        returning. If `wait` is set to `True`, the program will wait for the movement to complete before continuing
        execution. Defaults to False
          degrees (bool): boolean flag that specifies whether the joint positions are provided in degrees or not.
        If `degrees` is set to `True`, it indicates that the joint positions in the `positions` list are in degrees.
        Defaults to True
          frame (str): coordinate system in which the translation should be performed. It can be one of: "robot" or
        "gripper". Defaults to "robot".
          interpolation_mode (str): type of interpolation to be used when moving the arm's joints to the given
        position. Can be set to 'minimum_jerk' or 'linear'. Defaults to 'minimum_jerk'.

        Returns:
          a `GoToId` object, which is created by calling the `goto_from_matrix` method with the rotates pose computed in the
        specified frame.
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
        return self.goto_from_matrix(pose, duration=duration, wait=wait, interpolation_mode=interpolation_mode)

    def _goto_single_joint(
        self,
        arm_joint: int,
        goal_position: float,
        duration: float = 2,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
    ) -> GoToId:
        """
        Moves a single joint of the arm to a specified position with
        optional parameters for duration, interpolation mode, and waiting for completion.

        Args:
          arm_joint (int): specific joint of the arm that you want to move. It is an integer value that
        corresponds to a particular joint in the arm.
          goal_position (float): target position that the specified arm joint should move to.
        It is a float value that can be either in radians or degrees based on the `degrees` parameter.
          duration (float): time duration in seconds in which the joint should move to the specified goal position.
        Defaults to 2.
          wait (bool): boolean parameter that determines whether the program should wait for the movement to finish before
        returning. If `wait` is set to `True`, the program will wait for the movement to complete before continuing
        execution. Defaults to False
          interpolation_mode (str): type of interpolation to be used when moving the arm's joint to the given
        position. Can be set to 'minimum_jerk' or 'linear'. Defaults to 'minimum_jerk'.
          degrees (bool): boolean flag that specifies whether the joint positions are provided in degrees or not.
        If `degrees` is set to `True`, it indicates that the joint positions in the `positions` list are in degrees.
        Defaults to True

        Returns:
          a unique `GoToId` corresponding to this specific goto movement.
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

    def get_joints_positions(self, degrees: bool = True, round: Optional[int] = None) -> List[float]:
        """
        Returns the current joints positions of the arm, either in degrees or radians,
        with an option to round the values.

        Args:
          degrees (bool): boolean flag that specifies whether the joint positions should be returned in degrees or not.
        If `degrees` is set to `True`, the positions will be returned in degrees. If set to `False`, the positions will
        returned in radians. Defaults to True.
          round (Optional[int]): number of decimal places to round the computed joint angles to before returning them.

        Returns:
          a list of float values representing the current joint positions of the arm, given in the following order:
        [shoulder_pitch | shoulder_roll | elbow_yaw | elbow_pitch | wrist_roll | wrist_pitch | wrist_yaw]
        """
        response = self._stub.GetJointPosition(self._part_id)
        positions: List[float] = arm_position_to_list(response, degrees)
        if round is not None:
            positions = np.round(arm_position_to_list(response, degrees), round).tolist()
        return positions

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

    def send_goal_positions(self) -> None:
        """
        Sends goal positions to gripper and actuators if the parts are on.

        Returns:
          Returns `None`.
        """
        if self._gripper is not None:
            self._gripper.send_goal_positions()
        if self.is_off():
            self._logger.warning(f"{self._part_id.name} is off. Command not sent.")
            return
        for actuator in self._actuators.values():
            actuator.send_goal_positions()

    def goto_posture(
        self,
        common_posture: str = "default",
        duration: float = 2,
        wait: bool = False,
        wait_for_goto_end: bool = True,
        interpolation_mode: str = "minimum_jerk",
    ) -> GoToId:
        """
        Sends all joints to standard positions with optional parameters for duration, waiting, and interpolation mode.

        Args:
          common_posture (str): standard positions to which all joints will be sent. It can have values of 'default' or
        'elbow_90'. Defaults to 'default'.
          duration (float): time duration in seconds in which the robot should move to the specified posture. Defaults to 2.
          wait (bool): boolean parameter that determines whether the program should wait for the movement to finish before
        returning. If `wait` is set to `True`, the program will wait for the movement to complete before continuing
        execution. Defaults to False
          wait_for_goto_end (bool): boolean parameter that determines whether the commands will be sent to a part immediately or
        only after all previous commands in the queue have been executed. If `wait_for_goto_end` is set to `False`, the program
        will cancel all executing moves and queues. Defaults to True.
          interpolation_mode (str): type of interpolation to be used when moving the arm's joint to the given
        position. Can be set to 'minimum_jerk' or 'linear'. Defaults to 'minimum_jerk'.

        Returns:
          A unique `GoToId`.
        """

        """Send all joints to standard positions in specified duration.

        common_posture can be 'default' or 'elbow_90'.
        Setting wait_for_goto_end to False will cancel all gotos on all parts and immediately send the commands.
        Otherwise, the commands will be sent to a part when all gotos of its queue has been played.
        """
        joints = self.get_default_posture_joints(common_posture=common_posture)
        if common_posture == "default":
            if self._gripper is not None and self._gripper.is_on():
                self._gripper.open()
        if not wait_for_goto_end:
            self.cancel_all_goto()
        if self.is_on():
            return self.goto_joints(joints, duration, wait, interpolation_mode)
        else:
            self._logger.warning(f"{self._part_id.name} is off. No command sent.")
        return GoToId(id=-1)

    def _update_with(self, new_state: ArmState) -> None:
        """
        Updates the arm with a newly received (partial) state from the gRPC server by
        updating the shoulder, elbow, and wrist states accordingly.

        Args:
          new_state (ArmState): current state of the arm, including the states of the shoulder, elbow, and wrist.
        """
        self.shoulder._update_with(new_state.shoulder_state)
        self.elbow._update_with(new_state.elbow_state)
        self.wrist._update_with(new_state.wrist_state)

    def _update_audit_status(self, new_status: ArmStatus) -> None:
        """
        Updates the audit status of different components based on a new overall status.

        Args:
          new_status (ArmStatus): new status of the shoulder, elbow, and  wrist.
        """
        self.shoulder._update_audit_status(new_status.shoulder_status)
        self.elbow._update_audit_status(new_status.elbow_status)
        self.wrist._update_audit_status(new_status.wrist_status)
