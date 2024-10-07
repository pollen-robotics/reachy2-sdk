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
        """Set up the arm.

        Set up the arm's actuators (shoulder, elbow and wrist) with the arm's description and initial state.
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
        return self._shoulder

    @property
    def elbow(self) -> Orbita2d:
        return self._elbow

    @property
    def wrist(self) -> Orbita3d:
        return self._wrist

    @property
    def gripper(self) -> Optional[Hand]:
        return self._gripper

    def turn_on(self) -> None:
        """Turn all motors of the part on.

        All arm's motors will then be stiff.
        """
        if self._gripper is not None:
            self._gripper._turn_on()
        super().turn_on()

    def turn_off(self) -> None:
        """Turn all motors of the part off.

        All arm's motors will then be compliant.
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
        """Turn all motors of the part off.

        All arm's motors will see their torque limit reduces for 3 seconds, then will be fully compliant.
        """
        torque_limit_low = 35
        torque_limit_high = 100
        duration = 3

        self.set_torque_limits(torque_limit_low)
        self.set_pose(duration=duration, wait_for_moves_end=False)

        countingTime = 0
        while countingTime < duration:
            time.sleep(1)
            torque_limit_low -= 10
            self.set_torque_limits(torque_limit_low)
            countingTime += 1

        super().turn_off()
        self.set_torque_limits(torque_limit_high)

    def is_on(self) -> bool:
        """Return True if all actuators of the arm are stiff"""
        if not super().is_on():
            return False
        return True

    def is_off(self) -> bool:
        """Return True if all actuators of the arm are stiff"""
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
        """Compute the forward kinematics of the arm.

        It will return the pose 4x4 matrix (as a numpy array) expressed in Reachy coordinate systems.
        You can either specify a given joints position, otherwise it will use the current robot position.
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
        """Compute the inverse kinematics of the arm.

        Given a pose 4x4 target matrix (as a numpy array) expressed in Reachy coordinate systems,
        it will try to compute a joint solution to reach this target (or get close).

        It will raise a ValueError if no solution is found.

        You can also specify a basic joint configuration as a prior for the solution.
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

    def get_default_pose_joints(self, common_pose: str = "default") -> List[float]:
        """Return the list of the joints positions for the default poses."""
        if common_pose not in ["default", "elbow_90"]:
            raise ValueError(f"common_pose {common_pose} not supported! Should be 'default' or 'elbow_90'")
        if common_pose == "elbow_90":
            elbow_pitch = -90
        else:
            elbow_pitch = 0
        if self._part_id.name == "r_arm":
            return [0, -15, -15, elbow_pitch, 0, 0, 0]
        else:
            return [0, 15, 15, elbow_pitch, 0, 0, 0]

    def get_default_pose_matrix(self, common_pose: str = "default") -> npt.NDArray[np.float64]:
        """Return the 4x4 pose matrix of default robot poses."""
        joints = self.get_default_pose_joints(common_pose)
        return self.forward_kinematics(joints)

    def goto_from_matrix(
        self,
        target: npt.NDArray[np.float64],
        duration: float = 2,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        q0: Optional[List[float]] = None,
    ) -> GoToId:
        """Move the arm to a matrix target (or get close).

        Given a pose 4x4 target matrix (as a numpy array) expressed in Reachy coordinate systems,
        it will try to compute a joint solution to reach this target (or get close),
        and move to this position in the defined duration.
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
            while not self._is_move_finished(response):
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
        """Move the arm to a matrix target (or get close).

        Given a pose 4x4 target matrix (as a numpy array) expressed in Reachy coordinate systems,
        it will try to compute a joint solution to reach this target (or get close).
        It will interpolate the movement in cartesian space in a number of intermediate points defined
        by the interpolation frequency and the duration.

        Arguments :
            target          : the 4x4 goal pose matrix in the robot coordinate system, reached at the end of the interpolation
            duration        : the target duration of the movement
            arc_direction   : if None, leads to a linear interpolation. If set, direction of the elliptic interpolation.
                            Can be set to 'above', 'below', 'right', 'left', 'front' or 'back'
            elliptic_radius : the second radius of the computed ellipse, first radius being the present to target pose distance.
                            If None, leads to a circular interpolation.
            interpolation_frequency : the frequency of the interpolation
            precision_distance_xyz  : the maximum distance between the target pose and the reached pose after the interpolation.
                                      Precision is prioritized over duration.
        """

        self.cancel_all_moves()
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
        """Generate linear interpolation."""
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
        """Generate elliptical interpolation."""
        vector_target_origin = target_trans - origin_trans

        center = (origin_trans + target_trans) / 2
        radius = float(np.linalg.norm(vector_target_origin) / 2)

        vector_origin_center = origin_trans - center
        vector_target_center = target_trans - center

        if radius == 0:
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
        """Move the arm's joints to reach the given position.

        Given a list of joint positions (exactly 7 joint positions),
        it will move the arm to that position.
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
            while not self._is_move_finished(response):
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
        """Get a pose 4x4 matrix (as a numpy array) expressed in Reachy coordinate system, translated by x, y, z (in meters)
        from the initial pose.
        If no initial_pose has been sent, use the current pose.

        Two frames can be used:
        - robot frame : translation is done in Reachy's coordinate system
        - gripper frame : translation is done in the gripper's coordinate system
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
        """Create a goto to translate the arm's end effector from the last move sent on the part.
        If no move has been sent, use the current position.

        Two frames can be used:
        - robot frame : translation is done in Reachy's coordinate system
        - gripper frame : translation is done in the gripper's coordinate system
        """
        try:
            move = self.get_moves_queue()[-1]
        except IndexError:
            move = self.get_move_playing()

        if move.id != -1:
            joints_request = self._get_move_joints_request(move)
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
        """Get a pose 4x4 matrix (as a numpy array) expressed in Reachy coordinate system, rotated by roll, pitch, yaw
        from the initial pose.

        Two frames can be used:
        - robot frame : rotation is done around Reachy's coordinate system axis
        - gripper frame : rotation is done in the gripper's coordinate system
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
        """Create a goto to rotate the arm's end effector from the last move sent on the part.
        If no move has been sent, use the current position.

        Two frames can be used:
        - robot frame : rotation is done around Reachy's coordinate system axis
        - gripper frame : rotation is done in the gripper's coordinate system
        """
        if frame not in ["robot", "gripper"]:
            raise ValueError(f"Unknown frame {frame}! Should be 'robot' or 'gripper'")

        try:
            move = self.get_moves_queue()[-1]
        except IndexError:
            move = self.get_move_playing()

        if move.id != -1:
            joints_request = self._get_move_joints_request(move)
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
            while not self._is_move_finished(response):
                time.sleep(0.1)
            self._logger.info(f"Movement with {response} finished.")
        return response

    def get_joints_positions(self, degrees: bool = True, round: Optional[int] = None) -> List[float]:
        """Return the current joints positions of the arm, by default in degrees"""
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
        if self._gripper is not None:
            self._gripper.send_goal_positions()
        if self.is_off():
            self._logger.warning(f"{self._part_id.name} is off. Command not sent.")
            return
        for actuator in self._actuators.values():
            actuator.send_goal_positions()

    def set_pose(
        self,
        common_pose: str = "default",
        duration: float = 2,
        wait: bool = False,
        wait_for_moves_end: bool = True,
        interpolation_mode: str = "minimum_jerk",
    ) -> GoToId:
        """Send all joints to standard positions in specified duration.

        common_pose can be 'default', arms being straight, or 'elbow_90'.
        Setting wait_for_goto_end to False will cancel all gotos on all parts and immediately send the commands.
        Otherwise, the commands will be sent to a part when all gotos of its queue has been played.
        """
        joints = self.get_default_pose_joints(common_pose=common_pose)
        if common_pose == "default":
            if self._gripper is not None and self._gripper.is_on():
                self._gripper.open()
        if not wait_for_moves_end:
            self.cancel_all_moves()
        if self.is_on():
            return self.goto_joints(joints, duration, wait, interpolation_mode)
        else:
            self._logger.warning(f"{self._part_id.name} is off. No command sent.")
        return GoToId(id=-1)

    def _update_with(self, new_state: ArmState) -> None:
        """Update the arm with a newly received (partial) state received from the gRPC server."""
        self.shoulder._update_with(new_state.shoulder_state)
        self.elbow._update_with(new_state.elbow_state)
        self.wrist._update_with(new_state.wrist_state)

    def _update_audit_status(self, new_status: ArmStatus) -> None:
        self.shoulder._update_audit_status(new_status.shoulder_status)
        self.elbow._update_audit_status(new_status.elbow_status)
        self.wrist._update_audit_status(new_status.wrist_status)
