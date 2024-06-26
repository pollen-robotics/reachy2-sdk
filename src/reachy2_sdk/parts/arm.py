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
    ArmState,
    SpeedLimitRequest,
    TorqueLimitRequest,
)
from reachy2_sdk_api.arm_pb2_grpc import ArmServiceStub
from reachy2_sdk_api.goto_pb2 import (
    CartesianGoal,
    GoToAck,
    GoToId,
    GoToRequest,
    JointsGoal,
)
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.hand_pb2 import Hand as HandState
from reachy2_sdk_api.hand_pb2 import Hand as Hand_proto
from reachy2_sdk_api.kinematics_pb2 import Matrix4x4
from reachy2_sdk_api.part_pb2 import PartId

from ..orbita.orbita2d import Orbita2d
from ..orbita.orbita3d import Orbita3d
from ..orbita.orbita_joint import OrbitaJoint
from ..utils.custom_dict import CustomDict
from ..utils.utils import (
    arm_position_to_list,
    decompose_matrix,
    get_grpc_interpolation_mode,
    list_to_arm_position,
    recompose_matrix,
)
from .hand import Hand


class Arm:
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
        self._grpc_channel = grpc_channel
        self._arm_stub = ArmServiceStub(grpc_channel)
        self._goto_stub = goto_stub
        self._part_id = PartId(id=arm_msg.part_id.id, name=arm_msg.part_id.name)

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
        )
        self._elbow = Orbita2d(
            uid=description.elbow.id.id,
            name=description.elbow.id.name,
            axis1=description.elbow.axis_1,
            axis2=description.elbow.axis_2,
            initial_state=initial_state.elbow_state,
            grpc_channel=self._grpc_channel,
        )
        self._wrist = Orbita3d(
            uid=description.wrist.id.id,
            name=description.wrist.id.name,
            initial_state=initial_state.wrist_state,
            grpc_channel=self._grpc_channel,
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

    @property
    def joints(self) -> CustomDict[str, OrbitaJoint]:
        """Get all the arm's joints."""
        _joints: CustomDict[str, OrbitaJoint] = CustomDict({})
        for actuator_name, actuator in self._actuators.items():
            for joint in actuator._joints.values():
                _joints[actuator_name + "." + joint._axis_type] = joint
        return _joints

    def turn_on(self) -> None:
        """Turn all motors of the part on.

        All arm's motors will then be stiff.
        """
        self._arm_stub.TurnOn(self._part_id)
        if self._gripper is not None:
            self._gripper.turn_on()

    def turn_off(self) -> None:
        """Turn all motors of the part off.

        All arm's motors will then be compliant.
        """
        self._arm_stub.TurnOff(self._part_id)
        if self._gripper is not None:
            self._gripper.turn_off()

    def turn_off_smoothly(self, duration: float = 2) -> None:
        """Turn all motors of the part off.

        All arm's motors will see their torque limit reduces from a determined duration, then will be fully compliant.
        """
        self.set_torque_limit(20)
        time.sleep(duration)
        self._arm_stub.TurnOff(self._part_id)
        if self._gripper is not None:
            self._gripper.turn_off()
        time.sleep(0.2)
        self.set_torque_limit(100)

    def set_torque_limit(self, value: int) -> None:
        """Choose percentage of torque max value applied as limit to the arms."""
        req = TorqueLimitRequest(
            id=self._part_id,
            limit=value,
        )
        self._arm_stub.SetTorqueLimit(req)

    def set_speed_limit(self, value: int) -> None:
        """Choose percentage of speed max value applied as limit to the arms."""
        req = SpeedLimitRequest(
            id=self._part_id,
            limit=value,
        )
        self._arm_stub.SetSpeedLimit(req)

    def is_on(self) -> bool:
        """Return True if all actuators of the arm are stiff"""
        for actuator in self._actuators.values():
            if not actuator.is_on():
                return False
        if self._gripper is not None and not self._gripper.is_on():
            return False
        return True

    def is_off(self) -> bool:
        """Return True if all actuators of the arm are stiff"""
        for actuator in self._actuators.values():
            if actuator.is_on():
                return False
        if self._gripper is not None and self._gripper.is_on():
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
        resp = self._arm_stub.ComputeArmFK(req)
        if not resp.success:
            raise ValueError(f"No solution found for the given joints ({joints_positions})!")

        return np.array(resp.end_effector.pose.data).reshape((4, 4))

    def inverse_kinematics(
        self,
        target: npt.NDArray[np.float64],
        q0: Optional[List[float]] = None,
        degrees: bool = True,
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
        resp = self._arm_stub.ComputeArmIK(req)

        if not resp.success:
            raise ValueError(f"No solution found for the given target ({target})!")

        return arm_position_to_list(resp.arm_position)

    def goto_from_matrix(
        self,
        target: npt.NDArray[np.float64],
        duration: float = 2,
        interpolation_mode: str = "minimum_jerk",
        q0: Optional[List[float]] = None,
        with_cartesian_interpolation: bool = False,
        interpolation_frequency: float = 120,
    ) -> GoToId:
        """Move the arm to a matrix target (or get close).

        Given a pose 4x4 target matrix (as a numpy array) expressed in Reachy coordinate systems,
        it will try to compute a joint solution to reach this target (or get close),
        and move to this position in the defined duration.

        If with_cartesian_interpolation is set to True, it will interpolate the movement in cartesian space
        and send cartesian commands at the interpolation frequency (default value is 10hz).
        """
        if target.shape != (4, 4):
            raise ValueError("target shape should be (4, 4) (got {target.shape} instead)!")
        if q0 is not None and (len(q0) != 7):
            raise ValueError(f"q0 should be length 7 (got {len(q0)} instead)!")
        if duration == 0:
            raise ValueError("duration cannot be set to 0.")
        if self.is_off():
            raise RuntimeError("Arm is off. Goto not sent.")

        if with_cartesian_interpolation:
            return self._goto_cartesian_interpolation(target, duration, interpolation_frequency)

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
        return response

    def _goto_cartesian_interpolation(
        self,
        target: npt.NDArray[np.float64],
        duration: float = 2,
        interpolation_frequency: float = 120,
        precision_distance_xyz: float = 0.003,
    ) -> GoToId:
        """Move the arm to a matrix target (or get close).

        Given a pose 4x4 target matrix (as a numpy array) expressed in Reachy coordinate systems,
        it will try to compute a joint solution to reach this target (or get close).
        It will interpolate the movement in cartesian space in a number of intermediate points defined
        by the interpolation frequency and the duration.
        """
        if target.shape != (4, 4):
            raise ValueError("target shape should be (4, 4) (got {target.shape} instead)!")
        if duration == 0:
            raise ValueError("duration cannot be set to 0.")
        if self.is_off():
            raise RuntimeError("Arm is off. Goto not sent.")
        try:
            self.inverse_kinematics(target)
        except ValueError:
            print("Goal pose is not reachable!")
            return GoToId(id=-1)

        origin_matrix = self.forward_kinematics()
        nb_steps = int(duration * interpolation_frequency)
        time_step = duration / nb_steps

        q1, trans1 = decompose_matrix(origin_matrix)
        q2, trans2 = decompose_matrix(target)

        for t in np.linspace(0, 1, nb_steps):
            # Linear interpolation for translation
            trans_interpolated = (1 - t) * trans1 + t * trans2

            # SLERP for rotation interpolation
            q_interpolated = Quaternion.slerp(q1, q2, t)
            rot_interpolated = q_interpolated.rotation_matrix

            # Recompose the interpolated matrix
            interpolated_matrix = recompose_matrix(rot_interpolated, trans_interpolated)

            request = ArmCartesianGoal(
                id=self._part_id,
                goal_pose=Matrix4x4(data=interpolated_matrix.flatten().tolist()),
            )
            self._arm_stub.SendArmCartesianGoal(request)
            time.sleep(time_step)

        current_pose = self.forward_kinematics()
        current_precision_distance_xyz = np.linalg.norm(current_pose[:3, 3] - target[:3, 3])
        if current_precision_distance_xyz > precision_distance_xyz:
            print("Precision is not good enough, spamming the goal position!")
            for t in np.linspace(0, 1, nb_steps):
                # Spamming the goal position to make sure its reached
                request = ArmCartesianGoal(
                    id=self._part_id,
                    goal_pose=Matrix4x4(data=target.flatten().tolist()),
                )
                self._arm_stub.SendArmCartesianGoal(request)
                time.sleep(time_step)

            # Small delay to make sure the present position is correctly read
            time.sleep(0.1)
            current_pose = self.forward_kinematics()
            current_precision_distance_xyz = np.linalg.norm(current_pose[:3, 3] - target[:3, 3])
            print(f"l2 xyz distance to goal: {current_precision_distance_xyz}")

        return GoToId(id=0)

    def goto_joints(
        self, positions: List[float], duration: float = 2, interpolation_mode: str = "minimum_jerk", degrees: bool = True
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
            raise RuntimeError("Arm is off. Goto not sent.")

        arm_pos = list_to_arm_position(positions, degrees)
        request = GoToRequest(
            joints_goal=JointsGoal(
                arm_joint_goal=ArmJointGoal(id=self._part_id, joints_goal=arm_pos, duration=FloatValue(value=duration))
            ),
            interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
        )
        response = self._goto_stub.GoToJoints(request)
        return response

    def get_move_playing(self) -> GoToId:
        """Return the id of the goto currently playing on the arm"""
        response = self._goto_stub.GetPartGoToPlaying(self._part_id)
        return response

    def get_moves_queue(self) -> List[GoToId]:
        """Return the list of all goto ids waiting to be played on the arm"""
        response = self._goto_stub.GetPartGoToQueue(self._part_id)
        return [goal_id for goal_id in response.goto_ids]

    def cancel_all_moves(self) -> GoToAck:
        """Ask the cancellation of all waiting goto on the arm"""
        response = self._goto_stub.CancelPartAllGoTo(self._part_id)
        return response

    def get_joints_positions(self) -> List[float]:
        """Return the current joints positions of the arm in degrees"""
        response = self._arm_stub.GetJointPosition(self._part_id)
        positions = arm_position_to_list(response)
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

    def _update_with(self, new_state: ArmState) -> None:
        """Update the arm with a newly received (partial) state received from the gRPC server."""
        self.shoulder._update_with(new_state.shoulder_state)
        self.elbow._update_with(new_state.elbow_state)
        self.wrist._update_with(new_state.wrist_state)
