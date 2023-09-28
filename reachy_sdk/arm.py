"""Reachy Arm module.

Handles all specific method to an Arm (left and/or right) especially:
- the forward kinematics
- the inverse kinematics
"""

from typing import List

import grpc

from reachy_sdk_api_v2.arm_pb2_grpc import ArmStub
from reachy_sdk_api_v2.arm_pb2 import Arm as Arm_proto, ArmPosition
from reachy_sdk_api_v2.arm_pb2 import ArmJointGoal
from reachy_sdk_api_v2.arm_pb2 import JointsLimits, ArmTemperatures
from reachy_sdk_api_v2.part_pb2 import PartId


class Arm:
    """Arm abstract class used for both left/right arms.

    It exposes the kinematics of the arm:
    - you can access the joints actually used in the kinematic chain,
    - you can compute the forward and inverse kinematics
    """

    def __init__(self, arm: Arm_proto, grpc_channel: grpc.Channel) -> None:
        """Set up the arm with its kinematics."""
        self._arm_stub = ArmStub(grpc_channel)
        self.part_id = PartId(id=arm.part_id)

        self._joint_list = [
            "shoulder_pitch",
            "shoulder_roll",
            "elbow_yaw",
            "elbow_pitch",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        ]

    def turn_on(self) -> None:
        self._arm_stub.TurnOn(self.part_id)

    def turn_off(self) -> None:
        self._arm_stub.TurnOff(self.part_id)

    # def goto_point(self, position: List[float], orientation: List[float],
    #                position_tol: List[float], orientation_tol: List[float], duration: float) -> None:
    #     goal = ArmCartesianGoal(duration=duration)

    def goto_joints(self, joints: List[float], duration: float) -> None:
        arm_pos = ArmPosition()
        arm_pos.shoulder_pitch = joints[0]
        arm_pos.shoulder_roll = joints[1]
        arm_pos.elbow_yaw = joints[2]
        arm_pos.elbow_pitch = joints[3]
        arm_pos.wrist_roll = joints[4]
        arm_pos.wrist_pitch = joints[5]
        arm_pos.wrist_yaw = joints[6]
        goal = ArmJointGoal(id=self.part_id, position=arm_pos, duration=duration)
        self._arm_stub.GoToJointPosition(goal)

    @property
    def joints_limits(self) -> JointsLimits:
        limits = self._arm_stub.GetJointLimit(self.part_id)
        return limits

    @property
    def temperatures(self) -> ArmTemperatures:
        temperatures = self._arm_stub.GetTemperatures(self.part_id)
        return temperatures
