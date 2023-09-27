"""Reachy Arm module.

Handles all specific method to an Arm (left and/or right) especially:
- the forward kinematics
- the inverse kinematics
"""

from typing import List, Optional, Set

from reachy_sdk_api_v2.arm_pb2_grpc import ArmStub
from reachy_sdk_api_v2.arm_pb2 import Arm, ArmPosition
from reachy_sdk_api_v2.arm_pb2 import ArmJointGoal, ArmCartesianGoal
from reachy_sdk_api_v2.arm_pb2 import SpeedLimitRequest
from reachy_sdk_api_v2.part_pb2 import PartId

import numpy as np


class Arm():
    """Arm abstract class used for both left/right arms.

    It exposes the kinematics of the arm:
    - you can access the joints actually used in the kinematic chain,
    - you can compute the forward and inverse kinematics
    """

    def __init__(self, arm: Arm, grpc_channel) -> None:
        """Set up the arm with its kinematics."""
        self._arm_stub = ArmStub(grpc_channel)
        self.part_id = PartId(id=arm.part_id)

        self._joint_list = ['shoulder_pitch', 'shoulder_roll',
                            'elbow_yaw', 'elbow_pitch',
                            'wrist_roll', 'wrist_pitch', 'wrist_yaw']

    def turn_on(self):
        self._arm_stub.TurnOn(self.part_id)

    def turn_off(self):
        self._arm_stub.TurnOff(self.part_id)
    
    def goto_point(self, position, orientation, position_tol, orientation_tol, duration):
        goal = ArmCartesianGoal(duration=duration)
    
    def goto_joints(self, joints: List[float], duration: float):
        arm_pos = ArmPosition()
        arm_pos.shoulder_pitch = joints[0]
        arm_pos.shoulder_roll = joints[1]
        arm_pos.elbow_yaw = joints[2]
        arm_pos.elbow_pitch = joints[3]
        arm_pos.wrist_roll = joints[4]
        arm_pos.wrist_pitch = joints[5]
        arm_pos.wrist_yaw = joints[6]
        goal = ArmJointGoal(id=self.part_id,
                            position=arm_pos,
                            duration=duration)
        self._arm_stub.GoToJointPosition(goal)
    
    @property
    def joints_limits(self):
        limits = self._arm_stub.GetJointLimit(self.part_id)
        return limits

    @property
    def temperatures(self):
        temperatures = self._arm_stub.GetTemperatures(self.part_id)
        return temperatures

    @property
    def speed_limit(self):
        limits = self._arm_stub.GetArmState(self.part_id).speed_limit
        return limits

    @speed_limit.setter
    def speed_limit(self, value):
        req = SpeedLimitRequest(id=self.part_id, limit=value)
        self._arm_stub.SetSpeedLimit(req)
