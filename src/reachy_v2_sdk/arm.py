from typing import List

import grpc

from reachy_sdk_api_v2.arm_pb2_grpc import ArmServiceStub
from reachy_sdk_api_v2.arm_pb2 import Arm as Arm_proto, ArmPosition
from reachy_sdk_api_v2.arm_pb2 import ArmJointGoal
from reachy_sdk_api_v2.arm_pb2 import JointLimits, ArmTemperatures
from reachy_sdk_api_v2.part_pb2 import PartId

from .orbita2d import Orbita2d
from .orbita3d import Orbita3d


class Arm:
    def __init__(self, arm_msg: Arm_proto, grpc_channel: grpc.Channel) -> None:
        self._grpc_channel = grpc_channel
        self._arm_stub = ArmServiceStub(grpc_channel)
        self.part_id = PartId(id=arm_msg.part_id.id)

        self._setup_arm(arm_msg)
        self._actuators = {
            self.shoulder: "orbita2d",
            self.elbow: "orbita2d",
            self.wrist: "orbita3d",
        }

    def _setup_arm(self, arm: Arm_proto) -> None:
        description = arm.description
        self.shoulder = Orbita2d(
            name=description.shoulder.id.id,
            axis1=description.shoulder.axis_1,
            axis2=description.shoulder.axis_2,
            grpc_channel=self._grpc_channel,
        )
        self.elbow = Orbita2d(
            name=description.elbow.id.id,
            axis1=description.elbow.axis_1,
            axis2=description.elbow.axis_2,
            grpc_channel=self._grpc_channel,
        )
        self.wrist = Orbita3d(
            name=description.wrist.id.id,
            grpc_channel=self._grpc_channel,
        )

    def turn_on(self) -> None:
        self._arm_stub.TurnOn(self.part_id)

    def turn_off(self) -> None:
        self._arm_stub.TurnOff(self.part_id)

    # def goto_point(self, position: List[float], orientation: List[float],
    #                position_tol: List[float], orientation_tol: List[float], duration: float) -> None:
    #     goal = ArmCartesianGoal(duration=duration)

    def goto_joints(self, positions: List[float], duration: float) -> None:
        arm_pos = ArmPosition()

        joints = [field.name for field in ArmPosition.DESCRIPTOR.fields]
        for joint, position in zip(joints, positions):
            setattr(arm_pos, joint, position)

        goal = ArmJointGoal(id=self.part_id, position=arm_pos, duration=duration)
        self._arm_stub.GoToJointPosition(goal)

    @property
    def joints_limits(self) -> JointLimits:
        limits = self._arm_stub.GetJointLimit(self.part_id)
        return limits

    @property
    def temperatures(self) -> ArmTemperatures:
        temperatures = self._arm_stub.GetTemperatures(self.part_id)
        return temperatures
