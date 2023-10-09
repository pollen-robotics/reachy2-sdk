from typing import List

import grpc

from reachy_sdk_api_v2.arm_pb2_grpc import ArmServiceStub
from reachy_sdk_api_v2.arm_pb2 import Arm as Arm_proto, ArmPosition
from reachy_sdk_api_v2.arm_pb2 import ArmJointGoal, ArmState
from reachy_sdk_api_v2.arm_pb2 import ArmLimits, ArmTemperatures
from reachy_sdk_api_v2.part_pb2 import PartId

from .orbita2d import Orbita2d
from .orbita3d import Orbita3d


class Arm:
    def __init__(self, arm_msg: Arm_proto, initial_state: ArmState, grpc_channel: grpc.Channel) -> None:
        self._grpc_channel = grpc_channel
        self._arm_stub = ArmServiceStub(grpc_channel)
        self.part_id = PartId(id=arm_msg.part_id.id, name=arm_msg.part_id.name)

        self._setup_arm(arm_msg, initial_state)

    def _setup_arm(self, arm: Arm_proto, initial_state: ArmState) -> None:
        description = arm.description
        self.shoulder = Orbita2d(
            uid=description.shoulder.id.id,
            name=description.shoulder.id.name,
            axis1=description.shoulder.axis_1,
            axis2=description.shoulder.axis_2,
            initial_state=initial_state.shoulder_state,
            grpc_channel=self._grpc_channel,
        )
        self.elbow = Orbita2d(
            uid=description.elbow.id.id,
            name=description.elbow.id.name,
            axis1=description.elbow.axis_1,
            axis2=description.elbow.axis_2,
            initial_state=initial_state.elbow_state,
            grpc_channel=self._grpc_channel,
        )
        self.wrist = Orbita3d(
            uid=description.wrist.id.id,
            name=description.wrist.id.name,
            initial_state=initial_state.wrist_state,
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
    def joints_limits(self) -> ArmLimits:
        limits = self._arm_stub.GetJointLimit(self.part_id)
        return limits

    @property
    def temperatures(self) -> ArmTemperatures:
        temperatures = self._arm_stub.GetTemperatures(self.part_id)
        return temperatures

    def _update_with(self, new_state: ArmState) -> None:
        """Update the arm with a newly received (partial) state received from the gRPC server."""
        self.shoulder._update_with(new_state.shoulder_state)
        self.elbow._update_with(new_state.elbow_state)
        self.wrist._update_with(new_state.wrist_state)
