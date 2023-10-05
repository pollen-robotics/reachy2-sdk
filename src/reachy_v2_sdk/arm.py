from typing import List, Optional

import grpc

import numpy as np
import numpy.typing as npt

from reachy_sdk_api_v2.arm_pb2_grpc import ArmServiceStub
from reachy_sdk_api_v2.arm_pb2 import Arm as Arm_proto, ArmPosition
from reachy_sdk_api_v2.arm_pb2 import ArmJointGoal, ArmState
from reachy_sdk_api_v2.arm_pb2 import ArmLimits, ArmTemperatures
from reachy_sdk_api_v2.arm_pb2 import ArmFKRequest, ArmIKRequest, ArmEndEffector
from reachy_sdk_api_v2.part_pb2 import PartId
from reachy_sdk_api_v2.kinematics_pb2 import Matrix4x4

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
            name=description.shoulder.id.id,
            axis1=description.shoulder.axis_1,
            axis2=description.shoulder.axis_2,
            initial_state=initial_state.shoulder_state,
            grpc_channel=self._grpc_channel,
        )
        self.elbow = Orbita2d(
            name=description.elbow.id.id,
            axis1=description.elbow.axis_1,
            axis2=description.elbow.axis_2,
            initial_state=initial_state.elbow_state,
            grpc_channel=self._grpc_channel,
        )
        self.wrist = Orbita3d(
            name=description.wrist.id.id,
            initial_state=initial_state.wrist_state,
            grpc_channel=self._grpc_channel,
        )

    def turn_on(self) -> None:
        self._arm_stub.TurnOn(self.part_id)

    def turn_off(self) -> None:
        self._arm_stub.TurnOff(self.part_id)

    def forward_kinematics(self, joints_positions: Optional[List[float]] = None) -> npt.NDArray[np.float64]:
        req = ArmFKRequest(id=self.part_id)
        if joints_positions is not None:
            if len(joints_positions) != 7:
                raise ValueError(f"joints_positions should be length 7 (got {len(joints_positions)} instead)!")
            req.position = self._list_to_arm_position(joints_positions)
        resp = self._arm_stub.ComputeArmFK(req)
        if not resp.success:
            raise ValueError(f"No solution found for the given joints ({joints_positions})!")

        return np.array(resp.end_effector.pose.data).reshape((4, 4))

    def inverse_kinematics(self, target: npt.NDArray[np.float64], q0: Optional[List[float]] = None) -> List[float]:
        if target.shape != (4, 4):
            raise ValueError("target shape should be (4, 4) (got {target.shape} instead)!")

        if q0 is not None and (len(q0) != 7):
            raise ValueError(f"q0 should be length 7 (got {len(q0)} instead)!")

        if isinstance(q0, np.ndarray) and len(q0.shape) > 1:
            raise ValueError("Vectorized kinematics not supported!")

        req_params = {
            "target": ArmEndEffector(
                pose=Matrix4x4(data=target.flatten().tolist()),
            )
        }

        if q0 is not None:
            req_params["q0"] = self._list_to_arm_position(q0)

        req = ArmIKRequest(**req_params)
        resp = self._arm_stub.ComputeArmIK(req)

        if not resp.success:
            raise ValueError(f"No solution found for the given target ({target})!")

        return self._arm_position_to_list(resp.arm_position)

    def _list_to_arm_position(self, positions: List[float]) -> ArmPosition:
        arm_pos = ArmPosition()

        joints = [field.name for field in ArmPosition.DESCRIPTOR.fields]
        for joint, position in zip(joints, positions):
            setattr(arm_pos, joint, position)

        return arm_pos

    def _arm_position_to_list(self, arm_pos: ArmPosition) -> List[float]:
        positions = []
        for _, value in arm_pos.ListFields():
            positions.append(value)

        return positions

    # def goto_point(self, position: List[float], orientation: List[float],
    #                position_tol: List[float], orientation_tol: List[float], duration: float) -> None:
    #     goal = ArmCartesianGoal(duration=duration)

    def goto_joints(self, positions: List[float], duration: float) -> None:
        arm_pos = self._list_to_arm_position(positions)
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
