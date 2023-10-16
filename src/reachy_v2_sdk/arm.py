from typing import List, Optional, Tuple

import grpc

import numpy as np
import numpy.typing as npt

from pyquaternion import Quaternion as pyQuat

from google.protobuf.wrappers_pb2 import FloatValue

from reachy_sdk_api_v2.arm_pb2_grpc import ArmServiceStub
from reachy_sdk_api_v2.arm_pb2 import Arm as Arm_proto, ArmPosition
from reachy_sdk_api_v2.arm_pb2 import ArmJointGoal, ArmState, ArmCartesianGoal
from reachy_sdk_api_v2.arm_pb2 import ArmLimits, ArmTemperatures
from reachy_sdk_api_v2.arm_pb2 import ArmFKRequest, ArmIKRequest, ArmEndEffector
from reachy_sdk_api_v2.orbita2d_pb2 import Pose2D
from reachy_sdk_api_v2.part_pb2 import PartId
from reachy_sdk_api_v2.kinematics_pb2 import Matrix4x4, Point, Rotation3D, ExtEulerAngles, Matrix3x3, Quaternion
from reachy_sdk_api_v2.kinematics_pb2 import PointDistanceTolerances, ExtEulerAnglesTolerances

from .orbita2d import Orbita2d
from .orbita3d import Orbita3d


class Arm:
    def __init__(self, arm_msg: Arm_proto, initial_state: ArmState, grpc_channel: grpc.Channel) -> None:
        self._grpc_channel = grpc_channel
        self._arm_stub = ArmServiceStub(grpc_channel)
        self.part_id = PartId(id=arm_msg.part_id.id, name=arm_msg.part_id.name)

        self._setup_arm(arm_msg, initial_state)
        self._actuators = {
            self.shoulder: "orbita2d",
            self.elbow: "orbita2d",
            self.wrist: "orbita3d",
        }

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

    def forward_kinematics(self, joints_positions: Optional[List[float]] = None) -> npt.NDArray[np.float64]:
        req_params = {
            "id": self.part_id,
        }
        if joints_positions is not None:
            if len(joints_positions) != 7:
                raise ValueError(f"joints_positions should be length 7 (got {len(joints_positions)} instead)!")
            req_params["position"] = self._list_to_arm_position(joints_positions)
        req = ArmFKRequest(**req_params)
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
        arm_pos = ArmPosition(
            shoulder_position=Pose2D(axis_1=positions[0], axis_2=positions[1]),
            elbow_position=Pose2D(axis_1=positions[2], axis_2=positions[3]),
            wrist_position=Rotation3D(rpy=ExtEulerAngles(roll=positions[4], pitch=positions[5], yaw=positions[6])),
        )

        return arm_pos

    def _arm_position_to_list(self, arm_pos: ArmPosition) -> List[float]:
        positions = []

        for _, value in arm_pos.shoulder_position.ListFields():
            positions.append(value)
        for _, value in arm_pos.elbow_position.ListFields():
            positions.append(value)
        for _, value in arm_pos.wrist_position.rpy.ListFields():
            positions.append(value)

        return positions

    def goto_from_matrix(self, target: npt.NDArray[np.float64], duration: float = 0) -> None:
        position = target[:3, 3]
        orientation = target[:3, :3]
        target = ArmCartesianGoal(
            id=self.part_id,
            target_position=Point(x=position[0], y=position[1], z=position[2]),
            target_orientation=Rotation3D(matrix=Matrix3x3(roll=orientation[0], pitch=orientation[1], yaw=orientation[2])),
            duration=FloatValue(value=duration),
        )
        self._arm_stub.GoToCartesianPosition(target)

    def goto_from_quaternion(self, position: Tuple[float, float, float], orientation: pyQuat, duration: float = 0) -> None:
        target = ArmCartesianGoal(
            id=self.part_id,
            target_position=Point(x=position[0], y=position[1], z=position[2]),
            target_orientation=Rotation3D(q=Quaternion(w=orientation.w, x=orientation.x, y=orientation.y, z=orientation.z)),
            duration=FloatValue(value=duration),
        )
        self._arm_stub.GoToCartesianPosition(target)

    def goto(
        self,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float],
        position_tol: Optional[Tuple[float, float, float]] = (0, 0, 0),
        orientation_tol: Optional[Tuple[float, float, float]] = (0, 0, 0),
        duration: float = 0,
    ) -> None:
        target = ArmCartesianGoal(
            id=self.part_id,
            target_position=Point(x=position[0], y=position[1], z=position[2]),
            target_orientation=Rotation3D(rpy=ExtEulerAngles(roll=orientation[0], pitch=orientation[1], yaw=orientation[2])),
            duration=FloatValue(value=duration),
        )
        if position_tol is not None:
            target.position_tolerance = PointDistanceTolerances(
                x_tol=position_tol[0], y_tol=position_tol[1], z_tol=position_tol[2]
            )
        if orientation_tol is not None:
            target.orientation_tolerance = ExtEulerAnglesTolerances(
                x_tol=orientation_tol[0], y_tol=orientation_tol[1], z_tol=orientation_tol[2]
            )
        self._arm_stub.GoToCartesianPosition(target)

    def goto_joints(self, positions: List[float], duration: float = 0) -> None:
        arm_pos = self._list_to_arm_position(positions)
        goal = ArmJointGoal(id=self.part_id, position=arm_pos, duration=FloatValue(value=duration))
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
