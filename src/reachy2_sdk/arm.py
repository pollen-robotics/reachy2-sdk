"""Reachy Arm module.

Handles all specific method to an Arm (left and/or right) especially:
- the forward kinematics
- the inverse kinematics
- goto functions
"""
from typing import Any, Dict, List, Optional, Tuple

import grpc
import numpy as np
import numpy.typing as npt
from google.protobuf.wrappers_pb2 import FloatValue
from pyquaternion import Quaternion as pyQuat
from reachy2_sdk_api.arm_pb2 import Arm as Arm_proto
from reachy2_sdk_api.arm_pb2 import (
    ArmCartesianGoal,
    ArmEndEffector,
    ArmFKRequest,
    ArmIKRequest,
    ArmJointGoal,
    ArmLimits,
    ArmPosition,
    ArmState,
    ArmTemperatures,
)
from reachy2_sdk_api.arm_pb2_grpc import ArmServiceStub
from reachy2_sdk_api.kinematics_pb2 import (
    ExtEulerAngles,
    ExtEulerAnglesTolerances,
    Matrix3x3,
    Matrix4x4,
    Point,
    PointDistanceTolerances,
    Quaternion,
    Rotation3d,
)
from reachy2_sdk_api.orbita2d_pb2 import Pose2d
from reachy2_sdk_api.part_pb2 import PartId

from .orbita2d import Orbita2d
from .orbita3d import Orbita3d


class Arm:
    """Arm class used for both left/right arms.

    It exposes the kinematics functions for the arm:
    - you can compute the forward and inverse kinematics
    It also exposes movements functions.
    Arm can be turned on and off.
    """

    def __init__(self, arm_msg: Arm_proto, initial_state: ArmState, grpc_channel: grpc.Channel) -> None:
        """Define an arm (left or right).

        Connect to the arm's gRPC server stub and set up the arm's actuators.
        """
        self._grpc_channel = grpc_channel
        self._arm_stub = ArmServiceStub(grpc_channel)
        self.part_id = PartId(id=arm_msg.part_id.id, name=arm_msg.part_id.name)

        self._setup_arm(arm_msg, initial_state)
        self._actuators = {
            "shoulder": self.shoulder,
            "elbow": self.elbow,
            "wrist": self.wrist,
        }

    def _setup_arm(self, arm: Arm_proto, initial_state: ArmState) -> None:
        """Set up the arm.

        Set up the arm's actuators (shoulder, elbow and wrist) with the arm's description and initial state.
        """
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
        """Turn all motors of the part on.

        All arm's motors will then be stiff.
        """
        self._arm_stub.TurnOn(self.part_id)

    def turn_off(self) -> None:
        """Turn all motors of the part off.

        All arm's motors will then be compliant.
        """
        self._arm_stub.TurnOff(self.part_id)

    def __repr__(self) -> str:
        """Clean representation of an Arm."""
        s = "\n\t".join([act_name + ": " + str(actuator) for act_name, actuator in self._actuators.items()])
        return f"""<Arm actuators=\n\t{
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
            "id": self.part_id,
        }
        if joints_positions is None:
            present_joints_positions = [
                joint.present_position
                for orbita in self._actuators.values()
                for joint in orbita._joints.values()  # type: ignore
            ]
            req_params["position"] = self._list_to_arm_position(present_joints_positions, degrees)

        else:
            if len(joints_positions) != 7:
                raise ValueError(f"joints_positions should be length 7 (got {len(joints_positions)} instead)!")
            req_params["position"] = self._list_to_arm_position(joints_positions, degrees)
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
            )
        }

        if q0 is not None:
            req_params["q0"] = self._list_to_arm_position(q0, degrees)

        req = ArmIKRequest(**req_params)
        resp = self._arm_stub.ComputeArmIK(req)

        if not resp.success:
            raise ValueError(f"No solution found for the given target ({target})!")

        return self._arm_position_to_list(resp.arm_position)

    def _list_to_arm_position(self, positions: List[float], degrees: bool = True) -> ArmPosition:
        """Convert a list of joint positions to an ArmPosition message.

        This is used to send a joint position to the arm's gRPC server and to compute the forward
        and inverse kinematics.
        """
        if degrees:
            positions = self._convert_to_radians(positions)
        arm_pos = ArmPosition(
            shoulder_position=Pose2d(
                axis_1=FloatValue(value=positions[0]),
                axis_2=FloatValue(value=positions[1]),
            ),
            elbow_position=Pose2d(
                axis_1=FloatValue(value=positions[2]),
                axis_2=FloatValue(value=positions[3]),
            ),
            wrist_position=Rotation3d(
                rpy=ExtEulerAngles(
                    roll=positions[4],
                    pitch=positions[5],
                    yaw=positions[6],
                )
            ),
        )

        return arm_pos

    def _convert_to_radians(self, my_list: List[float]) -> Any:
        """Convert a list of angles from degrees to radians."""
        a = np.array(my_list)
        a = np.deg2rad(a)
        return a.tolist()

    def _arm_position_to_list(self, arm_pos: ArmPosition) -> List[float]:
        """Convert an ArmPosition message to a list of joint positions.

        It is used to convert the result of the inverse kinematics.
        """
        positions = []

        for _, value in arm_pos.shoulder_position.ListFields():
            positions.append(value.value)
        for _, value in arm_pos.elbow_position.ListFields():
            positions.append(value.value)
        for _, value in arm_pos.wrist_position.rpy.ListFields():
            positions.append(value.value)

        return positions

    def goto_from_matrix(self, target: npt.NDArray[np.float64], duration: float = 0) -> None:
        """Move the arm to a matrix target (or get close).

        Given a pose 4x4 target matrix (as a numpy array) expressed in Reachy coordinate systems,
        it will try to compute a joint solution to reach this target (or get close),
        and move to this position in the defined duration.
        """
        position = target[:3, 3]
        orientation = target[:3, :3]
        target = ArmCartesianGoal(
            id=self.part_id,
            target_position=Point(x=position[0], y=position[1], z=position[2]),
            target_orientation=Rotation3d(matrix=Matrix3x3(roll=orientation[0], pitch=orientation[1], yaw=orientation[2])),
            duration=FloatValue(value=duration),
        )
        self._arm_stub.GoToCartesianPosition(target)

    def goto_from_quaternion(self, position: Tuple[float, float, float], orientation: pyQuat, duration: float = 0) -> None:
        """Move the arm so that the end effector reaches the given position and orientation.

        Given a 3d position and a quaternion expressed in Reachy coordinate systems,
        it will try to compute a joint solution to reach this target (or get close),
        and move to this position in the defined duration.
        """
        target = ArmCartesianGoal(
            id=self.part_id,
            target_position=Point(x=position[0], y=position[1], z=position[2]),
            target_orientation=Rotation3d(q=Quaternion(w=orientation.w, x=orientation.x, y=orientation.y, z=orientation.z)),
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
        """Move the arm so that the end effector reaches the given position and orientation.

        Given a 3d position and a rpy rotation expressed in Reachy coordinate systems,
        it will try to compute a joint solution to reach this target (or get close),
        and move to this position in the defined duration.

        You can also defined tolerances for each axis of the position and of the orientation.
        """
        target = ArmCartesianGoal(
            id=self.part_id,
            target_position=Point(x=position[0], y=position[1], z=position[2]),
            target_orientation=Rotation3d(rpy=ExtEulerAngles(roll=orientation[0], pitch=orientation[1], yaw=orientation[2])),
            duration=FloatValue(value=duration),
        )
        if position_tol is not None:
            target.position_tolerance = PointDistanceTolerances(
                x_tol=position_tol[0], y_tol=position_tol[1], z_tol=position_tol[2]
            )
        if orientation_tol is not None:
            target.orientation_tolerance = ExtEulerAnglesTolerances(
                x_tol=orientation_tol[0],
                y_tol=orientation_tol[1],
                z_tol=orientation_tol[2],
            )
        self._arm_stub.GoToCartesianPosition(target)

    def goto_joints(self, positions: List[float], duration: float = 0, degrees: bool = True) -> None:
        """Move the arm's joints to reach the given position.

        Given a list of joint positions (exactly 7 joint positions),
        it will move the arm to that position.
        """
        arm_pos = self._list_to_arm_position(positions, degrees)
        goal = ArmJointGoal(id=self.part_id, position=arm_pos, duration=FloatValue(value=duration))
        self._arm_stub.GoToJointPosition(goal)

    @property
    def joints_limits(self) -> ArmLimits:
        """Get limits of all the part's joints"""
        limits = self._arm_stub.GetJointsLimits(self.part_id)
        return limits

    @property
    def temperatures(self) -> ArmTemperatures:
        """Get temperatures of all the part's motors"""
        temperatures = self._arm_stub.GetTemperatures(self.part_id)
        return temperatures

    def _update_with(self, new_state: ArmState) -> None:
        """Update the arm with a newly received (partial) state received from the gRPC server."""
        self.shoulder._update_with(new_state.shoulder_state)
        self.elbow._update_with(new_state.elbow_state)
        self.wrist._update_with(new_state.wrist_state)

    @property
    def compliant(self) -> Dict[str, bool]:
        """Get compliancy of all the part's actuators"""
        return {"shoulder": self.shoulder.compliant, "elbow": self.elbow.compliant, "wrist": self.wrist.compliant}

    @compliant.setter
    def compliant(self, value: bool) -> None:
        """Set compliancy of all the part's actuators"""
        if not isinstance(value, bool):
            raise ValueError("Expecting bool as compliant value")
        if value:
            self._arm_stub.TurnOff(self.part_id)
        else:
            self._arm_stub.TurnOn(self.part_id)