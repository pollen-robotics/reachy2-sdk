"""Reachy Head module.

Handles all specific method to an Head:
- the inverse kinematics
- look_at function
"""
import grpc

from pyquaternion import Quaternion as pyQuat

from google.protobuf.wrappers_pb2 import FloatValue

from typing import Optional, Tuple

from reachy_sdk_api_v2.head_pb2_grpc import HeadServiceStub
from reachy_sdk_api_v2.head_pb2 import Head as Head_proto, HeadState
from reachy_sdk_api_v2.head_pb2 import HeadLookAtGoal, NeckGoal
from reachy_sdk_api_v2.head_pb2 import HeadPosition, NeckOrientation
from reachy_sdk_api_v2.head_pb2 import NeckFKRequest, NeckIKRequest
from reachy_sdk_api_v2.part_pb2 import PartId
from reachy_sdk_api_v2.kinematics_pb2 import Point, Rotation3D, Quaternion, ExtEulerAngles

from .orbita3d import Orbita3d
from .dynamixel_motor import DynamixelMotor


class Head:
    """Head class.

    It exposes the neck orbita actuator at the base of the head.
    It provides look_at utility function to directly orient the head so it looks at a cartesian point
    expressed in Reachy's coordinate system.
    """

    def __init__(self, head_msg: Head_proto, initial_state: HeadState, grpc_channel: grpc.Channel) -> None:
        """Set up the head."""
        self._grpc_channel = grpc_channel
        self._head_stub = HeadServiceStub(grpc_channel)
        self.part_id = PartId(id=head_msg.part_id.id, name=head_msg.part_id.name)

        self._setup_head(head_msg, initial_state)
        self._actuators = {
            self.neck: "orbita3d",
            self.l_antenna: "dynamixel_motor",
            self.r_antenna: "dynamixel_motor",
        }

    def _setup_head(self, head: Head_proto, initial_state: HeadState) -> None:
        description = head.description
        self.neck = Orbita3d(
            uid=description.neck.id.id,
            name=description.neck.id.name,
            initial_state=initial_state.neck_state,
            grpc_channel=self._grpc_channel,
        )
        self.l_antenna = DynamixelMotor(
            uid=description.l_antenna.id.id,
            name=description.l_antenna.id.name,
            initial_state=initial_state.l_antenna_state,
            grpc_channel=self._grpc_channel,
        )
        self.r_antenna = DynamixelMotor(
            uid=description.r_antenna.id.id,
            name=description.r_antenna.id.name,
            initial_state=initial_state.r_antenna_state,
            grpc_channel=self._grpc_channel,
        )

    def get_orientation(self) -> pyQuat:
        quat = self._head_stub.GetOrientation(self.part_id).q
        return pyQuat(w=quat.w, x=quat.x, y=quat.y, z=quat.z)

    def forward_kinematics(self, rpy_position: Optional[Tuple[float, float, float]] = None) -> pyQuat:
        if rpy_position is None:
            return self.get_orientation()
        else:
            req = NeckFKRequest(
                id=self.part_id,
                position=HeadPosition(
                    neck_position=Rotation3D(
                        rpy=ExtEulerAngles(roll=rpy_position[0], pitch=rpy_position[1], yaw=rpy_position[2])
                    )
                ),
            )
            res = self._head_stub.ComputeNeckFK(req)
            quat = res.orientation.q
            return pyQuat(w=quat.w, x=quat.x, y=quat.y, z=quat.z)

    def inverse_kinematics(
        self, orientation: Optional[pyQuat] = None, rpy_q0: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[float, float, float]:
        req_params = {
            "id": self.part_id,
        }
        if orientation is not None:
            req_params["target"] = NeckOrientation(
                q=Quaternion(
                    w=orientation.w,
                    x=orientation.x,
                    y=orientation.y,
                    z=orientation.z,
                )
            )
        if rpy_q0 is not None:
            req_params["q0"] = Rotation3D(rpy=ExtEulerAngles(roll=rpy_q0[0], pitch=rpy_q0[1], yaw=rpy_q0[2]))
        req = NeckIKRequest(**req_params)
        rpy_pos = self._head_stub.ComputeNeckIK(req)
        return (rpy_pos.position.rpy.roll, rpy_pos.position.rpy.pitch, rpy_pos.position.rpy.yaw)

    def look_at(self, x: float, y: float, z: float, duration: float) -> None:
        req = HeadLookAtGoal(id=self.part_id, point=Point(x=x, y=y, z=z), duration=FloatValue(value=duration))
        self._head_stub.LookAt(req)

    def orient(self, q: pyQuat, duration: float) -> None:
        req = NeckGoal(id=self.part_id, rotation=Rotation3D(q=Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)), duration=duration)
        self._head_stub.GoToOrientation(req)

    def rotate_to(self, roll: float, pitch: float, yaw: float, duration: float) -> None:
        req = NeckGoal(
            id=self.part_id, rotation=Rotation3D(rpy=ExtEulerAngles(roll=roll, pitch=pitch, yaw=yaw)), duration=duration
        )
        self._head_stub.GoToOrientation(req)

    def turn_on(self) -> None:
        self._head_stub.TurnOn(self.part_id)

    def turn_off(self) -> None:
        self._head_stub.TurnOff(self.part_id)

    def _update_with(self, new_state: HeadState) -> None:
        """Update the head with a newly received (partial) state received from the gRPC server."""
        self.neck._update_with(new_state.neck_state)
        self.l_antenna._update_with(new_state.l_antenna_state)
        self.r_antenna._update_with(new_state.r_antenna_state)
