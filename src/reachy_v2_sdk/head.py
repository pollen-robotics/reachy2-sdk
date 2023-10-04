"""Reachy Head module.

Handles all specific method to an Head:
- the inverse kinematics
- look_at function
"""
import grpc

from pyquaternion import Quaternion as Quat

from typing import Optional, Tuple

from reachy_sdk_api_v2.head_pb2_grpc import HeadServiceStub
from reachy_sdk_api_v2.head_pb2 import Head as Head_proto, HeadState
from reachy_sdk_api_v2.head_pb2 import HeadPosition, NeckPosition, NeckOrientation
from reachy_sdk_api_v2.head_pb2 import NeckFKRequest, NeckIKRequest
from reachy_sdk_api_v2.part_pb2 import PartId
from reachy_sdk_api_v2.kinematics_pb2 import Quaternion

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

    def _setup_head(self, head: Head_proto, initial_state: HeadState) -> None:
        description = head.description
        self.neck = Orbita3d(
            name=description.neck.id.id,
            initial_state=initial_state.neck_state,
            grpc_channel=self._grpc_channel,
        )
        self.l_antenna = DynamixelMotor(
            name=description.l_antenna.id.id,
            initial_state=initial_state.l_antenna_state,
            grpc_channel=self._grpc_channel,
        )
        self.r_antenna = DynamixelMotor(
            name=description.r_antenna.id.id,
            initial_state=initial_state.r_antenna_state,
            grpc_channel=self._grpc_channel,
        )

    def get_orientation(self) -> Quat:
        quat = self._head_stub.GetOrientation(self.part_id)
        return Quat(w=quat.w, x=quat.x, y=quat.y, z=quat.z)

    def forward_kinematics(self, rpy_position: Optional[Tuple[float, float, float]] = None) -> Quat:
        if rpy_position is None:
            self.get_orientation()
        else:
            req = NeckFKRequest(
                id=self.part_id,
                position=HeadPosition(
                    neck_position=NeckPosition(neck_roll=rpy_position[0], neck_pitch=rpy_position[1], neck_yaw=rpy_position[2])
                ),
            )
            quat = self._head_stub.ComputeNeckFK(req)
            return Quat(w=quat.w, x=quat.x, y=quat.y, z=quat.z)

    def inverse_kinematics(
        self, orientation: Optional[Quat] = None, rpy_q0: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[float, float, float]:
        req = NeckIKRequest(
            id=self.part_id,
        )
        if orientation is not None:
            req.target = NeckOrientation(
                q=Quaternion(
                    w=orientation.w,
                    x=orientation.x,
                    y=orientation.y,
                    z=orientation.z,
                )
            )
        if rpy_q0 is not None:
            req.q0 = NeckPosition(neck_roll=rpy_q0[0], neck_pitch=rpy_q0[1], neck_yaw=rpy_q0[2])
        rpy_pos = self._head_stub.ComputeNeckIK(req)
        return (rpy_pos.position.neck_roll, rpy_pos.position.neck_pitch, rpy_pos.position.neck_yaw)

    def look_at(self, x: float, y: float, z: float, duration: float) -> None:
        pass

    def orient(self, q: Quaternion, duration: float) -> None:
        pass

    def rotate_to(self, roll: float, pitch: float, yaw: float, duration: float) -> None:
        pass

    def turn_on(self) -> None:
        self._head_stub.TurnOn(self.part_id)

    def turn_off(self) -> None:
        self._head_stub.TurnOff(self.part_id)

    def _update_with(self, new_state: HeadState) -> None:
        """Update the head with a newly received (partial) state received from the gRPC server."""
        self.neck._update_with(new_state.neck_state)
        self.l_antenna._update_with(new_state.l_antenna_state)
        self.r_antenna._update_with(new_state.r_antenna_state)
