"""Reachy Head module.

Handles all specific method to an Head:
- the inverse kinematics
- look_at function
"""
import grpc

from pyquaternion import Quaternion as Quat

from reachy_sdk_api_v2.head_pb2_grpc import HeadServiceStub
from reachy_sdk_api_v2.head_pb2 import Head as Head_proto, HeadState
from reachy_sdk_api_v2.head_pb2 import HeadTargetPoint, NeckGoal
from reachy_sdk_api_v2.part_pb2 import PartId
from reachy_sdk_api_v2.kinematics_pb2 import Point, Rotation3D, Quaternion

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

    def look_at(self, x: float, y: float, z: float, duration: float) -> None:
        req = HeadTargetPoint(id=self.part_id, point=Point(x=x, y=y, z=z))
        self._head_stub.LookAt(req)

    def orient(self, q: Quat, duration: float) -> None:
        req = NeckGoal(id=self.part_id, rotation=Rotation3D(q=Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)), duration=duration)
        self._head_stub.GoToOrientation(req)

    def rotate_to(self, roll: float, pitch: float, yaw: float, duration: float) -> None:
        # quat = self.inverse_kinematics((roll, pitch, yaw))
        # self.orient(quat, duration)
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
