"""Reachy Head module.

Handles all specific method to an Head:
- the inverse kinematics
- look_at function
"""
import grpc

from pyquaternion import Quaternion

from reachy_sdk_api_v2.head_pb2_grpc import HeadStub
from reachy_sdk_api_v2.head_pb2 import Head as Head_proto
from reachy_sdk_api_v2.part_pb2 import PartId


class Head:
    """Head class.

    It exposes the neck orbita actuator at the base of the head.
    It provides look_at utility function to directly orient the head so it looks at a cartesian point
    expressed in Reachy's coordinate system.
    """

    def __init__(self, head: Head_proto, grpc_channel: grpc.Channel) -> None:
        """Set up the head."""
        self._head_stub = HeadStub(grpc_channel)
        self.part_id = PartId(id=head.part_id)

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
