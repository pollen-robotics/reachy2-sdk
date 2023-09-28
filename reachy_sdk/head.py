"""Reachy Head module.

Handles all specific method to an Head:
- the inverse kinematics
- look_at function
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from pyquaternion import Quaternion

from reachy_sdk_api_v2.head_pb2_grpc import HeadStub
from reachy_sdk_api_v2.head_pb2 import Head
from reachy_sdk_api_v2.part_pb2 import PartId


class Head:
    """Head class.

    It exposes the neck orbita actuator at the base of the head.
    It provides look_at utility function to directly orient the head so it looks at a cartesian point
    expressed in Reachy's coordinate system.
    """

    def __init__(self, head: Head, grpc_channel) -> None:
        """Set up the head."""
        self._head_stub = HeadStub(grpc_channel)
        self.part_id = PartId(id=head.part_id)

    def look_at(self, x, y, z, duration):
        pass

    def orient(self, q, duration):
        pass

    def rotate_to(self, roll, pitch, yaw, duration):
        pass

    def turn_on(self):
        self._head_stub.TurnOn(self.part_id)

    def turn_off(self):
        self._head_stub.TurnOff(self.part_id)