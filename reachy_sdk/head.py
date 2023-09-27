"""Reachy Head module.

Handles all specific method to an Head:
- the inverse kinematics
- look_at function
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from pyquaternion import Quaternion

from reachy_sdk_api.head_kinematics_pb2_grpc import HeadKinematicsStub
from reachy_sdk_api.head_kinematics_pb2 import HeadFKRequest, HeadIKRequest
from reachy_sdk_api.kinematics_pb2 import joint__pb2
from reachy_sdk_api.arm_kinematics_pb2 import kinematics__pb2

from .device_holder import DeviceHolder
from .joint import Joint
from .trajectory import goto, goto_async
from .trajectory.interpolation import InterpolationMode


JointId = joint__pb2.JointId
JointPosition = kinematics__pb2.JointPosition
Matrix4x4 = kinematics__pb2.Matrix4x4
QuaternionPb = kinematics__pb2.Quaternion


class Head:
    """Head class.

    It exposes the neck orbita actuator at the base of the head.
    It provides look_at utility function to directly orient the head so it looks at a cartesian point
    expressed in Reachy's coordinate system.
    """

    def __init__(self, joints: List[Joint], grpc_channel) -> None:
        """Set up the head."""
