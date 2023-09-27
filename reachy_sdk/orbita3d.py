"""Reachy Arm module.

Handles all specific method to an Arm (left and/or right) especially:
- the forward kinematics
- the inverse kinematics
"""

from typing import List, Optional, Set

from reachy_sdk_api_v2.orbita3d_pb2 import Orbita3D

import numpy as np


class Orbita3D():
    """Arm abstract class used for both left/right arms.

    It exposes the kinematics of the arm:
    - you can access the joints actually used in the kinematic chain,
    - you can compute the forward and inverse kinematics
    """

    def __init__(self, part: Orbita3D, grpc_channel) -> None:
        """Set up the arm with its kinematics."""

