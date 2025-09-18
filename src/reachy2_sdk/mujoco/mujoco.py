"""MuJoCo module for Reachy2 SDK.

This module provides access to objects poses in the MuJoCo scenes.
"""

import logging
from typing import List
import grpc

from google.protobuf.empty_pb2 import Empty


import reachy2_sdk_api
from reachy2_sdk_api.reachy_pb2 import Reachy
from reachy2_sdk_api.mujoco_pb2 import MujocoObjectPose, MujocoObjectsPoses
from reachy2_sdk_api.mujoco_pb2_grpc import MujocoServiceStub


class Mujoco:
    """The ReachyMujoco class provides access to the MuJoCo simulation data.

    The ReachyMujoco class allows users to retrieve the poses of objects in the MuJoCo simulation environment.
    It interacts with the MujocoServiceStub to fetch the relevant data.
    """

    def __init__(self, grpc_channel: grpc.Channel) -> None:
        """Initialize the ReachyMujoco instance with robot details.

        Args:
            reachy: The Reachy robot object, which provides the robot's info and configuration details.
        """
        self._logger = logging.getLogger(__name__)
        self._mujoco_stub = MujocoServiceStub(grpc_channel)

    def get_objects_poses(self) -> List[MujocoObjectPose]:
        """Retrieve the poses of objects in the MuJoCo simulation.

        Returns:
            A list of MujocoObjectPose instances representing the poses of objects in the simulation.
        """
        try:
            response = self._mujoco_stub.GetObjectsPoses(Empty())
            return list(response.poses)
        except Exception as e:
            print(f"Failed to retrieve object poses from MuJoCo: {e}")
            return []