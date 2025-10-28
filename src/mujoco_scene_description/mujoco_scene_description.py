"""MuJoCo module for Reachy2 SDK.

This module provides access to objects poses in the MuJoCo scenes.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from google.protobuf.empty_pb2 import Empty
from reachy2_sdk_api.mujoco_pb2_grpc import MujocoServiceStub

from reachy2_sdk import ReachySDK
from reachy2_sdk.utils.utils import invert_affine_transformation_matrix


class MujocoSceneDescription:
    """The MujocoSceneDescription class provides access to the MuJoCo simulation data.

    The MujocoSceneDescription class allows users to retrieve the poses of objects in the MuJoCo simulation environment.
    It interacts with the MujocoServiceStub to fetch the relevant data.
    """

    def __init__(self, reachy: ReachySDK) -> None:
        """Initialize the MujocoSceneDescription instance with robot details.

        Args:
            reachy: An instance of the ReachySDK class representing the robot.
        """
        if reachy._mode != "MUJOCO":
            raise ValueError(f"Reachy is in {reachy._mode} mode, not MUJOCO.")
        self._reachy = reachy
        self._logger = logging.getLogger(__name__)
        self._mujoco_stub = MujocoServiceStub(reachy._grpc_channel)

    def get_objects_poses(self) -> Dict[str, npt.NDArray[np.float32]]:
        """Retrieve the poses of objects in the MuJoCo simulation.

        Returns:
            A dictionary mapping object names to their poses (4x4 numpy arrays).
        """
        try:
            response = self._mujoco_stub.GetObjectsPoses(Empty())
            obj_dict = {}
            for obj in response.poses:
                obj_name = obj.name
                obj_pose: npt.NDArray[np.float32] = np.array(obj.pose.data).reshape(4, 4)
                obj_dict[obj_name] = obj_pose
            return obj_dict

        except Exception as e:
            print(f"Failed to retrieve object poses from MuJoCo: {e}")
            return {}

    def get_object_pose(self, object_name: str) -> Optional[npt.NDArray[np.float32]]:
        """Retrieve the pose of a specific object in the MuJoCo simulation.

        Args:
            object_name: The name of the object whose pose is to be retrieved.

        Returns:
            A 4x4 numpy array representing the pose of the specified object.
        """
        try:
            response = self._mujoco_stub.GetObjectsPoses(Empty())
            available_objects = []
            for obj in response.poses:
                available_objects.append(obj.name)
                if obj.name == object_name:
                    obj_pose: npt.NDArray[np.float32] = np.array(obj.pose.data).reshape(4, 4)
                    return obj_pose
            raise ValueError(f"'{object_name}' not found : available objects are {available_objects}")
        except Exception as e:
            self._logger.error(f"Failed to retrieve object pose:\n{e}")
            return None

    def get_relative_object_pose(self, object_name: str) -> Optional[npt.NDArray[np.float32]]:
        """Retrieve the pose of a specific object in the MuJoCo simulation relative to Reachy's frame.

        Args:
            object_name: The name of the object whose relative pose is to be retrieved.

        Returns:
            A 4x4 numpy array representing the pose of the specified object relative to Reachy's torso
        """
        if object_name == "torso":
            raise ValueError("The object name must be different from 'torso'.")

        try:
            reachy_pose: Optional[npt.NDArray[np.float32]] = self.get_object_pose("torso")
            obj_pose: Optional[npt.NDArray[np.float32]] = self.get_object_pose(object_name)
            if reachy_pose is None or obj_pose is None:
                raise ValueError("Could not compute relative pose.")

            reachy_pose_inv: npt.NDArray[np.float32] = invert_affine_transformation_matrix(reachy_pose)
            obj_pose_relative: npt.NDArray[np.float32] = reachy_pose_inv @ obj_pose
            return obj_pose_relative

        except Exception as e:
            self._logger.error(f"{e}")
            return None

    def get_tracked_objects_list(self) -> List[str]:
        """Retrieve the list of tracked objects in the MuJoCo simulation.

        Returns:
            A list of strings representing the names of tracked objects.
        """
        try:
            response = self._mujoco_stub.GetObjectsPoses(Empty())
            tracked_objects = [obj.name for obj in response.poses]
            return tracked_objects

        except Exception as e:
            self._logger.error(f"Failed to retrieve tracked objects list:\n{e}")
            return []
