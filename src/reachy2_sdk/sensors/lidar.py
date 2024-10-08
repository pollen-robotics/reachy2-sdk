"""LIDAR module for mobile base SDK.

Handles the LIDAR features:
    - get the map of the environment
    - set the safety distance
    - set the critical distance
    - enable/disable the safety feature
"""
import logging
from typing import Optional

import cv2
import grpc
import numpy as np
import numpy.typing as npt
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.mobile_base_lidar_pb2 import (
    LidarObstacleDetectionEnum,
    LidarObstacleDetectionStatus,
    LidarSafety,
)
from reachy2_sdk_api.mobile_base_lidar_pb2_grpc import MobileBaseLidarServiceStub

from ..parts.part import Part


class Lidar:
    """LIDAR class for mobile base SDK."""

    def __init__(self, initial_state: LidarSafety, grpc_channel: grpc.Channel, part: Part) -> None:
        """Initialize the LIDAR class."""
        self._logger = logging.getLogger(__name__)
        self._stub = MobileBaseLidarServiceStub(grpc_channel)
        self._part = part

        self._safety_enabled: bool = initial_state.safety_on.value
        self._safety_distance: float = initial_state.safety_distance.value
        self._critical_distance: float = initial_state.critical_distance.value

        self._obstacle_detection_status: str = LidarObstacleDetectionEnum.Name(initial_state.obstacle_detection_status.status)

    def __repr__(self) -> str:
        """Clean representation of a Reachy."""
        return f"""<Lidar safety_enabled={self.safety_enabled}>"""

    def get_map(self) -> Optional[npt.NDArray[np.uint8]]:
        """
        Retrieves the current map of the environment using lidar data.

        Returns:
          current map of the environment as an image (numpy array) if the lidar map is successfully retrieved.
        If no lidar map is retrieved, it returns `None`.
        """
        compressed_map = self._stub.GetLidarMap(self._part._part_id)
        if compressed_map.data == b"":
            self._logger.error("No lidar map retrieved")
            return None
        np_data = np.frombuffer(compressed_map.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        return img  # type: ignore[no-any-return]

    @property
    def safety_slowdown_distance(self) -> float:
        """
        Returns the safety distance from obstacles in meters of the mobile base.

        The mobile base's speed is slowed down if the direction of speed matches the direction of
        at least 1 LIDAR point in the safety_distance range.

        Returns:
          the safety distance in meters of the mobile base from obstacles.
        """
        return float(self._safety_distance)

    @safety_slowdown_distance.setter
    def safety_slowdown_distance(self, value: float) -> None:
        """
        Sets the safety distance for a Lidar sensor.

        Args:
          value (float): safety distance that is being set for the LidarSafety object. It is of type float and is used to
        specify the distance at which a safety slowdown should be initiated.
        """
        self._stub.SetZuuuSafety(
            LidarSafety(
                safety_distance=FloatValue(value=value),
            )
        )

    @property
    def safety_critical_distance(self) -> float:
        """
        Returns the critical distance in meters of the mobile base from obstacles.

        The mobile base's speed is changed to 0 if the direction of speed matches the direction of
        at least 1 LIDAR point in the critical_distance range.
        If at least 1 point is in the critical distance, then even motions that move away from the obstacles are
        slowed down to the "safety_zone" speed.

        Returns:
          the critical distance in meters of the mobile base from obstacles as a float value.
        """
        return float(self._critical_distance)

    @safety_critical_distance.setter
    def safety_critical_distance(self, value: float) -> None:
        """
        Sets the critical distance for a Lidar safety feature.

        Args:
          value (float): critical distance in meters for safety. It specifies the distance at which the mobile base
        should stop if trying to get in the direction of an obstacle. If at least 1 point is in the critical distance,
        then even motions that move away from the obstacles are slowed down to the "safety_zone" speed.
        """
        self._stub.SetZuuuSafety(
            LidarSafety(
                critical_distance=FloatValue(value=value),
            )
        )

    @property
    def safety_enabled(self) -> bool:
        """
        Returns the current status of the safety feature.

        Returns:
          a boolean indicating whether the safety feature is enabled or disabled. If `True`, the safety feature is enabled.
        """
        return self._safety_enabled

    @safety_enabled.setter
    def safety_enabled(self, value: bool) -> None:
        """
        Sets the safety status for the Lidar device.

        Args:
          value (bool): a boolean value that indicates whether safety features are enabled or disabled.
        If `True`, the safety feature is enabled.
        """
        self._stub.SetZuuuSafety(
            LidarSafety(
                safety_on=BoolValue(value=value),
            )
        )

    @property
    def obstacle_detection_status(self) -> LidarObstacleDetectionStatus:
        """
        Returns the status of the lidar obstacle detection, to know if an obstacle is in the LIDAR safety area.

        Returns:
          the status of the lidar obstacle detection, which can be one of the following values:
        NO_OBJECT_DETECTED, OBJECT_DETECTED_SLOWDOWN, OBJECT_DETECTED_STOP, or DETECTION_ERROR.
        """
        return self._obstacle_detection_status

    def reset_safety_default_values(self) -> None:
        """
        Resets default distance values for safety detection.

        Reset values are:
        - safety_critical_distance
        - safety_slowdown_distance.
        """
        self._stub.ResetDefaultValues(self._part._part_id)

    def _update_with(self, new_lidar_state: LidarSafety) -> None:
        """
        Updates lidar information with a new state received from a gRPC server.

        Args:
          new_lidar_state (LidarSafety): contains information about the lidar state received from the gRPC server.
        """
        self._safety_enabled = new_lidar_state.safety_on.value
        self._safety_distance = new_lidar_state.safety_distance.value
        self._critical_distance = new_lidar_state.critical_distance.value

        self._obstacle_detection_status = LidarObstacleDetectionEnum.Name(new_lidar_state.obstacle_detection_status.status)
