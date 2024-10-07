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
        The function `get_map` retrieves the current map of the environment using lidar data.

        Returns:
          The `get_map` method returns the current map of the environment as an image (OpenCV format) if
        the lidar map is successfully retrieved. If no lidar map is retrieved, it returns `None`.
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
        This function returns the safety distance in meters of the mobile base from obstacles.

        The mobile base's speed is slowed down if the direction of speed matches the direction of
        at least 1 LIDAR point in the safety_distance range.

        Returns:
          the safety distance in meters of the mobile base from obstacles.
        """
        return float(self._safety_distance)

    @safety_slowdown_distance.setter
    def safety_slowdown_distance(self, value: float) -> None:
        """
        This function sets the safety distance for a Lidar sensor.

        Args:
          value (float): The `value` parameter in the `safety_slowdown_distance` method represents the
        safety distance that is being set for the LidarSafety object. It is of type float and is used to
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
        This function returns the critical distance in meters of the mobile base from obstacles.

        The mobile base's speed is changed to 0 if the direction of speed matches the direction of
        at least 1 LIDAR point in the critical_distance range.
        If at least 1 point is in the critical distance, then even motions that move away from the obstacles are
        slowed down to the "safety_zone" speed.

        Returns:
          The method `safety_critical_distance` returns the critical distance in meters of the mobile
        base from obstacles as a float value.
        """
        return float(self._critical_distance)

    @safety_critical_distance.setter
    def safety_critical_distance(self, value: float) -> None:
        """
        The function `safety_critical_distance` sets the critical distance for a Lidar safety feature.

        Args:
          value (float): The `value` parameter in the `safety_critical_distance` method represents the
        critical distance for safety in the Lidar system. It is a floating-point value that specifies
        the distance at which certain safety measures or actions should be taken to avoid potential
        hazards or collisions.
        """
        self._stub.SetZuuuSafety(
            LidarSafety(
                critical_distance=FloatValue(value=value),
            )
        )

    @property
    def safety_enabled(self) -> bool:
        """
        The function `safety_enabled` returns the current status of the safety feature.

        Returns:
          The method `safety_enabled` is returning the value of the `_safety_enabled` attribute, which
        is a boolean indicating whether the safety feature is enabled or disabled.
        """
        return self._safety_enabled

    @safety_enabled.setter
    def safety_enabled(self, value: bool) -> None:
        """
        The function `safety_enabled` sets the safety status for a Lidar device.

        Args:
          value (bool): The `value` parameter in the `safety_enabled` method is a boolean value that
        indicates whether safety features are enabled or disabled. It is used to set the safety status
        of the Lidar device.
        """
        self._stub.SetZuuuSafety(
            LidarSafety(
                safety_on=BoolValue(value=value),
            )
        )

    @property
    def obstacle_detection_status(self) -> LidarObstacleDetectionStatus:
        """
        This function returns the status of the lidar obstacle detection, which can be one of four
        possible values.

        Returns:
          the status of the lidar obstacle detection, which can be one of the following values:
        NO_OBJECT_DETECTED, OBJECT_DETECTED_SLOWDOWN, OBJECT_DETECTED_STOP, or DETECTION_ERROR.
        """
        return self._obstacle_detection_status

    def reset_safety_default_values(self) -> None:
        """
        The function `reset_safety_default_values` resets default distance values for safety detection.

        Reset values are:
        - safety_critical_distance
        - safety_slowdown_distance.
        """
        self._stub.ResetDefaultValues(self._part._part_id)

    def _update_with(self, new_lidar_state: LidarSafety) -> None:
        """
        The function `_update_with` updates lidar information with a new state received from a gRPC
        server.

        Args:
          new_lidar_state (LidarSafety): `new_lidar_state` is an object of type `LidarSafety` that
        contains information about the lidar state received from the gRPC server.
        """
        self._safety_enabled = new_lidar_state.safety_on.value
        self._safety_distance = new_lidar_state.safety_distance.value
        self._critical_distance = new_lidar_state.critical_distance.value

        self._obstacle_detection_status = LidarObstacleDetectionEnum.Name(new_lidar_state.obstacle_detection_status.status)
