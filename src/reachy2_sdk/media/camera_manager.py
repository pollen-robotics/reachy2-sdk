"""Reachy Camera Manager module.

Initialize the head and torso cameras if they are available.
"""

import logging
from typing import Optional

import grpc
from google.protobuf.empty_pb2 import Empty
from reachy2_sdk_api.video_pb2_grpc import VideoServiceStub

from .camera import Camera, CameraType, DepthCamera


class CameraManager:
    """CameraManager class manages the available cameras on the robot.

    Provides access to the robot's cameras, including teleoperation and depth cameras.
    It handles the initialization of cameras and offers methods to retrieve camera objects for use.
    """

    def __init__(self, host: str, port: int) -> None:
        """Set up the camera manager module.

        This initializes the gRPC channel for communicating with the camera service,
        sets up logging, and prepares the available cameras.

        Args:
            host: The host address for the gRPC service.
            port: The port number for the gRPC service.
        """
        self._logger = logging.getLogger(__name__)
        self._grpc_video_channel = grpc.insecure_channel(f"{host}:{port}")
        self._host = host

        self._video_stub = VideoServiceStub(self._grpc_video_channel)

        self._teleop: Optional[Camera] = None
        self._depth: Optional[DepthCamera] = None
        self._setup_cameras()

    def __repr__(self) -> str:
        """Clean representation of a reachy cameras."""
        s = "\n\t".join([str(cam) for cam in [self._depth, self._teleop] if cam is not None])
        return f"""<CameraManager intialized_cameras=\n\t{s}\n>"""

    def _setup_cameras(self) -> None:
        """Initialize cameras based on availability.

        This method retrieves the available cameras and sets
        up the teleop and depth cameras if they are found.
        """
        cams = self._video_stub.GetAvailableCameras(Empty())
        self._teleop = None
        self._depth = None
        if len(cams.camera_feat) == 0:
            self._logger.warning("There is no available camera.")
        else:
            self._logger.debug(cams.camera_feat)
            for c in cams.camera_feat:
                if c.name == CameraType.TELEOP.value:
                    self._logger.debug("Teleop Camera initialized.")
                    self._teleop = Camera(c, self._video_stub)
                elif c.name == CameraType.DEPTH.value:
                    self._logger.debug("Depth Camera initialized.")
                    self._depth = DepthCamera(c, self._video_stub)
                else:
                    self._logger.error(f"Camera {c.name} not defined")

    def initialize_cameras(self) -> None:
        """Manually re-initialize cameras.

        This method can be used to reinitialize the camera setup if changes occur
        or new cameras are connected.
        """
        self._setup_cameras()

    @property
    def teleop(self) -> Optional[Camera]:
        """Retrieve the teleop camera.

        Returns:
            The teleop Camera object if it is initialized; otherwise, logs an error
            and returns None.
        """
        if self._teleop is None:
            self._logger.error("Teleop camera is not initialized.")
            return None

        return self._teleop

    @property
    def depth(self) -> Optional[DepthCamera]:
        """Retrieve the depth camera.

        Returns:
            The DepthCamera object if it is initialized; otherwise, logs an error
            and returns None.
        """
        if self._depth is None:
            self._logger.error("Depth camera is not initialized.")
            return None

        return self._depth
