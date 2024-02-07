"""Reachy Camera Manager module.

Initialize the two teleop and SR cameras if they are available.

"""
import atexit
import logging
import threading
from enum import Enum
from typing import Optional

import grpc
from google.protobuf.empty_pb2 import Empty
from reachy2_sdk_api.video_pb2_grpc import VideoServiceStub

from .camera import Camera, SRCamera


class CameraType(Enum):
    # values defined in pollen-vision
    TELEOP = "teleop_head"
    SR = "other"


class CameraManager:
    """CameraManager class provide the available cameras."""

    def __init__(self, host: str, port: int) -> None:
        """Set up video module"""
        self._logger = logging.getLogger(__name__)
        self._grpc_video_channel = grpc.insecure_channel(f"{host}:{port}")

        self._video_stub = VideoServiceStub(self._grpc_video_channel)

        self._teleop: Optional[Camera] = None
        self._SR: Optional[SRCamera] = None

        self._init_thread = threading.Thread(target=self._setup_cameras)
        self._init_thread.start()
        atexit.register(self.cleanup)

    def _setup_cameras(self) -> None:
        cams = self._video_stub.InitAllCameras(Empty())
        if len(cams.camera_info) == 0:
            self._logger.error("Cameras not initialized.")
        else:
            self._logger.info(cams.camera_info)
            for c in cams.camera_info:
                if c.name == CameraType.TELEOP.value:
                    self._logger.debug("Teleop Camera initialized.")
                    self._teleop = Camera(c, self._video_stub)
                elif c.name == CameraType.SR.value:
                    self._logger.debug("SR Camera initialized.")
                    self._SR = SRCamera(c, self._video_stub)
                else:
                    self._logger.error(f"Camera {c.name} not defined")

    def cleanup(self) -> None:
        self._init_thread.join()
        self._video_stub.GoodBye(Empty())

    @property
    def teleop(self) -> Optional[Camera]:
        if self._init_thread.is_alive():
            self._logger.info("waiting for camera to be initialized")
            self._init_thread.join()
        return self._teleop

    @property
    def SR(self) -> Optional[SRCamera]:
        if self._init_thread.is_alive():
            self._logger.info("waiting for camera to be initialized")
            self._init_thread.join()
        return self._SR
