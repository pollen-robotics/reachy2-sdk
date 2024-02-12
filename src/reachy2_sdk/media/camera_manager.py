"""Reachy Camera Manager module.

Initialize the two teleop and SR cameras if they are available.

"""
import atexit
import logging
import threading
from typing import Optional

import grpc
from google.protobuf.empty_pb2 import Empty
from reachy2_sdk_api.video_pb2_grpc import VideoServiceStub

from .camera import Camera, CameraType, SRCamera


class CameraManager:
    """CameraManager class provide the available cameras."""

    def __init__(self, host: str, port: int) -> None:
        """Set up camera manager module"""
        self._logger = logging.getLogger(__name__)
        self._grpc_video_channel = grpc.insecure_channel(f"{host}:{port}")

        self._video_stub = VideoServiceStub(self._grpc_video_channel)

        self._teleop: Optional[Camera] = None
        self._SR: Optional[SRCamera] = None

        self._init_thread = threading.Thread(target=self._setup_cameras)
        self._init_thread.start()
        # SDK Server count the number of clients to release the cameras if there is no one left
        self._cleaned = False
        atexit.register(self._cleanup)

    def __repr__(self) -> str:
        """Clean representation of a reachy cameras."""
        s = "\n\t".join([str(cam) for cam in [self._SR, self._teleop] if cam is not None])
        return f"""<CameraManager intialized_cameras=\n\t{s}\n>"""

    def _setup_cameras(self) -> None:
        """Thread initializing cameras"""
        cams = self._video_stub.InitAllCameras(Empty())
        if len(cams.camera_info) == 0:
            self._logger.error("Cameras not initialized.")
        else:
            self._logger.debug(cams.camera_info)
            for c in cams.camera_info:
                if c.name == CameraType.TELEOP.value:
                    self._logger.debug("Teleop Camera initialized.")
                    self._teleop = Camera(c, self._video_stub)
                elif c.name == CameraType.SR.value:
                    self._logger.debug("SR Camera initialized.")
                    self._SR = SRCamera(c, self._video_stub)
                else:
                    self._logger.error(f"Camera {c.name} not defined")

    def _cleanup(self) -> None:
        """Let the server know that the cameras are not longer used by this client"""
        if not self._cleaned:
            self.wait_end_of_initialization()
            self._video_stub.GoodBye(Empty())
            self._cleaned = True

    def wait_end_of_initialization(self) -> None:
        self._init_thread.join()

    @property
    def teleop(self) -> Optional[Camera]:
        """Get Teleop camera"""
        if self._init_thread.is_alive():
            self._logger.info("waiting for camera to be initialized")
            self.wait_end_of_initialization()

        if self._teleop is None:
            raise AttributeError("There is no Teleop camera")

        return self._teleop

    @property
    def SR(self) -> Optional[SRCamera]:
        """Get SR Camera"""
        if self._init_thread.is_alive():
            self._logger.info("waiting for camera to be initialized")
            self.wait_end_of_initialization()

        if self._SR is None:
            raise AttributeError("There is no SR camera")
        return self._SR
