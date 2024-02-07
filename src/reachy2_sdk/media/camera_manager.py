"""Reachy Video module.

Handles all specific method related to video especially:
- get rgb frame
- get depth frame
"""
import atexit
import logging
import threading
from enum import Enum
from typing import List, Optional

import cv2
import grpc
import numpy as np
import numpy.typing as npt
from google.protobuf.empty_pb2 import Empty
from reachy2_sdk_api.video_pb2 import CameraInfo, VideoAck, View, ViewRequest
from reachy2_sdk_api.video_pb2_grpc import VideoServiceStub

from .camera import Camera, SRCamera


class CameraType(Enum):
    # values defined in pollen-vision
    TELEOP = "teleop_head"
    SR = "other"


class CameraManager:
    """Video class used for accessing cameras."""

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

    """
    def get_all_cameras(self) -> List[CameraInfo]:
        cams = self._video_stub.GetAllCameras(Empty())
        return [c for c in cams.camera_info]

    def get_frame(self, cam_info: CameraInfo, view: View = None) -> Optional[npt.NDArray[np.uint8]]:
        frame = self._video_stub.GetFrame(request=ViewRequest(camera_info=cam_info, view=view))
        if frame.data == b"":
            self._logger.error("No frame retrieved")
            return None
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        return img  # type: ignore[no-any-return]

    def get_depth_frame(self, cam_info: CameraInfo, view: View = None) -> Optional[npt.NDArray[np.uint8]]:
        frame = self._video_stub.GetDepthFrame(request=ViewRequest(camera_info=cam_info, view=view))
        if frame.data == b"":
            self._logger.error("No frame retrieved")
            return None
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_GRAYSCALE)
        return img  # type: ignore[no-any-return]

    def get_depthmap(self, cam_info: CameraInfo) -> Optional[npt.NDArray[np.uint16]]:
        frame = self._video_stub.GetDepthMap(request=cam_info)
        if frame.data == b"":
            self._logger.error("No frame retrieved")
            return None
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        return img  # type: ignore[no-any-return]

    def get_disparity(self, cam_info: CameraInfo) -> Optional[npt.NDArray[np.uint16]]:
        frame = self._video_stub.GetDisparity(request=cam_info)
        if frame.data == b"":
            self._logger.error("No frame retrieved")
            return None
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        return img  # type: ignore[no-any-return]

    def capture(self, cam_info: CameraInfo) -> bool:
        ret: VideoAck = self._video_stub.Capture(request=cam_info)
        if not ret.success.value:
            self._logger.error(f"Capture failed: {ret.error}")
            return False
        else:
            return True

    def init_camera(self, cam_info: CameraInfo) -> bool:
        ret: VideoAck = self._video_stub.InitCamera(request=cam_info)
        if not ret.success.value:
            self._logger.error(f"Camera not initialized: {ret.error}")
            return False
        else:
            return True

    def close_camera(self, cam_info: CameraInfo) -> bool:
        ret: VideoAck = self._video_stub.CloseCamera(request=cam_info)
        if not ret.success.value:
            self._logger.error(f"Camera not closed: {ret.error}")
            return False
        else:
            return True

    @staticmethod
    def get_camera_info(list_cam_info: List[CameraInfo], cam_type: CameraType) -> Optional[CameraInfo]:
        cam = None
        for c in list_cam_info:
            if c.name == cam_type.value:
                cam = c
                break

        if cam is None:
            logging.warning(f"There is no camera {cam_type.value}")

        return cam
    """
