"""Reachy Camera module.

Define a RGB Camera (Teleop) and a RGBD Camera (SR). Provide access to the frames (color, depth, disparity)

"""

import logging
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from reachy2_sdk_api.video_pb2 import CameraFeatures, View, ViewRequest
from reachy2_sdk_api.video_pb2_grpc import VideoServiceStub


class CameraView(Enum):
    LEFT = View.LEFT
    RIGHT = View.RIGHT
    DEPTH = View.DEPTH


class CameraType(Enum):
    """Camera names defined in pollen-vision"""

    TELEOP = "teleop_head"
    DEPTH = "depth_camera"


class Camera:
    """
    RGB Camera. Mainly for Reachy Teleop Camera.
    """

    def __init__(self, cam_info: CameraFeatures, video_stub: VideoServiceStub) -> None:
        self._logger = logging.getLogger(__name__)
        self._cam_info = cam_info
        self._video_stub = video_stub

    def get_frame(self, view: CameraView = CameraView.LEFT) -> Optional[Tuple[npt.NDArray[np.uint8], int]]:
        """Get RGB frame (OpenCV) and timestamp in nanosecs"""
        frame = self._video_stub.GetFrame(request=ViewRequest(camera_feat=self._cam_info, view=view.value))
        if frame.data == b"":
            self._logger.warning("No frame retrieved")
            return None
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        return img, frame.timestamp.ToNanoseconds()

    def get_parameters(
        self, view: CameraView = CameraView.LEFT
    ) -> Optional[
        Tuple[int, int, str, npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]
    ]:
        params = self._video_stub.GetParameters(request=ViewRequest(camera_feat=self._cam_info, view=view.value))
        if params.K == []:
            self._logger.warning("No parameter retrieved")
            return None
        return params.height, params.width, params.distortion_model, params.D, params.K, params.R, params.P

    def __repr__(self) -> str:
        """Clean representation of a RGB camera"""
        if self._cam_info.name == CameraType.TELEOP.value:
            name = "teleop"
        elif self._cam_info.name == CameraType.DEPTH.value:
            name = "depth"
        else:
            name = self._cam_info.name
        return f"""<Camera name="{name}" stereo={self._cam_info.stereo}> """


class DepthCamera(Camera):
    """
    RGBD Camera
    """

    def get_depth_frame(self, view: CameraView = CameraView.DEPTH) -> Optional[Tuple[npt.NDArray[np.uint16], int]]:
        """Get 16bit depth view (OpenCV format)"""
        frame = self._video_stub.GetDepth(request=ViewRequest(camera_feat=self._cam_info, view=view.value))
        if frame.data == b"":
            self._logger.error("No frame retrieved")
            return None

        if frame.encoding != "16UC1":
            self._logger.error("Depth is not encoded in 16bit")
        np_data = np.frombuffer(frame.data, np.uint16)
        np_data = np_data.reshape((frame.height, frame.width))

        return np_data, frame.timestamp.ToNanoseconds()
