"""Reachy Camera module.

Define the RGB Camera of Reachy's head (Teleop) and the RGBD Camera of its torso (Depth). 
Provide access to the frames (color, depth, disparity) and the camera parameters. 

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
    RGB Camera. Mainly for the teleoperation camera, but also for the RGB part of the RGBD torso camera.
    Allows access to the frame and to the camera parameters.
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
        """Get camera parameters:
        - frame height
        - frame width
        - distortion model
        - distortion parameters D
        - intrinsic camera matrix K
        - rectification matrix R
        - projection matrix P
        """
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
    Depth part of the RGBD torso camera.
    Allows access to the depth frame.
    """

    def get_depth_frame(self, view: CameraView = CameraView.DEPTH) -> Optional[Tuple[npt.NDArray[np.uint16], int]]:
        """Get 16bit depth view (OpenCV format) and timestamp in nanosecs"""
        frame = self._video_stub.GetDepth(request=ViewRequest(camera_feat=self._cam_info, view=view.value))
        if frame.data == b"":
            self._logger.error("No frame retrieved")
            return None

        if frame.encoding != "16UC1":
            self._logger.error("Depth is not encoded in 16bit")
        np_data = np.frombuffer(frame.data, np.uint16)
        np_data = np_data.reshape((frame.height, frame.width))

        return np_data, frame.timestamp.ToNanoseconds()
