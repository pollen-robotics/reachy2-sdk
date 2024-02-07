import logging
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
from reachy2_sdk_api.video_pb2 import CameraInfo, VideoAck, View, ViewRequest
from reachy2_sdk_api.video_pb2_grpc import VideoServiceStub


class CameraView(Enum):
    LEFT = View.LEFT
    RIGHT = View.RIGHT


class Camera:
    def __init__(self, cam_info: CameraInfo, video_stub: VideoServiceStub) -> None:
        self._logger = logging.getLogger(__name__)
        self._cam_info = cam_info
        self._video_stub = video_stub

    def capture(self) -> bool:
        ret: VideoAck = self._video_stub.Capture(request=self._cam_info)
        if not ret.success.value:
            self._logger.error(f"Capture failed: {ret.error}")
            return False
        else:
            return True

    def get_frame(self, view: CameraView = CameraView.LEFT) -> Optional[npt.NDArray[np.uint8]]:
        frame = self._video_stub.GetFrame(request=ViewRequest(camera_info=self._cam_info, view=view.value))
        if frame.data == b"":
            self._logger.error("No frame retrieved")
            return None
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        return img  # type: ignore[no-any-return]


class SRCamera(Camera):
    def get_depth_frame(self, view: CameraView = CameraView.LEFT) -> Optional[npt.NDArray[np.uint8]]:
        frame = self._video_stub.GetDepthFrame(request=ViewRequest(camera_info=self._cam_info, view=view.value))
        if frame.data == b"":
            self._logger.error("No frame retrieved")
            return None
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_GRAYSCALE)
        return img  # type: ignore[no-any-return]

    def get_depthmap(self) -> Optional[npt.NDArray[np.uint16]]:
        frame = self._video_stub.GetDepthMap(request=self._cam_info)
        if frame.data == b"":
            self._logger.error("No frame retrieved")
            return None
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        return img  # type: ignore[no-any-return]

    def get_disparity(self) -> Optional[npt.NDArray[np.uint16]]:
        frame = self._video_stub.GetDisparity(request=self._cam_info)
        if frame.data == b"":
            self._logger.error("No frame retrieved")
            return None
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        return img  # type: ignore[no-any-return]
