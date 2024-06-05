"""Reachy Camera module.

Define a RGB Camera (Teleop) and a RGBD Camera (SR). Provide access to the frames (color, depth, disparity)

"""

import logging
from enum import Enum
from google.protobuf.empty_pb2 import Empty
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
from threading import Event, Thread

from reachy2_sdk_api.video_pb2 import CameraInfo, VideoAck, View, ViewRequest
from reachy2_sdk_api.video_pb2_grpc import VideoServiceStub, VideoStreamStub


class CameraView(Enum):
    LEFT = View.LEFT
    RIGHT = View.RIGHT


class CameraType(Enum):
    """Camera names defined in pollen-vision"""

    TELEOP = "teleop_head"
    SR = "other"


class Camera:
    """
    RGB Camera. Mainly for Reachy Teleop Camera.
    """

    def __init__(self, cam_info: CameraInfo, video_stub: VideoServiceStub) -> None:
        self._logger = logging.getLogger(__name__)
        self._cam_info = cam_info
        self._video_stub = video_stub

    def capture(self) -> bool:
        """Synchronized capture of all frames (RGB, Depth, etc)"""
        ret: VideoAck = self._video_stub.Capture(request=self._cam_info)
        if not ret.success.value:
            self._logger.error(f"Capture failed: {ret.error}")
            return False
        else:
            return True

    def get_frame(self, view: CameraView = CameraView.LEFT) -> Optional[npt.NDArray[np.uint8]]:
        """Get RGB frame (OpenCV)"""
        frame = self._video_stub.GetFrame(request=ViewRequest(camera_info=self._cam_info, view=view.value))
        if frame.data == b"":
            self._logger.error("No frame retrieved")
            return None
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        return img  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        """Clean representation of a RGB camera"""
        if self._cam_info.name == CameraType.TELEOP.value:
            name = "teleop"
        elif self._cam_info.name == CameraType.SR.value:
            name = "SR"
        else:
            name = self._cam_info.name
        return f"""<Camera name="{name}" stereo={self._cam_info.stereo}> """


class SRCamera(Camera):
    """
    RGBD Camera. Meant to control a Luxonis SR
    """

    def get_depth_frame(self, view: CameraView = CameraView.LEFT) -> Optional[npt.NDArray[np.uint8]]:
        """Get 8bit depth view (OpenCV format)"""
        frame = self._video_stub.GetDepthFrame(request=ViewRequest(camera_info=self._cam_info, view=view.value))
        if frame.data == b"":
            self._logger.error("No frame retrieved")
            return None
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_GRAYSCALE)
        return img  # type: ignore[no-any-return]

    def get_depthmap(self) -> Optional[npt.NDArray[np.uint16]]:
        """Get 16bit depthmap (OpenCV format)"""
        frame = self._video_stub.GetDepthMap(request=self._cam_info)
        if frame.data == b"":
            self._logger.error("No frame retrieved")
            return None
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        return img  # type: ignore[no-any-return]

    def get_disparity(self) -> Optional[npt.NDArray[np.uint16]]:
        """Get 16bit disparity (OpenCV format)"""
        frame = self._video_stub.GetDisparity(request=self._cam_info)
        if frame.data == b"":
            self._logger.error("No frame retrieved")
            return None
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        return img  # type: ignore[no-any-return]


class CameraTeleopRecorder:
    """
    CameraTeleopRecorder class to get the frames from the teleoperation cameras through grpc.
    """
    def __init__(self, video_stub: VideoStreamStub) -> None:
        self._logger = logging.getLogger(__name__)
        self._got_img = Event()
        self._last_frame: Optional[npt.NDArray[np.uint16]] = None
        self._video_stub = video_stub
        print("Init CameraTeleopRecorder")

    def get_frame(self) -> npt.NDArray[np.uint16]:
        """Get the last frame from the teleoperation camera"""
        frame = self._video_stub.GetFrame(request=Empty())
        if frame.data == b"":
            self._logger.error("No frame retrieved")
            return None
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        return img  # type: ignore[no-any-return]

    @property
    def last_frame(self):
        """Return the last retrieved frame."""
        return self._last_frame

    def _start_sync_in_bg(self) -> None:
        def poll_img() -> None:
            for resp in self._video_stub.GetFrame(request=Empty()):
                buff = np.frombuffer(resp.data, dtype=np.uint8)
                print("buff: ", buff)
                self._last_frame = cv2.imdecode(buff, cv2.IMREAD_COLOR)
                print("last_frame: ", self._last_frame)
                self._got_img.set()

        self._t = Thread(target=poll_img)
        self._t.daemon = True
        self._t.start()

        self._got_img.wait()
