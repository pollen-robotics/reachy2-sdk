"""Reachy Video module.

Handles all specific method related to video especially:
- get rgb frame
- get depth frame
"""
import logging
from typing import Any, List

import cv2
import grpc
import numpy as np
from google.protobuf.empty_pb2 import Empty
from reachy2_sdk_api.video_pb2 import CameraInfo, VideoAck, View, ViewRequest
from reachy2_sdk_api.video_pb2_grpc import VideoServiceStub


class Video:
    """Video class used for accessing cameras."""

    def __init__(self, host: str, port: int) -> None:
        """Set up video module"""
        self._logger = logging.getLogger(__name__)
        self._grpc_video_channel = grpc.insecure_channel(f"{host}:{port}")

        self._video_stub = VideoServiceStub(self._grpc_video_channel)

    def get_all_cameras(self) -> List[CameraInfo]:
        cams = self._video_stub.GetAllCameras(Empty())
        return [c for c in cams.camera_info]

    def get_frame(self, cam_info: CameraInfo, view: View = None) -> Any:
        frame = self._video_stub.GetFrame(request=ViewRequest(camera_info=cam_info, view=view))
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        return img

    def get_depth_frame(self, cam_info: CameraInfo, view: View = None) -> Any:
        frame = self._video_stub.GetDepthFrame(request=ViewRequest(camera_info=cam_info, view=view))
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_GRAYSCALE)
        return img

    def get_depthmap(self, cam_info: CameraInfo) -> Any:
        frame = self._video_stub.GetDepthMap(request=cam_info)
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        return img

    def get_disparity(self, cam_info: CameraInfo) -> Any:
        frame = self._video_stub.GetDisparity(request=cam_info)
        np_data = np.frombuffer(frame.data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        return img

    def capture(self, cam_info: CameraInfo) -> bool:
        ret: VideoAck = self._video_stub.Capture(request=cam_info)
        if not ret.success:
            self._logger.error(f"Capture failed: {ret.error}")
            return False
        else:
            return True

    def init_camera(self, cam_info: CameraInfo) -> bool:
        ret: VideoAck = self._video_stub.InitCamera(request=cam_info)
        if not ret.success:
            self._logger.error(f"Camera not initialized: {ret.error}")
            return False
        else:
            return True

    def close_camera(self, cam_info: CameraInfo) -> bool:
        ret: VideoAck = self._video_stub.CloseCamera(request=cam_info)
        if not ret.success:
            self._logger.error(f"Camera not closed: {ret.error}")
            return False
        else:
            return True
