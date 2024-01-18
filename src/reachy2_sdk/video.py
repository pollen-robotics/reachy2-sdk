"""Reachy Video module.

Handles all specific method related to video especially:
- get rgb frame
- get depth frame
"""
import logging

import grpc
from google.protobuf.empty_pb2 import Empty
from reachy2_sdk_api.component_pb2 import ComponentId
from reachy2_sdk_api.video_pb2 import ListOfCameraInfo
from reachy2_sdk_api.video_pb2_grpc import VideoServiceStub


class Video:
    """Video class used for accessing cameras."""

    def __init__(self, host: str, port: int) -> None:
        """Set up video module"""
        self._logger = logging.getLogger(__name__)
        self._grpc_video_channel = grpc.insecure_channel(f"{host}:{port}")

        self._video_stub = VideoServiceStub(self._grpc_video_channel)

    def get_all_cameras(self) -> None:
        cams_list: ListOfCameraInfo = self._video_stub.GetAllCameras(Empty())
        self._logger.info(cams_list)

    def get_frame(self) -> None:
        self._logger.info("there")
        fake_com = ComponentId(id=0, name="test")
        self._video_stub.GetFrame(fake_com)
        self._logger.info("here")
