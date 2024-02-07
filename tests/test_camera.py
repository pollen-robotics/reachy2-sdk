from typing import List

import numpy as np
import pytest
from reachy2_sdk_api.video_pb2 import CameraInfo, View

from reachy2_sdk.media.camera import Camera, CameraView
from src.reachy2_sdk.reachy_sdk import ReachySDK


@pytest.fixture(scope="module")
def reachy_sdk() -> ReachySDK:
    reachy = ReachySDK(host="localhost")
    assert reachy.grpc_status == "connected"

    assert reachy.turn_on()

    yield reachy

    assert reachy.turn_off()

    reachy.disconnect()
    ReachySDK.clear()


@pytest.mark.online
def test_no_camera(reachy_sdk: ReachySDK) -> None:
    assert reachy_sdk.cameras.teleop is None
    assert reachy_sdk.cameras.SR is None


@pytest.mark.sr_camera
def test_sr_camera(reachy_sdk: ReachySDK) -> None:
    assert reachy_sdk.cameras.SR is not None

    assert reachy_sdk.cameras.SR.capture()

    frame = reachy_sdk.cameras.SR.get_frame()
    assert frame is not None
    assert frame.dtype == np.uint8

    frame = reachy_sdk.cameras.SR.get_depth_frame(CameraView.LEFT)
    assert frame is not None
    assert frame.dtype == np.uint8

    frame_right = reachy_sdk.cameras.SR.get_depth_frame(CameraView.RIGHT)
    assert frame_right is not None
    assert frame_right.dtype == np.uint8

    # check that we don't return the same view
    assert not np.array_equal(frame, frame_right)

    frame = reachy_sdk.cameras.SR.get_depthmap()
    assert frame is not None
    assert frame.dtype == np.uint16

    frame = reachy_sdk.cameras.SR.get_disparity()
    assert frame is not None
    assert frame.dtype == np.uint16


@pytest.mark.teleop_camera
def test_teleop_camera(reachy_sdk: ReachySDK) -> None:
    assert reachy_sdk.cameras.teleop is not None

    assert reachy_sdk.cameras.teleop.capture()

    frame = reachy_sdk.cameras.teleop.get_frame(CameraView.LEFT)
    assert frame is not None
    assert frame.dtype == np.uint8

    frame_right = reachy_sdk.cameras.teleop.get_frame(CameraView.RIGHT)
    assert frame_right is not None
    assert frame_right.dtype == np.uint8

    # check that we don't return the same frame
    assert not np.array_equal(frame, frame_right)


@pytest.mark.offline
def test_class() -> None:
    cam_teleop = CameraInfo(mxid="fakemixid_tel", name="teleop_head", stereo=True, depth=False)
    cam = Camera(cam_info=cam_teleop, video_stub=None)

    assert cam is not None
