from typing import List

import numpy as np
import pytest
from reachy2_sdk_api.video_pb2 import CameraFeatures, View

from reachy2_sdk.media.camera import Camera, CameraView
from reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.online
def test_no_camera(reachy_sdk: ReachySDK) -> None:
    assert reachy_sdk.cameras.teleop is not None

    res = reachy_sdk.cameras.teleop.get_frame()

    assert res is None


"""
@pytest.mark.sr_camera
def test_sr_camera(reachy_sdk: ReachySDK) -> None:
    assert reachy_sdk._cameras.SR is not None

    assert reachy_sdk._cameras.SR.capture()

    frame = reachy_sdk._cameras.SR.get_frame()
    assert frame is not None
    assert frame.dtype == np.uint8

    frame = reachy_sdk._cameras.SR.get_depth_frame(CameraView.LEFT)
    assert frame is not None
    assert frame.dtype == np.uint8

    frame_right = reachy_sdk._cameras.SR.get_depth_frame(CameraView.RIGHT)
    assert frame_right is not None
    assert frame_right.dtype == np.uint8

    # check that we don't return the same view
    assert not np.array_equal(frame, frame_right)

    frame = reachy_sdk._cameras.SR.get_depthmap()
    assert frame is not None
    assert frame.dtype == np.uint16

    frame = reachy_sdk._cameras.SR.get_disparity()
    assert frame is not None
    assert frame.dtype == np.uint16
"""


@pytest.mark.teleop_camera
def test_teleop_camera(reachy_sdk: ReachySDK) -> None:
    assert reachy_sdk.cameras.teleop is not None

    frame, ts = reachy_sdk.cameras.teleop.get_frame(CameraView.LEFT)
    assert frame is not None
    assert frame.dtype == np.uint8
    assert type(ts) == int
    assert ts > 0

    frame_right, ts_right = reachy_sdk.cameras.teleop.get_frame(CameraView.RIGHT)
    assert frame_right is not None
    assert frame_right.dtype == np.uint8
    assert type(ts_right) == int
    assert ts_right > 0

    # check that we don't return the same frame
    assert not np.array_equal(frame, frame_right)

    height, width, distortion_model, D, K, R, P = reachy_sdk.cameras.teleop.get_parameters(CameraView.LEFT)

    assert height == 720
    assert width == 960
    assert distortion_model == "equidistant"
    assert len(D) == 4
    assert len(K) == 9
    assert len(R) == 9
    assert len(P) == 12

    height, width, distortion_model, D, K, R, P = reachy_sdk.cameras.teleop.get_parameters(CameraView.RIGHT)

    assert height == 720
    assert width == 960
    assert distortion_model == "equidistant"
    assert len(D) == 4
    assert len(K) == 9
    assert len(R) == 9
    assert len(P) == 12
