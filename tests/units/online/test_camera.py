from typing import List

import numpy as np
import pytest
from reachy2_sdk_api.video_pb2 import CameraFeatures, View

from reachy2_sdk.media.camera import Camera, CameraView
from reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.online
def test_no_camera(reachy_sdk: ReachySDK) -> None:
    assert reachy_sdk.cameras.teleop is None
    assert reachy_sdk.cameras.depth is None


@pytest.mark.cameras
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
    assert D.shape == (4,)
    assert K.shape == (3, 3)
    assert R.shape == (3, 3)
    assert P.shape == (3, 4)

    height, width, distortion_model, D, K, R, P = reachy_sdk.cameras.teleop.get_parameters(CameraView.RIGHT)

    assert height == 720
    assert width == 960
    assert distortion_model == "equidistant"
    assert D.shape == (4,)
    assert K.shape == (3, 3)
    assert R.shape == (3, 3)
    assert P.shape == (3, 4)


@pytest.mark.cameras
def test_depth_camera(reachy_sdk: ReachySDK) -> None:
    assert reachy_sdk.cameras.depth is not None

    frame, ts = reachy_sdk.cameras.depth.get_frame()
    assert frame is not None
    assert frame.dtype == np.uint8
    assert type(ts) == int
    assert ts > 0

    frame_right, ts_right = reachy_sdk.cameras.depth.get_frame(CameraView.RIGHT)

    # there is only one view
    assert np.array_equal(frame, frame_right)

    height, width, distortion_model, D, K, R, P = reachy_sdk.cameras.depth.get_parameters()

    assert height == 720
    assert width == 1280
    assert distortion_model == "rational_polynomial"
    assert D.shape == (8,)
    assert K.shape == (3, 3)
    assert R.shape == (3, 3)
    assert P.shape == (3, 4)

    frame, ts = reachy_sdk.cameras.depth.get_depth_frame()
    assert frame is not None
    assert frame.dtype == np.uint16
    assert type(ts) == int
    assert ts > 0

    height, width, distortion_model, D, K, R, P = reachy_sdk.cameras.depth.get_parameters(CameraView.DEPTH)

    assert height == 720
    assert width == 1280
    assert distortion_model == "rational_polynomial"
    assert D.shape == (8,)
    assert K.shape == (3, 3)
    assert R.shape == (3, 3)
    assert P.shape == (3, 4)


@pytest.mark.cameras
def test_connect_disconnect(reachy_sdk: ReachySDK) -> None:
    reachy_sdk.disconnect()
    assert reachy_sdk.cameras.depth is None
    assert reachy_sdk.cameras.teleop is None
    assert not reachy_sdk._grpc_connected
    assert not reachy_sdk.cameras._grpc_connected

    reachy_sdk.connect()
    assert reachy_sdk.cameras.depth is not None
    assert reachy_sdk.cameras.teleop is not None
    assert reachy_sdk._grpc_connected
    assert reachy_sdk.cameras._grpc_connected

    reachy_sdk.cameras.disconnect()
    assert reachy_sdk.cameras.depth is None
    assert reachy_sdk.cameras.teleop is None
    assert reachy_sdk._grpc_connected
    assert not reachy_sdk.cameras._grpc_connected

    reachy_sdk.cameras.connect()
    assert reachy_sdk.cameras.depth is not None
    assert reachy_sdk.cameras.teleop is not None
    assert reachy_sdk._grpc_connected
    assert reachy_sdk.cameras._grpc_connected
