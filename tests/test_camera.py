import numpy as np
import pytest
from reachy2_sdk_api.video_pb2 import CameraInfo, View

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


@pytest.mark.no_camera
@pytest.mark.online
def test_no_camera(reachy_sdk: ReachySDK) -> None:
    list_cam = reachy_sdk.video.get_all_cameras()
    assert (len(list_cam)) == 0

    fake_cam_info = CameraInfo(mxid="fake_mxid", name="fake_name", stereo=True, depth=True)
    assert not reachy_sdk.video.init_camera(fake_cam_info)

    assert not reachy_sdk.video.capture(fake_cam_info)

    assert reachy_sdk.video.get_frame(cam_info=fake_cam_info, view=View.LEFT) is None

    assert reachy_sdk.video.get_depth_frame(cam_info=fake_cam_info, view=View.LEFT) is None

    assert reachy_sdk.video.get_depthmap(cam_info=fake_cam_info) is None

    assert reachy_sdk.video.get_disparity(cam_info=fake_cam_info) is None

    assert not reachy_sdk.video.close_camera(fake_cam_info)


@pytest.mark.sr_camera
@pytest.mark.online
def test_sr_camera(reachy_sdk: ReachySDK) -> None:
    list_cam = reachy_sdk.video.get_all_cameras()

    assert (len(list_cam)) > 0

    cam_info = None
    for c in list_cam:
        if c.name == "other":
            cam_info = c
            break

    assert cam_info is not None

    assert reachy_sdk.video.init_camera(cam_info)

    assert reachy_sdk.video.capture(cam_info)

    frame = reachy_sdk.video.get_frame(cam_info=cam_info)
    assert frame is not None
    assert frame.dtype == np.uint8

    frame = reachy_sdk.video.get_depth_frame(cam_info=cam_info, view=View.LEFT)
    assert frame is not None
    assert frame.dtype == np.uint8

    frame_right = reachy_sdk.video.get_depth_frame(cam_info=cam_info, view=View.RIGHT)
    assert frame_right is not None
    assert frame_right.dtype == np.uint8

    # check that we don't return the same view
    assert not np.array_equal(frame, frame_right)

    frame = reachy_sdk.video.get_depthmap(cam_info=cam_info)
    assert frame is not None
    assert frame.dtype == np.uint16

    frame = reachy_sdk.video.get_disparity(cam_info=cam_info)
    assert frame is not None
    assert frame.dtype == np.uint16

    assert reachy_sdk.video.close_camera(cam_info)

    assert not reachy_sdk.video.capture(cam_info)


@pytest.mark.teleop_camera
@pytest.mark.online
def test_sr_camera(reachy_sdk: ReachySDK) -> None:
    list_cam = reachy_sdk.video.get_all_cameras()

    assert (len(list_cam)) > 0

    cam_info = None
    for c in list_cam:
        if c.name == "teleop_head":
            cam_info = c
            break

    assert cam_info is not None

    assert reachy_sdk.video.init_camera(cam_info)

    assert reachy_sdk.video.capture(cam_info)

    frame = reachy_sdk.video.get_frame(cam_info=cam_info, view=View.LEFT)
    assert frame is not None
    assert frame.dtype == np.uint8

    frame_right = reachy_sdk.video.get_frame(cam_info=cam_info, view=View.RIGHT)
    assert frame_right is not None
    assert frame_right.dtype == np.uint8

    # check that we don't return the same frame
    assert not np.array_equal(frame, frame_right)

    frame = reachy_sdk.video.get_depth_frame(cam_info=cam_info)
    assert frame is None

    frame = reachy_sdk.video.get_depthmap(cam_info=cam_info)
    assert frame is None

    frame = reachy_sdk.video.get_disparity(cam_info=cam_info)
    assert frame is None

    assert reachy_sdk.video.close_camera(cam_info)

    assert not reachy_sdk.video.capture(cam_info)
