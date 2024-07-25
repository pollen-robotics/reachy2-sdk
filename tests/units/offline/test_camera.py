import pytest
from reachy2_sdk_api.video_pb2 import CameraFeatures

from reachy2_sdk.media.camera import Camera


@pytest.mark.offline
def test_class() -> None:
    cam_teleop = CameraFeatures(name="teleop_head", stereo=True, depth=False)
    cam = Camera(cam_info=cam_teleop, video_stub=None)

    assert cam is not None
