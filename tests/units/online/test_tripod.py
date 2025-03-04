import time

import pytest
from google.protobuf.wrappers_pb2 import FloatValue
from reachy2_sdk_api.part_pb2 import PartId
from reachy2_sdk_api.tripod_pb2 import Tripod as Tripod_proto
from reachy2_sdk_api.tripod_pb2 import (
    TripodAxis,
    TripodDescription,
    TripodJoint,
    TripodJointState,
    TripodState,
)

from reachy2_sdk.parts.tripod import Tripod
from reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.online
def test_tripod(reachy_sdk_zeroed: ReachySDK) -> None:
    assert reachy_sdk_zeroed.tripod is not None

    reachy_sdk_zeroed.tripod.set_height(1.1)
    time.sleep(0.2)
    assert reachy_sdk_zeroed.tripod.height == 1.1

    reachy_sdk_zeroed.tripod.set_height(1.16)
    time.sleep(0.2)
    assert reachy_sdk_zeroed.tripod.height == 1.16

    height_min, height_max = reachy_sdk_zeroed.tripod.get_limits()
    assert height_min < height_max

    reachy_sdk_zeroed.tripod.set_height(height_min - 0.1)
    time.sleep(0.2)
    assert reachy_sdk_zeroed.tripod.height == height_min

    reachy_sdk_zeroed.tripod.set_height(height_max + 0.1)
    time.sleep(0.2)
    assert reachy_sdk_zeroed.tripod.height == height_max

    reachy_sdk_zeroed.tripod.reset_height()
    time.sleep(0.2)
    assert reachy_sdk_zeroed.tripod.height == height_min
