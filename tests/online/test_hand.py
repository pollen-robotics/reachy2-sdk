import time

import grpc
import numpy as np
import pytest
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.hand_pb2 import Hand as Hand_proto
from reachy2_sdk_api.hand_pb2 import (
    HandPosition,
    HandState,
    HandTemperatures,
    JointLimits,
    JointsLimits,
    ParallelGripperLimits,
    ParallelGripperPosition,
    Temperatures,
)

from reachy2_sdk.parts.hand import Hand
from reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.online
def test_gripper(reachy_sdk_zeroed: ReachySDK) -> None:
    # https://github.com/pollen-robotics/reachy2_sdk_server/pull/80#issuecomment-1907665226
    # https://github.com/pollen-robotics/reachy2_sdk_server/issues/102

    reachy_sdk_zeroed.r_arm.gripper.close()
    reachy_sdk_zeroed.l_arm.gripper.close()

    time.sleep(1.0)

    # ToDo: these values are not correct
    # assert reachy_sdk_zeroed.r_arm.gripper.opening == 11.64
    assert reachy_sdk_zeroed.l_arm.gripper.opening == 0

    reachy_sdk_zeroed.r_arm.gripper.open()
    reachy_sdk_zeroed.l_arm.gripper.open()

    time.sleep(1.0)

    assert reachy_sdk_zeroed.r_arm.gripper.opening == 0
    assert reachy_sdk_zeroed.l_arm.gripper.opening == 100
