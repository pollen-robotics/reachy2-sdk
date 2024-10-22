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
    reachy_sdk_zeroed.r_arm.gripper.close()
    reachy_sdk_zeroed.l_arm.gripper.close()

    time.sleep(1.0)

    assert reachy_sdk_zeroed.r_arm.gripper.opening == 0
    assert reachy_sdk_zeroed.l_arm.gripper.opening == 0

    reachy_sdk_zeroed.r_arm.gripper.open()
    reachy_sdk_zeroed.l_arm.gripper.open()

    time.sleep(1.0)

    assert reachy_sdk_zeroed.r_arm.gripper.opening == 100
    assert reachy_sdk_zeroed.l_arm.gripper.opening == 100


@pytest.mark.online
def test_gripper_goal_position(reachy_sdk_zeroed: ReachySDK) -> None:
    reachy_sdk_zeroed.r_arm.gripper.close()
    reachy_sdk_zeroed.l_arm.gripper.close()

    time.sleep(1.0)

    assert reachy_sdk_zeroed.r_arm.gripper.opening == 0
    assert reachy_sdk_zeroed.l_arm.gripper.opening == 0

    reachy_sdk_zeroed.r_arm.gripper.goal_position = 40
    reachy_sdk_zeroed.l_arm.gripper.goal_position = 70
    reachy_sdk_zeroed.send_goal_positions()

    time.sleep(1.0)

    assert np.isclose(reachy_sdk_zeroed.r_arm.gripper.goal_position, 40, 1e-01)
    assert np.isclose(reachy_sdk_zeroed.l_arm.gripper.goal_position, 70, 1e-01)


@pytest.mark.online
def test_gripper_off(reachy_sdk_zeroed: ReachySDK) -> None:
    reachy_sdk_zeroed.goto_posture("elbow_90")
    reachy_sdk_zeroed.l_arm.gripper.turn_off()
    reachy_sdk_zeroed.l_arm.goto([15, 15, 0, -60, 0, 0, 0])
    time.sleep(3.0)
    assert reachy_sdk_zeroed.l_arm.is_on() == True
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.pitch.present_position, -60, 10)

    reachy_sdk_zeroed.turn_on()


@pytest.mark.online
def test_gripper_is_moving(reachy_sdk_zeroed: ReachySDK) -> None:
    assert not reachy_sdk_zeroed.r_arm.gripper.is_moving()

    reachy_sdk_zeroed.r_arm.gripper.close()
    assert reachy_sdk_zeroed.r_arm.gripper.is_moving()
    time.sleep(1.0)
    assert not reachy_sdk_zeroed.r_arm.gripper.is_moving()
    assert np.isclose(reachy_sdk_zeroed.r_arm.gripper.opening, 0, 1e-01)

    reachy_sdk_zeroed.r_arm.gripper.open()
    assert reachy_sdk_zeroed.r_arm.gripper.is_moving()
    time.sleep(1.0)
    assert not reachy_sdk_zeroed.r_arm.gripper.is_moving()
    assert np.isclose(reachy_sdk_zeroed.r_arm.gripper.opening, 100, 1e-01)

    reachy_sdk_zeroed.l_arm.gripper.close()
    assert reachy_sdk_zeroed.l_arm.gripper.is_moving()
    time.sleep(1.0)
    assert not reachy_sdk_zeroed.l_arm.gripper.is_moving()
    assert np.isclose(reachy_sdk_zeroed.l_arm.gripper.opening, 0, 1e-01)

    reachy_sdk_zeroed.l_arm.gripper.open()
    assert reachy_sdk_zeroed.l_arm.gripper.is_moving()
    time.sleep(1.0)
    assert not reachy_sdk_zeroed.l_arm.gripper.is_moving()
    assert np.isclose(reachy_sdk_zeroed.l_arm.gripper.opening, 100, 1e-01)

    reachy_sdk_zeroed.r_arm.gripper.goal_position = 50
    reachy_sdk_zeroed.l_arm.gripper.goal_position = 50
    reachy_sdk_zeroed.send_goal_positions()
    assert reachy_sdk_zeroed.l_arm.gripper.is_moving()
    assert reachy_sdk_zeroed.r_arm.gripper.is_moving()
    time.sleep(1.0)
    assert not reachy_sdk_zeroed.l_arm.gripper.is_moving()
    assert not reachy_sdk_zeroed.r_arm.gripper.is_moving()
