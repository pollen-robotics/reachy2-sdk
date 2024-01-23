import time

import numpy as np
import pytest

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


@pytest.fixture
def reachy_sdk_zeroed(reachy_sdk: ReachySDK) -> ReachySDK:
    for joint in reachy_sdk.joints.values():
        joint.goal_position = 0
        time.sleep(0.01)

    time.sleep(2)

    return reachy_sdk


@pytest.mark.online
def test_cancel_goto_by_id(reachy_sdk_zeroed: ReachySDK) -> None:
    req = reachy_sdk_zeroed.head.rotate_to(0, 40, 0, duration=10, interpolation_mode="linear")
    time.sleep(2)
    cancel = reachy_sdk_zeroed.cancel_goto_by_id(req)
    assert cancel.ack

    # 40*2/10 -> 8° ideally. but timing is not precise
    assert np.isclose(reachy_sdk_zeroed.head.neck.pitch.present_position, 8, atol=1)
    assert np.isclose(reachy_sdk_zeroed.head.neck.roll.present_position, 0)
    assert np.isclose(reachy_sdk_zeroed.head.neck.yaw.present_position, 0)

    req2 = reachy_sdk_zeroed.l_arm.goto_joints([15, 10, 20, -50, 10, 10, 20], duration=10, interpolation_mode="linear")
    time.sleep(2)
    cancel2 = reachy_sdk_zeroed.cancel_goto_by_id(req2)
    assert cancel2.ack
    assert np.isclose(reachy_sdk_zeroed.l_arm.shoulder.pitch.present_position, 3, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.shoulder.roll.present_position, 2.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.yaw.present_position, 4.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.pitch.present_position, -10, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.roll.present_position, 2.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.pitch.present_position, 2.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.yaw.present_position, 4.0, atol=1)


@pytest.mark.online
def test_goto_queue(reachy_sdk_zeroed: ReachySDK) -> None:
    _ = reachy_sdk_zeroed.head.rotate_to(0, 40, 0, duration=10, interpolation_mode="linear")
    req2 = reachy_sdk_zeroed.head.rotate_to(20, 0, 0, duration=10, interpolation_mode="linear")
    _ = reachy_sdk_zeroed.l_arm.goto_joints([15, 10, 20, -50, 10, 10, 20], duration=10, interpolation_mode="linear")
    _ = reachy_sdk_zeroed.head.rotate_to(0, 40, 0, duration=10, interpolation_mode="linear")

    assert len(reachy_sdk_zeroed.head.get_goto_queue()) == 2
    assert len(reachy_sdk_zeroed.l_arm.get_goto_queue()) == 0

    cancel = reachy_sdk_zeroed.cancel_goto_by_id(req2)
    assert cancel.ack

    assert len(reachy_sdk_zeroed.head.get_goto_queue()) == 1


@pytest.mark.online
def test_cancel_all_goto(reachy_sdk_zeroed: ReachySDK) -> None:
    _ = reachy_sdk_zeroed.head.rotate_to(0, 40, 0, duration=10, interpolation_mode="linear")
    _ = reachy_sdk_zeroed.l_arm.goto_joints([15, 10, 20, -50, 10, 10, 20], duration=10, interpolation_mode="linear")
    time.sleep(2)
    cancel = reachy_sdk_zeroed.cancel_all_goto()
    assert cancel.ack

    # 40*2/10 -> 8° ideally. but timing is not precise
    assert np.isclose(reachy_sdk_zeroed.head.neck.pitch.present_position, 8.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.head.neck.roll.present_position, 0.0)
    assert np.isclose(reachy_sdk_zeroed.head.neck.yaw.present_position, 0.0)
    assert np.isclose(reachy_sdk_zeroed.l_arm.shoulder.pitch.present_position, 3.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.shoulder.roll.present_position, 2.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.yaw.present_position, 4.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.pitch.present_position, -10.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.roll.present_position, 2.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.pitch.present_position, 2.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.yaw.present_position, 4.0, atol=1)


@pytest.mark.online
def test_cancel_part_all_goto(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.head.rotate_to(0, 40, 0, duration=10, interpolation_mode="linear")
    req2 = reachy_sdk_zeroed.l_arm.goto_joints([15, 10, 20, -50, 10, 10, 20], duration=10, interpolation_mode="linear")
    time.sleep(2)
    cancel = reachy_sdk_zeroed.head.cancel_all_goto()
    assert cancel.ack

    # 40*2/10 -> 8° ideally. but timing is not precise
    assert np.isclose(reachy_sdk_zeroed.head.neck.pitch.present_position, 8.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.head.neck.roll.present_position, 0.0)
    assert np.isclose(reachy_sdk_zeroed.head.neck.yaw.present_position, 0.0)
    assert np.isclose(reachy_sdk_zeroed.l_arm.shoulder.pitch.present_position, 3.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.shoulder.roll.present_position, 2.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.yaw.present_position, 4.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.pitch.present_position, -10.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.roll.present_position, 2.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.pitch.present_position, 2.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.yaw.present_position, 4.0, atol=1)

    time.sleep(2)
    assert np.isclose(reachy_sdk_zeroed.head.neck.pitch.present_position, 8.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.head.neck.roll.present_position, 0.0)
    assert np.isclose(reachy_sdk_zeroed.head.neck.yaw.present_position, 0.0)
    assert np.isclose(reachy_sdk_zeroed.l_arm.shoulder.pitch.present_position, 6.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.shoulder.roll.present_position, 4.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.yaw.present_position, 8.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.pitch.present_position, -20.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.roll.present_position, 4.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.pitch.present_position, 4.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.yaw.present_position, 8.0, atol=1)
