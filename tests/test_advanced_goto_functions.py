import time

import numpy as np
import pytest
from pyquaternion import Quaternion
from reachy2_sdk_api.goto_pb2 import GoalStatus, GoToId

from src.reachy2_sdk.reachy_sdk import ReachySDK

from .test_basic_movements import is_goto_finished


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
    reachy_sdk.cancel_all_goto()
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
    req4 = reachy_sdk_zeroed.head.rotate_to(0, 40, 0, duration=10, interpolation_mode="linear")

    assert len(reachy_sdk_zeroed.head.get_goto_queue()) == 2
    assert reachy_sdk_zeroed.head.get_goto_queue() == [req2, req4]
    assert len(reachy_sdk_zeroed.l_arm.get_goto_queue()) == 0

    cancel = reachy_sdk_zeroed.cancel_goto_by_id(req2)
    assert cancel.ack

    assert len(reachy_sdk_zeroed.head.get_goto_queue()) == 1
    assert reachy_sdk_zeroed.head.get_goto_queue() == [req4]

    cancel = reachy_sdk_zeroed.cancel_all_goto()
    assert cancel.ack


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
    _ = reachy_sdk_zeroed.head.rotate_to(0, 40, 0, duration=10, interpolation_mode="linear")
    _ = reachy_sdk_zeroed.l_arm.goto_joints([15, 10, 20, -50, 10, 10, 20], duration=10, interpolation_mode="linear")
    time.sleep(2)
    cancel = reachy_sdk_zeroed.head.cancel_all_goto()
    assert cancel.ack

    l_arm_position = reachy_sdk_zeroed.l_arm.get_joints_positions()

    # 40*2/10 -> 8° ideally. but timing is not precise
    assert np.isclose(reachy_sdk_zeroed.head.neck.pitch.present_position, 8.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.head.neck.roll.present_position, 0.0)
    assert np.isclose(reachy_sdk_zeroed.head.neck.yaw.present_position, 0.0)
    assert np.isclose(l_arm_position[0], 3.0, atol=1)
    assert np.isclose(l_arm_position[1], 2.0, atol=1)
    assert np.isclose(l_arm_position[2], 4.0, atol=1)
    assert np.isclose(l_arm_position[3], -10.0, atol=1)
    assert np.isclose(l_arm_position[4], 2.0, atol=1)
    assert np.isclose(l_arm_position[5], 2.0, atol=1)
    assert np.isclose(l_arm_position[6], 4.0, atol=1)

    time.sleep(2)
    l_arm_position = reachy_sdk_zeroed.l_arm.get_joints_positions()

    assert np.isclose(reachy_sdk_zeroed.head.neck.pitch.present_position, 8.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.head.neck.roll.present_position, 0.0)
    assert np.isclose(reachy_sdk_zeroed.head.neck.yaw.present_position, 0.0)
    assert np.isclose(l_arm_position[0], 6.0, atol=1)
    assert np.isclose(l_arm_position[1], 4.0, atol=1)
    assert np.isclose(l_arm_position[2], 8.0, atol=1)
    assert np.isclose(l_arm_position[3], -20.0, atol=1)
    assert np.isclose(l_arm_position[4], 4.0, atol=1)
    assert np.isclose(l_arm_position[5], 4.0, atol=1)
    assert np.isclose(l_arm_position[6], 8.0, atol=1)

    req2 = reachy_sdk_zeroed.l_arm.goto_joints([0, 0, 0, 0, 0, 0, 0], duration=10, interpolation_mode="linear")
    req3 = reachy_sdk_zeroed.l_arm.goto_joints([15, 10, 20, -50, 10, 10, 20], duration=10, interpolation_mode="linear")

    assert reachy_sdk_zeroed.l_arm.get_goto_queue() == [req2, req3]

    cancel2 = reachy_sdk_zeroed.l_arm.cancel_all_goto()
    assert cancel2.ack
    assert reachy_sdk_zeroed.l_arm.get_goto_queue() == []

    cancel3 = reachy_sdk_zeroed.cancel_all_goto()
    assert cancel3.ack


@pytest.mark.online
def test_get_goto_playing(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.head.rotate_to(0, 0, -10, duration=3)
    req2 = reachy_sdk_zeroed.l_arm.goto_joints([10, 10, 15, -20, 15, -15, -10], duration=5)
    req3 = reachy_sdk_zeroed.r_arm.goto_joints([0, 10, 20, -40, 10, 10, -15], duration=10)

    req4 = reachy_sdk_zeroed.head.rotate_to(30, 0, 0, duration=5)
    req5 = reachy_sdk_zeroed.l_arm.goto_joints([0, 0, 5, -40, 10, -10, 0], duration=6)
    req6 = reachy_sdk_zeroed.r_arm.goto_joints([15, 15, 0, 0, 25, 20, -5], duration=5)

    assert reachy_sdk_zeroed.head.get_goto_playing() == req1
    assert reachy_sdk_zeroed.l_arm.get_goto_playing() == req2
    assert reachy_sdk_zeroed.r_arm.get_goto_playing() == req3

    while not is_goto_finished(reachy_sdk_zeroed, req1):
        time.sleep(0.1)

    assert reachy_sdk_zeroed.head.get_goto_playing() == req4
    assert reachy_sdk_zeroed.l_arm.get_goto_playing() == req2
    assert reachy_sdk_zeroed.r_arm.get_goto_playing() == req3

    while not is_goto_finished(reachy_sdk_zeroed, req2):
        time.sleep(0.1)

    assert reachy_sdk_zeroed.head.get_goto_playing() == req4
    assert reachy_sdk_zeroed.l_arm.get_goto_playing() == req5
    assert reachy_sdk_zeroed.r_arm.get_goto_playing() == req3

    while not is_goto_finished(reachy_sdk_zeroed, req3):
        time.sleep(0.1)

    assert reachy_sdk_zeroed.head.get_goto_playing() == GoToId(id=-1)
    assert reachy_sdk_zeroed.l_arm.get_goto_playing() == req5
    assert reachy_sdk_zeroed.r_arm.get_goto_playing() == req6

    cancel = reachy_sdk_zeroed.cancel_all_goto()
    assert cancel.ack


@pytest.mark.online
def test_get_goto_state(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.head.rotate_to(0, 0, -10, duration=3)
    req2 = reachy_sdk_zeroed.l_arm.goto_joints([10, 10, 15, -20, 15, -15, -10], duration=5)
    req3 = reachy_sdk_zeroed.r_arm.goto_joints([0, 10, 20, -40, 10, 10, -15], duration=10)

    req4 = reachy_sdk_zeroed.head.rotate_to(30, 0, 0, duration=5)
    req5 = reachy_sdk_zeroed.l_arm.goto_joints([0, 0, 5, -40, 10, -10, 0], duration=6)
    req6 = reachy_sdk_zeroed.r_arm.goto_joints([15, 15, 0, 0, 25, 20, -5], duration=5)

    assert reachy_sdk_zeroed.get_goto_state(req1).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed.get_goto_state(req2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed.get_goto_state(req3).goal_status == GoalStatus.STATUS_EXECUTING

    assert (reachy_sdk_zeroed.get_goto_state(req4).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed.get_goto_state(req4).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?
    assert (reachy_sdk_zeroed.get_goto_state(req5).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed.get_goto_state(req5).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?
    assert (reachy_sdk_zeroed.get_goto_state(req6).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed.get_goto_state(req6).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?

    while not is_goto_finished(reachy_sdk_zeroed, req1):
        time.sleep(0.1)

    assert reachy_sdk_zeroed.get_goto_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed.get_goto_state(req2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed.get_goto_state(req3).goal_status == GoalStatus.STATUS_EXECUTING

    assert reachy_sdk_zeroed.get_goto_state(req4).goal_status == GoalStatus.STATUS_EXECUTING
    assert (reachy_sdk_zeroed.get_goto_state(req5).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed.get_goto_state(req5).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?
    assert (reachy_sdk_zeroed.get_goto_state(req6).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed.get_goto_state(req6).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?

    while not is_goto_finished(reachy_sdk_zeroed, req3):
        time.sleep(0.1)

    cancel_l_arm = reachy_sdk_zeroed.l_arm.cancel_all_goto()
    assert cancel_l_arm.ack

    assert reachy_sdk_zeroed.get_goto_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed.get_goto_state(req2).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed.get_goto_state(req3).goal_status == GoalStatus.STATUS_SUCCEEDED

    assert reachy_sdk_zeroed.get_goto_state(req4).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert (reachy_sdk_zeroed.get_goto_state(req5).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed.get_goto_state(req5).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert reachy_sdk_zeroed.get_goto_state(req6).goal_status == GoalStatus.STATUS_EXECUTING

    cancel = reachy_sdk_zeroed.cancel_all_goto()
    assert cancel.ack

    assert reachy_sdk_zeroed.get_goto_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed.get_goto_state(req2).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed.get_goto_state(req3).goal_status == GoalStatus.STATUS_SUCCEEDED

    assert reachy_sdk_zeroed.get_goto_state(req4).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed.get_goto_state(req5).goal_status == GoalStatus.STATUS_CANCELED
    assert (reachy_sdk_zeroed.get_goto_state(req6).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed.get_goto_state(req6).goal_status == GoalStatus.STATUS_CANCELED
    )


@pytest.mark.online
def test_get_goto_joints_request(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.head.rotate_to(30, 0, 0, duration=5)
    req2 = reachy_sdk_zeroed.l_arm.goto_joints([10, 10, 15, -20, 15, -15, -10], duration=7, interpolation_mode="linear")
    req3 = reachy_sdk_zeroed.r_arm.goto_joints([0, 10, 20, -40, 10, 10, -15], duration=10)

    ans1 = reachy_sdk_zeroed.get_goto_joints_request(req1)
    assert ans1.part == "head"
    assert np.allclose(ans1.goal_positions, [30, 0, 0], atol=1e-01)
    assert ans1.duration == 5
    assert ans1.mode == "minimum_jerk"

    ans2 = reachy_sdk_zeroed.get_goto_joints_request(req2)
    assert ans2.part == "l_arm"
    assert np.allclose(ans2.goal_positions, [10, 10, 15, -20, 15, -15, -10], atol=1e-01)
    assert ans2.duration == 7
    assert ans2.mode == "linear"

    ans3 = reachy_sdk_zeroed.get_goto_joints_request(req3)
    assert ans3.part == "r_arm"
    assert np.allclose(ans3.goal_positions, [0, 10, 20, -40, 10, 10, -15], atol=1e-01)
    assert ans3.duration == 10
    assert ans3.mode == "minimum_jerk"

    cancel = reachy_sdk_zeroed.cancel_all_goto()
    assert cancel.ack


@pytest.mark.online
def test_reachy_home(reachy_sdk_zeroed: ReachySDK) -> None:
    zero_arm = [0, 0, 0, 0, 0, 0, 0]
    zero_head = Quaternion(axis=[1, 0, 0], angle=0.0)

    # Test waiting for part's gotos to end

    req1 = reachy_sdk_zeroed.head.rotate_to(30, 0, 0, duration=4)
    req2 = reachy_sdk_zeroed.r_arm.goto_joints([0, 10, 20, -40, 10, 10, -15], duration=5)
    req3 = reachy_sdk_zeroed.l_arm.goto_joints([10, 10, 15, -20, 15, -15, -10], duration=6)

    time.sleep(2)

    req_h, req_r, req_l = reachy_sdk_zeroed.home()

    assert reachy_sdk_zeroed.get_goto_state(req1).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed.get_goto_state(req2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed.get_goto_state(req3).goal_status == GoalStatus.STATUS_EXECUTING

    assert (reachy_sdk_zeroed.get_goto_state(req_h).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed.get_goto_state(req_h).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?
    assert (reachy_sdk_zeroed.get_goto_state(req_r).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed.get_goto_state(req_r).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?
    assert (reachy_sdk_zeroed.get_goto_state(req_l).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed.get_goto_state(req_l).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?

    while not is_goto_finished(reachy_sdk_zeroed, req2):
        time.sleep(0.1)

    assert reachy_sdk_zeroed.get_goto_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed.get_goto_state(req2).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed.get_goto_state(req3).goal_status == GoalStatus.STATUS_EXECUTING

    assert reachy_sdk_zeroed.get_goto_state(req_h).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed.get_goto_state(req_r).goal_status == GoalStatus.STATUS_EXECUTING
    assert (reachy_sdk_zeroed.get_goto_state(req_l).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed.get_goto_state(req_l).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?

    while not is_goto_finished(reachy_sdk_zeroed, req_l):
        time.sleep(0.1)

    ans_r = reachy_sdk_zeroed.get_goto_joints_request(req_r)
    assert ans_r.part == "r_arm"
    assert np.allclose(ans_r.goal_positions, zero_arm, atol=1e-01)
    assert ans_r.duration == 2
    assert ans_r.mode == "minimum_jerk"

    assert reachy_sdk_zeroed.get_goto_state(req_h).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed.get_goto_state(req_r).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed.get_goto_state(req_l).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert np.isclose(Quaternion.distance(reachy_sdk_zeroed.head.get_orientation(), zero_head), 0, atol=1e-04)
    assert np.allclose(reachy_sdk_zeroed.r_arm.get_joints_positions(), zero_arm, atol=1e-01)
    assert np.allclose(reachy_sdk_zeroed.l_arm.get_joints_positions(), zero_arm, atol=1e-01)

    cancel = reachy_sdk_zeroed.cancel_all_goto()
    assert cancel.ack

    # Test without waiting for part's gotos to end

    req4 = reachy_sdk_zeroed.head.rotate_to(30, 0, 0, duration=4)
    req5 = reachy_sdk_zeroed.r_arm.goto_joints([0, 10, 20, -40, 10, 10, -15], duration=5)
    req6 = reachy_sdk_zeroed.l_arm.goto_joints([10, 10, 15, -20, 15, -15, -10], duration=6)

    time.sleep(2)

    req_h2, req_r2, req_l2 = reachy_sdk_zeroed.home(wait_for_goto_end=False, duration=1, interpolation_mode="linear")

    assert (reachy_sdk_zeroed.get_goto_state(req4).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed.get_goto_state(req4).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert (reachy_sdk_zeroed.get_goto_state(req5).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed.get_goto_state(req5).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert (reachy_sdk_zeroed.get_goto_state(req6).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed.get_goto_state(req6).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert reachy_sdk_zeroed.get_goto_state(req_h2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed.get_goto_state(req_r2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed.get_goto_state(req_l2).goal_status == GoalStatus.STATUS_EXECUTING

    while not is_goto_finished(reachy_sdk_zeroed, req_l2):
        time.sleep(0.1)

    ans_l2 = reachy_sdk_zeroed.get_goto_joints_request(req_l2)
    assert ans_l2.part == "l_arm"
    assert np.allclose(ans_l2.goal_positions, zero_arm, atol=1e-01)
    assert ans_l2.duration == 1
    assert ans_l2.mode == "linear"

    assert reachy_sdk_zeroed.get_goto_state(req_h).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed.get_goto_state(req_r).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed.get_goto_state(req_l).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert np.isclose(
        Quaternion.distance(reachy_sdk_zeroed.head.get_orientation(), zero_head), 0, atol=1e-03
    )  # why not 1e-04 here?
    assert np.allclose(reachy_sdk_zeroed.r_arm.get_joints_positions(), zero_arm, atol=1e-01)
    assert np.allclose(reachy_sdk_zeroed.l_arm.get_joints_positions(), zero_arm, atol=1e-01)