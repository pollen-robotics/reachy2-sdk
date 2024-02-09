import time

import numpy as np
import pytest
from pyquaternion import Quaternion
from reachy2_sdk_api.goto_pb2 import GoalStatus, GoToId

from src.reachy2_sdk.reachy_sdk import ReachySDK

from .test_basic_movements import is_goto_finished


@pytest.mark.online
def test_cancel_move_by_id(reachy_sdk_zeroed: ReachySDK) -> None:
    req = reachy_sdk_zeroed.head.rotate_to(0, 40, 0, duration=10, interpolation_mode="linear")
    time.sleep(2)
    cancel = reachy_sdk_zeroed.cancel_move_by_id(req)
    assert cancel.ack

    # 40*2/10 -> 8° ideally. but timing is not precise
    assert np.isclose(reachy_sdk_zeroed.head.neck.pitch.present_position, 8, atol=1)
    assert np.isclose(reachy_sdk_zeroed.head.neck.roll.present_position, 0)
    assert np.isclose(reachy_sdk_zeroed.head.neck.yaw.present_position, 0)

    req2 = reachy_sdk_zeroed.l_arm.goto_joints([15, 10, 20, -50, 10, 10, 20], duration=10, interpolation_mode="linear")
    time.sleep(2)
    cancel2 = reachy_sdk_zeroed.cancel_move_by_id(req2)
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

    assert len(reachy_sdk_zeroed.head.get_moves_queue()) == 2
    assert reachy_sdk_zeroed.head.get_moves_queue() == [req2, req4]
    assert len(reachy_sdk_zeroed.l_arm.get_moves_queue()) == 0

    cancel = reachy_sdk_zeroed.cancel_move_by_id(req2)
    assert cancel.ack

    assert len(reachy_sdk_zeroed.head.get_moves_queue()) == 1
    assert reachy_sdk_zeroed.head.get_moves_queue() == [req4]

    cancel = reachy_sdk_zeroed.cancel_all_moves()
    assert cancel.ack


@pytest.mark.online
def test_cancel_all_moves(reachy_sdk_zeroed: ReachySDK) -> None:
    _ = reachy_sdk_zeroed.head.rotate_to(0, 40, 0, duration=10, interpolation_mode="linear")
    _ = reachy_sdk_zeroed.l_arm.goto_joints([15, 10, 20, -50, 10, 10, 20], duration=10, interpolation_mode="linear")
    time.sleep(2)
    cancel = reachy_sdk_zeroed.cancel_all_moves()
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
    cancel = reachy_sdk_zeroed.head.cancel_all_moves()
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

    assert reachy_sdk_zeroed.l_arm.get_moves_queue() == [req2, req3]

    cancel2 = reachy_sdk_zeroed.l_arm.cancel_all_moves()
    assert cancel2.ack
    assert reachy_sdk_zeroed.l_arm.get_moves_queue() == []

    cancel3 = reachy_sdk_zeroed.cancel_all_moves()
    assert cancel3.ack


@pytest.mark.online
def test_get_move_playing(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.head.rotate_to(0, 0, -10, duration=3)
    req2 = reachy_sdk_zeroed.l_arm.goto_joints([10, 10, 15, -20, 15, -15, -10], duration=5)
    req3 = reachy_sdk_zeroed.r_arm.goto_joints([0, 10, 20, -40, 10, 10, -15], duration=10)

    req4 = reachy_sdk_zeroed.head.rotate_to(30, 0, 0, duration=5)
    req5 = reachy_sdk_zeroed.l_arm.goto_joints([0, 0, 5, -40, 10, -10, 0], duration=6)
    req6 = reachy_sdk_zeroed.r_arm.goto_joints([15, 15, 0, 0, 25, 20, -5], duration=5)

    assert reachy_sdk_zeroed.head.get_move_playing() == req1
    assert reachy_sdk_zeroed.l_arm.get_move_playing() == req2
    assert reachy_sdk_zeroed.r_arm.get_move_playing() == req3

    while not is_goto_finished(reachy_sdk_zeroed, req1):
        time.sleep(0.1)

    assert reachy_sdk_zeroed.head.get_move_playing() == req4
    assert reachy_sdk_zeroed.l_arm.get_move_playing() == req2
    assert reachy_sdk_zeroed.r_arm.get_move_playing() == req3

    while not is_goto_finished(reachy_sdk_zeroed, req2):
        time.sleep(0.1)

    assert reachy_sdk_zeroed.head.get_move_playing() == req4
    assert reachy_sdk_zeroed.l_arm.get_move_playing() == req5
    assert reachy_sdk_zeroed.r_arm.get_move_playing() == req3

    while not is_goto_finished(reachy_sdk_zeroed, req3):
        time.sleep(0.1)

    assert reachy_sdk_zeroed.head.get_move_playing() == GoToId(id=-1)
    assert reachy_sdk_zeroed.l_arm.get_move_playing() == req5
    assert reachy_sdk_zeroed.r_arm.get_move_playing() == req6

    cancel = reachy_sdk_zeroed.cancel_all_moves()
    assert cancel.ack


@pytest.mark.online
def test__get_move_state(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.head.rotate_to(0, 0, -10, duration=3)
    req2 = reachy_sdk_zeroed.l_arm.goto_joints([10, 10, 15, -20, 15, -15, -10], duration=5)
    req3 = reachy_sdk_zeroed.r_arm.goto_joints([0, 10, 20, -40, 10, 10, -15], duration=10)

    req4 = reachy_sdk_zeroed.head.rotate_to(30, 0, 0, duration=5)
    req5 = reachy_sdk_zeroed.l_arm.goto_joints([0, 0, 5, -40, 10, -10, 0], duration=6)
    req6 = reachy_sdk_zeroed.r_arm.goto_joints([15, 15, 0, 0, 25, 20, -5], duration=5)

    assert reachy_sdk_zeroed._get_move_state(req1).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_move_state(req2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_move_state(req3).goal_status == GoalStatus.STATUS_EXECUTING

    assert (reachy_sdk_zeroed._get_move_state(req4).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_move_state(req4).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?
    assert (reachy_sdk_zeroed._get_move_state(req5).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_move_state(req5).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?
    assert (reachy_sdk_zeroed._get_move_state(req6).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_move_state(req6).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?

    while not is_goto_finished(reachy_sdk_zeroed, req1):
        time.sleep(0.1)

    assert reachy_sdk_zeroed._get_move_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_move_state(req3).goal_status == GoalStatus.STATUS_EXECUTING

    assert reachy_sdk_zeroed._get_move_state(req4).goal_status == GoalStatus.STATUS_EXECUTING
    assert (reachy_sdk_zeroed._get_move_state(req5).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_move_state(req5).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?
    assert (reachy_sdk_zeroed._get_move_state(req6).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_move_state(req6).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?

    while not is_goto_finished(reachy_sdk_zeroed, req3):
        time.sleep(0.1)

    cancel_l_arm = reachy_sdk_zeroed.l_arm.cancel_all_moves()
    assert cancel_l_arm.ack

    assert reachy_sdk_zeroed._get_move_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req2).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req3).goal_status == GoalStatus.STATUS_SUCCEEDED

    assert reachy_sdk_zeroed._get_move_state(req4).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert (reachy_sdk_zeroed._get_move_state(req5).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed._get_move_state(req5).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert reachy_sdk_zeroed._get_move_state(req6).goal_status == GoalStatus.STATUS_EXECUTING

    cancel = reachy_sdk_zeroed.cancel_all_moves()
    assert cancel.ack

    assert reachy_sdk_zeroed._get_move_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req2).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req3).goal_status == GoalStatus.STATUS_SUCCEEDED

    assert reachy_sdk_zeroed._get_move_state(req4).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req5).goal_status == GoalStatus.STATUS_CANCELED
    assert (reachy_sdk_zeroed._get_move_state(req6).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed._get_move_state(req6).goal_status == GoalStatus.STATUS_CANCELED
    )


@pytest.mark.online
def test_get_move_joints_request(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.head.rotate_to(30, 0, 0, duration=5)
    req2 = reachy_sdk_zeroed.l_arm.goto_joints([10, 10, 15, -20, 15, -15, -10], duration=7, interpolation_mode="linear")
    req3 = reachy_sdk_zeroed.r_arm.goto_joints([0, 10, 20, -40, 10, 10, -15], duration=10)

    ans1 = reachy_sdk_zeroed.get_move_joints_request(req1)
    assert ans1.part == "head"
    assert np.allclose(ans1.goal_positions, [30, 0, 0], atol=1e-01)
    assert ans1.duration == 5
    assert ans1.mode == "minimum_jerk"

    ans2 = reachy_sdk_zeroed.get_move_joints_request(req2)
    assert ans2.part == "l_arm"
    assert np.allclose(ans2.goal_positions, [10, 10, 15, -20, 15, -15, -10], atol=1e-01)
    assert ans2.duration == 7
    assert ans2.mode == "linear"

    ans3 = reachy_sdk_zeroed.get_move_joints_request(req3)
    assert ans3.part == "r_arm"
    assert np.allclose(ans3.goal_positions, [0, 10, 20, -40, 10, 10, -15], atol=1e-01)
    assert ans3.duration == 10
    assert ans3.mode == "minimum_jerk"

    cancel = reachy_sdk_zeroed.cancel_all_moves()
    assert cancel.ack


@pytest.mark.online
def test_reachy_set_pose(reachy_sdk_zeroed: ReachySDK) -> None:
    zero_arm = [0, 0, 0, 0, 0, 0, 0]
    elbow_90_arm = [0, 0, 0, -90, 0, 0, 0]
    zero_head = Quaternion(axis=[1, 0, 0], angle=0.0)

    # Test waiting for part's gotos to end

    req1 = reachy_sdk_zeroed.head.rotate_to(30, 0, 0, duration=4)
    req2 = reachy_sdk_zeroed.r_arm.goto_joints([0, 10, 20, -40, 10, 10, -15], duration=5)
    req3 = reachy_sdk_zeroed.l_arm.goto_joints([10, 10, 15, -20, 15, -15, -10], duration=6)

    time.sleep(2)

    req_h, req_r, req_l = reachy_sdk_zeroed.set_pose()

    assert reachy_sdk_zeroed._get_move_state(req1).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_move_state(req2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_move_state(req3).goal_status == GoalStatus.STATUS_EXECUTING

    assert (reachy_sdk_zeroed._get_move_state(req_h).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_move_state(req_h).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?
    assert (reachy_sdk_zeroed._get_move_state(req_r).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_move_state(req_r).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?
    assert (reachy_sdk_zeroed._get_move_state(req_l).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_move_state(req_l).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?

    while not is_goto_finished(reachy_sdk_zeroed, req2):
        time.sleep(0.1)

    assert reachy_sdk_zeroed._get_move_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req2).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req3).goal_status == GoalStatus.STATUS_EXECUTING

    assert reachy_sdk_zeroed._get_move_state(req_h).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_move_state(req_r).goal_status == GoalStatus.STATUS_EXECUTING
    assert (reachy_sdk_zeroed._get_move_state(req_l).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_move_state(req_l).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?

    while not is_goto_finished(reachy_sdk_zeroed, req_l):
        time.sleep(0.1)

    ans_r = reachy_sdk_zeroed.get_move_joints_request(req_r)
    assert ans_r.part == "r_arm"
    assert np.allclose(ans_r.goal_positions, zero_arm, atol=1e-01)
    assert ans_r.duration == 2
    assert ans_r.mode == "minimum_jerk"

    assert reachy_sdk_zeroed._get_move_state(req_h).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req_r).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req_l).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert np.isclose(Quaternion.distance(reachy_sdk_zeroed.head.get_orientation(), zero_head), 0, atol=1e-04)
    assert np.allclose(reachy_sdk_zeroed.r_arm.get_joints_positions(), zero_arm, atol=1e-01)
    assert np.allclose(reachy_sdk_zeroed.l_arm.get_joints_positions(), zero_arm, atol=1e-01)

    cancel = reachy_sdk_zeroed.cancel_all_moves()
    assert cancel.ack

    # Test without waiting for part's gotos to end

    req4 = reachy_sdk_zeroed.head.rotate_to(30, 0, 0, duration=4)
    req5 = reachy_sdk_zeroed.r_arm.goto_joints([0, 10, 20, -40, 10, 10, -15], duration=5)
    req6 = reachy_sdk_zeroed.l_arm.goto_joints([10, 10, 15, -20, 15, -15, -10], duration=6)

    time.sleep(2)

    req_h2, req_r2, req_l2 = reachy_sdk_zeroed.set_pose(
        "zero", wait_for_moves_end=False, duration=1, interpolation_mode="linear"
    )

    assert (reachy_sdk_zeroed._get_move_state(req4).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed._get_move_state(req4).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert (reachy_sdk_zeroed._get_move_state(req5).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed._get_move_state(req5).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert (reachy_sdk_zeroed._get_move_state(req6).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed._get_move_state(req6).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert reachy_sdk_zeroed._get_move_state(req_h2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_move_state(req_r2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_move_state(req_l2).goal_status == GoalStatus.STATUS_EXECUTING

    while not is_goto_finished(reachy_sdk_zeroed, req_l2):
        time.sleep(0.1)

    ans_l2 = reachy_sdk_zeroed.get_move_joints_request(req_l2)
    assert ans_l2.part == "l_arm"
    assert np.allclose(ans_l2.goal_positions, zero_arm, atol=1e-01)
    assert ans_l2.duration == 1
    assert ans_l2.mode == "linear"

    assert reachy_sdk_zeroed._get_move_state(req_h2).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req_r2).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req_l2).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert np.isclose(
        Quaternion.distance(reachy_sdk_zeroed.head.get_orientation(), zero_head), 0, atol=1e-03
    )  # why not 1e-04 here?
    assert np.allclose(reachy_sdk_zeroed.r_arm.get_joints_positions(), zero_arm, atol=1e-01)
    assert np.allclose(reachy_sdk_zeroed.l_arm.get_joints_positions(), zero_arm, atol=1e-01)

    # Test with 'elbow_90' instead of 'zero'

    req7 = reachy_sdk_zeroed.head.rotate_to(30, 0, 0, duration=4)
    req8 = reachy_sdk_zeroed.r_arm.goto_joints([0, 10, 20, -40, 10, 10, -15], duration=5)
    req9 = reachy_sdk_zeroed.l_arm.goto_joints([10, 10, 15, -20, 15, -15, -10], duration=6)

    time.sleep(2)

    req_h3, req_r3, req_l3 = reachy_sdk_zeroed.set_pose("elbow_90", wait_for_moves_end=False, duration=2)

    assert (reachy_sdk_zeroed._get_move_state(req7).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed._get_move_state(req7).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert (reachy_sdk_zeroed._get_move_state(req8).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed._get_move_state(req8).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert (reachy_sdk_zeroed._get_move_state(req9).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed._get_move_state(req9).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert reachy_sdk_zeroed._get_move_state(req_h3).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_move_state(req_r3).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_move_state(req_l3).goal_status == GoalStatus.STATUS_EXECUTING

    while not is_goto_finished(reachy_sdk_zeroed, req_l3):
        time.sleep(0.1)

    ans_l3 = reachy_sdk_zeroed.get_move_joints_request(req_l3)
    assert ans_l3.part == "l_arm"
    assert np.allclose(ans_l3.goal_positions, elbow_90_arm, atol=1e-01)
    assert ans_l3.duration == 2
    assert ans_l3.mode == "minimum_jerk"

    assert reachy_sdk_zeroed._get_move_state(req_h3).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req_r3).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req_l3).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert np.isclose(
        Quaternion.distance(reachy_sdk_zeroed.head.get_orientation(), zero_head), 0, atol=1e-03
    )  # why not 1e-04 here?
    assert np.allclose(reachy_sdk_zeroed.r_arm.get_joints_positions(), elbow_90_arm, atol=1e-01)
    assert np.allclose(reachy_sdk_zeroed.l_arm.get_joints_positions(), elbow_90_arm, atol=1e-01)


@pytest.mark.online
def test_is_move_finished(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.head.rotate_to(30, 0, 0, duration=2)
    req2 = reachy_sdk_zeroed.l_arm.goto_joints([10, 10, 15, -20, 15, -15, -10], duration=3, interpolation_mode="linear")
    req3 = reachy_sdk_zeroed.r_arm.goto_joints([0, 10, 20, -40, 10, 10, -15], duration=4)

    time.sleep(1)
    assert not reachy_sdk_zeroed.is_move_finished(req1)
    assert not reachy_sdk_zeroed.is_move_finished(req2)
    assert not reachy_sdk_zeroed.is_move_finished(req3)

    req4 = reachy_sdk_zeroed.head.rotate_to(0, 0, 0, duration=1)
    req5 = reachy_sdk_zeroed.l_arm.goto_joints([0, 0, 0, 0, 0, 0, 0], duration=1)
    req6 = reachy_sdk_zeroed.r_arm.goto_joints([0, 0, 0, 0, 0, 0, 0], duration=4)

    time.sleep(1)
    assert reachy_sdk_zeroed._get_move_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_move_state(req3).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed.is_move_finished(req1)
    assert not reachy_sdk_zeroed.is_move_finished(req2)
    assert not reachy_sdk_zeroed.is_move_finished(req3)
    assert not reachy_sdk_zeroed.is_move_finished(req4)
    assert not reachy_sdk_zeroed.is_move_finished(req5)
    assert not reachy_sdk_zeroed.is_move_finished(req6)

    cancel = reachy_sdk_zeroed.l_arm.cancel_all_moves()
    assert cancel.ack
    assert reachy_sdk_zeroed.is_move_finished(req2)

    time.sleep(2)
    assert reachy_sdk_zeroed._get_move_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req2).goal_status == GoalStatus.STATUS_CANCELED
    assert reachy_sdk_zeroed._get_move_state(req3).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed.is_move_finished(req1)
    assert reachy_sdk_zeroed.is_move_finished(req2)
    assert reachy_sdk_zeroed.is_move_finished(req3)
    assert reachy_sdk_zeroed.is_move_finished(req4)
    assert reachy_sdk_zeroed.is_move_finished(req5)
    assert not reachy_sdk_zeroed.is_move_finished(req6)


@pytest.mark.online
def test_is_move_playing(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.head.rotate_to(30, 0, 0, duration=2)
    req2 = reachy_sdk_zeroed.l_arm.goto_joints([10, 10, 15, -20, 15, -15, -10], duration=3, interpolation_mode="linear")
    req3 = reachy_sdk_zeroed.r_arm.goto_joints([0, 10, 20, -40, 10, 10, -15], duration=4)

    time.sleep(1)
    assert reachy_sdk_zeroed.is_move_playing(req1)
    assert reachy_sdk_zeroed.is_move_playing(req2)
    assert reachy_sdk_zeroed.is_move_playing(req3)

    req4 = reachy_sdk_zeroed.head.rotate_to(0, 0, 0, duration=1)
    req5 = reachy_sdk_zeroed.l_arm.goto_joints([0, 0, 0, 0, 0, 0, 0], duration=1)
    req6 = reachy_sdk_zeroed.r_arm.goto_joints([0, 0, 0, 0, 0, 0, 0], duration=4)

    time.sleep(1)
    assert reachy_sdk_zeroed._get_move_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_move_state(req3).goal_status == GoalStatus.STATUS_EXECUTING
    assert not reachy_sdk_zeroed.is_move_playing(req1)
    assert reachy_sdk_zeroed.is_move_playing(req2)
    assert reachy_sdk_zeroed.is_move_playing(req3)
    assert reachy_sdk_zeroed.is_move_playing(req4)
    assert not reachy_sdk_zeroed.is_move_playing(req5)
    assert not reachy_sdk_zeroed.is_move_playing(req6)

    cancel = reachy_sdk_zeroed.l_arm.cancel_all_moves()
    assert cancel.ack
    assert not reachy_sdk_zeroed.is_move_playing(req2)

    time.sleep(2)
    assert reachy_sdk_zeroed._get_move_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_move_state(req2).goal_status == GoalStatus.STATUS_CANCELED
    assert reachy_sdk_zeroed._get_move_state(req3).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert not reachy_sdk_zeroed.is_move_playing(req1)
    assert not reachy_sdk_zeroed.is_move_playing(req2)
    assert not reachy_sdk_zeroed.is_move_playing(req3)
    assert not reachy_sdk_zeroed.is_move_playing(req4)
    assert not reachy_sdk_zeroed.is_move_playing(req5)
    assert reachy_sdk_zeroed.is_move_playing(req6)
