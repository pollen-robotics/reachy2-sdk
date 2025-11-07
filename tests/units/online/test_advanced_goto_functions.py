import time

import numpy as np
import pytest
from pyquaternion import Quaternion
from reachy2_sdk_api.goto_pb2 import GoalStatus, GoToId

from reachy2_sdk.reachy_sdk import ReachySDK
from reachy2_sdk.utils.utils import matrix_from_euler_angles

from .test_basic_movements import build_pose_matrix, is_goto_finished


@pytest.mark.online
def test_cancel_goto_by_id(reachy_sdk_zeroed: ReachySDK) -> None:
    req = reachy_sdk_zeroed.head.goto([0, 40, 0], duration=10, interpolation_mode="linear")
    time.sleep(2)
    cancel = reachy_sdk_zeroed.cancel_goto_by_id(req)
    assert cancel.ack

    # 40*2/10 -> 8° ideally. but timing is not precise
    assert np.isclose(reachy_sdk_zeroed.head.neck.pitch.present_position, 8, atol=1)
    assert np.isclose(reachy_sdk_zeroed.head.neck.roll.present_position, 0)
    assert np.isclose(reachy_sdk_zeroed.head.neck.yaw.present_position, 0)

    req2 = reachy_sdk_zeroed.l_arm.goto([15, 10, 20, -50, 10, 10, 20], duration=10, interpolation_mode="linear")
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

    with pytest.raises(TypeError):
        reachy_sdk_zeroed.cancel_goto_by_id(3)


@pytest.mark.online
def test_goto_queue(reachy_sdk_zeroed: ReachySDK) -> None:
    _ = reachy_sdk_zeroed.head.goto([0, 40, 0], duration=10, interpolation_mode="linear")
    req2 = reachy_sdk_zeroed.head.goto([20, 0, 0], duration=10, interpolation_mode="linear")
    _ = reachy_sdk_zeroed.l_arm.goto([15, 10, 20, -50, 10, 10, 20], duration=10, interpolation_mode="linear")
    req4 = reachy_sdk_zeroed.head.goto([0, 40, 0], duration=10, interpolation_mode="linear")

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
    _ = reachy_sdk_zeroed.head.goto([0, 40, 0], duration=10, interpolation_mode="linear")
    _ = reachy_sdk_zeroed.l_arm.goto([15, 10, 20, -50, 10, 10, 20], duration=10, interpolation_mode="linear")
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
    _ = reachy_sdk_zeroed.head.goto([0, 40, 0], duration=10, interpolation_mode="linear")
    _ = reachy_sdk_zeroed.l_arm.goto([15, 10, 20, -50, 10, 10, 20], duration=10, interpolation_mode="linear")
    time.sleep(2)
    cancel = reachy_sdk_zeroed.head.cancel_all_goto()
    assert cancel.ack

    l_arm_position = reachy_sdk_zeroed.l_arm.get_current_positions()

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
    l_arm_position = reachy_sdk_zeroed.l_arm.get_current_positions()

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

    req2 = reachy_sdk_zeroed.l_arm.goto([0, 0, 0, 0, 0, 0, 0], duration=10, interpolation_mode="linear")
    req3 = reachy_sdk_zeroed.l_arm.goto([15, 10, 20, -50, 10, 10, 20], duration=10, interpolation_mode="linear")

    assert reachy_sdk_zeroed.l_arm.get_goto_queue() == [req2, req3]

    cancel2 = reachy_sdk_zeroed.l_arm.cancel_all_goto()
    assert cancel2.ack
    assert reachy_sdk_zeroed.l_arm.get_goto_queue() == []

    cancel3 = reachy_sdk_zeroed.cancel_all_goto()
    assert cancel3.ack


@pytest.mark.online
def test_get_goto_playing(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.head.goto([0, 0, -10], duration=3)
    req2 = reachy_sdk_zeroed.l_arm.goto([10, 10, 15, -20, 15, -15, -10], duration=5)
    req3 = reachy_sdk_zeroed.r_arm.goto([0, 10, 20, -40, 10, 10, -15], duration=10)

    req4 = reachy_sdk_zeroed.head.goto([30, 0, 0], duration=5)
    req5 = reachy_sdk_zeroed.l_arm.goto([0, 0, 5, -40, 10, -10, 0], duration=6)
    req6 = reachy_sdk_zeroed.r_arm.goto([15, 15, 0, 0, 25, 20, -5], duration=5)

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
    req1 = reachy_sdk_zeroed.head.goto([0, 0, -10], duration=3)
    req2 = reachy_sdk_zeroed.l_arm.goto([10, 10, 15, -20, 15, -15, -10], duration=5)
    req3 = reachy_sdk_zeroed.r_arm.goto([0, 10, 20, -40, 10, 10, -15], duration=10)

    req4 = reachy_sdk_zeroed.head.goto([30, 0, 0], duration=5)
    req5 = reachy_sdk_zeroed.l_arm.goto([0, 0, 5, -40, 10, -10, 0], duration=6)
    req6 = reachy_sdk_zeroed.r_arm.goto([15, 15, 0, 0, 25, 20, -5], duration=5)

    assert reachy_sdk_zeroed._get_goto_state(req1).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_goto_state(req2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_goto_state(req3).goal_status == GoalStatus.STATUS_EXECUTING

    assert (reachy_sdk_zeroed._get_goto_state(req4).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_goto_state(req4).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?
    assert (reachy_sdk_zeroed._get_goto_state(req5).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_goto_state(req5).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?
    assert (reachy_sdk_zeroed._get_goto_state(req6).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_goto_state(req6).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?

    while not is_goto_finished(reachy_sdk_zeroed, req1):
        time.sleep(0.1)

    assert reachy_sdk_zeroed._get_goto_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_goto_state(req2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_goto_state(req3).goal_status == GoalStatus.STATUS_EXECUTING

    assert reachy_sdk_zeroed._get_goto_state(req4).goal_status == GoalStatus.STATUS_EXECUTING
    assert (reachy_sdk_zeroed._get_goto_state(req5).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_goto_state(req5).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?
    assert (reachy_sdk_zeroed._get_goto_state(req6).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_goto_state(req6).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?

    while not is_goto_finished(reachy_sdk_zeroed, req3):
        time.sleep(0.1)

    cancel_l_arm = reachy_sdk_zeroed.l_arm.cancel_all_goto()
    assert cancel_l_arm.ack

    assert reachy_sdk_zeroed._get_goto_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_goto_state(req2).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_goto_state(req3).goal_status == GoalStatus.STATUS_SUCCEEDED

    assert reachy_sdk_zeroed._get_goto_state(req4).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert (reachy_sdk_zeroed._get_goto_state(req5).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed._get_goto_state(req5).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert reachy_sdk_zeroed._get_goto_state(req6).goal_status == GoalStatus.STATUS_EXECUTING

    cancel = reachy_sdk_zeroed.cancel_all_goto()
    assert cancel.ack

    assert reachy_sdk_zeroed._get_goto_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_goto_state(req2).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_goto_state(req3).goal_status == GoalStatus.STATUS_SUCCEEDED

    assert reachy_sdk_zeroed._get_goto_state(req4).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_goto_state(req5).goal_status == GoalStatus.STATUS_CANCELED
    assert (reachy_sdk_zeroed._get_goto_state(req6).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed._get_goto_state(req6).goal_status == GoalStatus.STATUS_CANCELED
    )


@pytest.mark.online
def test_get_goto_request(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.head.goto([30, 0, 0], duration=5)
    req2 = reachy_sdk_zeroed.l_arm.goto([10, 10, 15, -20, 15, -15, -10], duration=7, interpolation_mode="linear")
    req3 = reachy_sdk_zeroed.r_arm.goto([0, 10, 20, -40, 10, 10, -15], duration=10)

    ans1 = reachy_sdk_zeroed.get_goto_request(req1)
    assert ans1.part == "head"
    assert np.allclose(ans1.request.target.joints, [30, 0, 0], atol=1e-03)
    assert ans1.request.duration == 5
    assert ans1.request.mode == "minimum_jerk"

    ans2 = reachy_sdk_zeroed.get_goto_request(req2)
    assert ans2.part == "l_arm"
    assert np.allclose(ans2.request.target.joints, [10, 10, 15, -20, 15, -15, -10], atol=1e-03)
    assert ans2.request.duration == 7
    assert ans2.request.mode == "linear"

    ans3 = reachy_sdk_zeroed.get_goto_request(req3)
    assert ans3.part == "r_arm"
    assert np.allclose(ans3.request.target.joints, [0, 10, 20, -40, 10, 10, -15], atol=1e-03)
    assert ans3.request.duration == 10
    assert ans3.request.mode == "minimum_jerk"

    cancel = reachy_sdk_zeroed.cancel_all_goto()
    assert cancel.ack

    with pytest.raises(TypeError):
        reachy_sdk_zeroed.get_goto_request(3)


@pytest.mark.online
def test_reachy_goto_posture(reachy_sdk_zeroed: ReachySDK) -> None:
    # Test the default pose
    reachy_sdk_zeroed.l_arm.gripper.close()
    reachy_sdk_zeroed.r_arm.gripper.close()
    reachy_sdk_zeroed.goto_posture("default", open_gripper=True)
    time.sleep(2)
    assert reachy_sdk_zeroed.l_arm.gripper.opening == 100.0
    assert reachy_sdk_zeroed.r_arm.gripper.opening == 100.0
    reachy_sdk_zeroed.l_arm.gripper.set_opening(30)
    reachy_sdk_zeroed.r_arm.gripper.close()
    reachy_sdk_zeroed.goto_posture("elbow_90")
    time.sleep(2)
    reachy_sdk_zeroed.l_arm.turn_off()
    reachy_sdk_zeroed.goto_posture("default", open_gripper=True)
    time.sleep(2)
    assert np.isclose(reachy_sdk_zeroed.l_arm.gripper.opening, 30, atol=5)
    assert reachy_sdk_zeroed.r_arm.gripper.opening == 100.0

    reachy_sdk_zeroed.turn_on()

    zero_r_arm = [0, 10, -10, 0, 0, 0, 0]
    zero_l_arm = [0, -10, 10, 0, 0, 0, 0]
    elbow_90_r_arm = [0, 10, -10, -90, 0, 0, 0]
    elbow_90_l_arm = [0, -10, 10, -90, 0, 0, 0]
    zero_head = Quaternion(axis=[1, 0, 0], angle=0.0)

    # Test waiting for part's gotos to end

    req1 = reachy_sdk_zeroed.head.goto([30, 0, 0], duration=4)
    req2 = reachy_sdk_zeroed.r_arm.goto([0, 10, 20, -40, 10, 10, -15], duration=5)
    req3 = reachy_sdk_zeroed.l_arm.goto([10, 10, 15, -20, 15, -15, -10], duration=6)

    time.sleep(2)

    req_h, req_r, req_l = reachy_sdk_zeroed.goto_posture()

    assert reachy_sdk_zeroed._get_goto_state(req1).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_goto_state(req2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_goto_state(req3).goal_status == GoalStatus.STATUS_EXECUTING

    assert (reachy_sdk_zeroed._get_goto_state(req_h).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_goto_state(req_h).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?
    assert (reachy_sdk_zeroed._get_goto_state(req_r).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_goto_state(req_r).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?
    assert (reachy_sdk_zeroed._get_goto_state(req_l).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_goto_state(req_l).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?

    while not is_goto_finished(reachy_sdk_zeroed, req2):
        time.sleep(0.1)

    assert reachy_sdk_zeroed._get_goto_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_goto_state(req2).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_goto_state(req3).goal_status == GoalStatus.STATUS_EXECUTING

    assert reachy_sdk_zeroed._get_goto_state(req_h).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_goto_state(req_r).goal_status == GoalStatus.STATUS_EXECUTING
    assert (reachy_sdk_zeroed._get_goto_state(req_l).goal_status == GoalStatus.STATUS_ACCEPTED) | (
        reachy_sdk_zeroed._get_goto_state(req_l).goal_status == GoalStatus.STATUS_UNKNOWN
    )  # should be ACCEPTED ?

    while not is_goto_finished(reachy_sdk_zeroed, req_l):
        time.sleep(0.1)

    ans_r = reachy_sdk_zeroed.get_goto_request(req_r)
    assert ans_r.part == "r_arm"
    assert np.allclose(ans_r.request.target.joints, zero_r_arm, atol=1e-03)
    assert ans_r.request.duration == 2
    assert ans_r.request.mode == "minimum_jerk"

    assert reachy_sdk_zeroed._get_goto_state(req_h).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_goto_state(req_r).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_goto_state(req_l).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert np.isclose(
        Quaternion.distance(reachy_sdk_zeroed.head.get_current_orientation(), zero_head),
        0,
        atol=1e-04,
    )
    assert np.allclose(reachy_sdk_zeroed.r_arm.get_current_positions(), zero_r_arm, atol=1e-03)
    assert np.allclose(reachy_sdk_zeroed.l_arm.get_current_positions(), zero_l_arm, atol=1e-03)

    cancel = reachy_sdk_zeroed.cancel_all_goto()
    assert cancel.ack

    # Test without waiting for part's gotos to end

    req4 = reachy_sdk_zeroed.head.goto([30, 0, 0], duration=4)
    req5 = reachy_sdk_zeroed.r_arm.goto([0, 10, 20, -40, 10, 10, -15], duration=5)
    req6 = reachy_sdk_zeroed.l_arm.goto([10, 10, 15, -20, 15, -15, -10], duration=6)

    time.sleep(2)

    req_h2, req_r2, req_l2 = reachy_sdk_zeroed.goto_posture(
        "default", wait_for_goto_end=False, duration=1, interpolation_mode="linear"
    )

    assert (reachy_sdk_zeroed._get_goto_state(req4).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed._get_goto_state(req4).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert (reachy_sdk_zeroed._get_goto_state(req5).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed._get_goto_state(req5).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert (reachy_sdk_zeroed._get_goto_state(req6).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed._get_goto_state(req6).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert reachy_sdk_zeroed._get_goto_state(req_h2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_goto_state(req_r2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_goto_state(req_l2).goal_status == GoalStatus.STATUS_EXECUTING

    while not is_goto_finished(reachy_sdk_zeroed, req_l2):
        time.sleep(0.1)
    time.sleep(0.05)

    ans_l2 = reachy_sdk_zeroed.get_goto_request(req_l2)
    assert ans_l2.part == "l_arm"
    assert np.allclose(ans_l2.request.target.joints, zero_l_arm, atol=1e-03)
    assert ans_l2.request.duration == 1
    assert ans_l2.request.mode == "linear"

    assert reachy_sdk_zeroed._get_goto_state(req_h2).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_goto_state(req_r2).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_goto_state(req_l2).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert np.isclose(
        Quaternion.distance(reachy_sdk_zeroed.head.get_current_orientation(), zero_head),
        0,
        atol=1e-04,
    )
    assert np.allclose(reachy_sdk_zeroed.r_arm.get_current_positions(), zero_r_arm, atol=1e-03)
    assert np.allclose(reachy_sdk_zeroed.l_arm.get_current_positions(), zero_l_arm, atol=1e-03)

    # Test with 'elbow_90' instead of 'default'

    req7 = reachy_sdk_zeroed.head.goto([30, 0, 0], duration=4)
    req8 = reachy_sdk_zeroed.r_arm.goto([0, 10, 20, -40, 10, 10, -15], duration=5)
    req9 = reachy_sdk_zeroed.l_arm.goto([10, 10, 15, -20, 15, -15, -10], duration=6)

    time.sleep(2)

    req_h3, req_r3, req_l3 = reachy_sdk_zeroed.goto_posture("elbow_90", wait_for_goto_end=False, duration=2)

    assert (reachy_sdk_zeroed._get_goto_state(req7).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed._get_goto_state(req7).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert (reachy_sdk_zeroed._get_goto_state(req8).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed._get_goto_state(req8).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert (reachy_sdk_zeroed._get_goto_state(req9).goal_status == GoalStatus.STATUS_CANCELING) | (
        reachy_sdk_zeroed._get_goto_state(req9).goal_status == GoalStatus.STATUS_CANCELED
    )
    assert reachy_sdk_zeroed._get_goto_state(req_h3).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_goto_state(req_r3).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_goto_state(req_l3).goal_status == GoalStatus.STATUS_EXECUTING

    while not is_goto_finished(reachy_sdk_zeroed, req_l3):
        time.sleep(0.1)

    ans_l3 = reachy_sdk_zeroed.get_goto_request(req_l3)
    assert ans_l3.part == "l_arm"
    assert np.allclose(ans_l3.request.target.joints, elbow_90_l_arm, atol=1e-03)
    assert ans_l3.request.duration == 2
    assert ans_l3.request.mode == "minimum_jerk"

    assert reachy_sdk_zeroed._get_goto_state(req_h3).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_goto_state(req_r3).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_goto_state(req_l3).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert np.isclose(
        Quaternion.distance(reachy_sdk_zeroed.head.get_current_orientation(), zero_head),
        0,
        atol=1e-04,
    )  # why not 1e-04 here?
    assert np.allclose(reachy_sdk_zeroed.r_arm.get_current_positions(), elbow_90_r_arm, atol=1e-03)
    assert np.allclose(reachy_sdk_zeroed.l_arm.get_current_positions(), elbow_90_l_arm, atol=1e-03)

    # Test with some parts off

    reachy_sdk_zeroed.l_arm.turn_off()
    reachy_sdk_zeroed.head.turn_off()
    time.sleep(0.1)

    req_h4, req_r4, req_l4 = reachy_sdk_zeroed.goto_posture("default", wait_for_goto_end=True, duration=2)

    time.sleep(0.5)

    assert req_h4.id == -1
    assert req_l4.id == -1
    assert req_r4 != -1

    reachy_sdk_zeroed.turn_on()


@pytest.mark.online
def test_wait_move(reachy_sdk_zeroed: ReachySDK) -> None:
    tic = time.time()
    reachy_sdk_zeroed.head.goto([30, 0, 0], duration=4)
    reachy_sdk_zeroed.l_arm.goto(
        [10, 10, 15, -20, 15, -15, -10],
        duration=3,
        wait=True,
        interpolation_mode="linear",
    )
    elapsed_time = time.time() - tic
    assert np.isclose(elapsed_time, 3.0, 1e-01)

    reachy_sdk_zeroed.r_arm.goto([0, 10, 20, -40, 10, 10, -15], duration=4)
    time.sleep(0.1)

    assert reachy_sdk_zeroed.head.get_goto_playing().id != -1
    assert reachy_sdk_zeroed.l_arm.get_goto_playing().id == -1
    assert reachy_sdk_zeroed.r_arm.get_goto_playing().id != -1

    tic = time.time()
    reachy_sdk_zeroed.goto_posture("default", duration=2, wait=True, wait_for_goto_end=False)
    elapsed_time = time.time() - tic
    assert np.isclose(elapsed_time, 2.0, 1e-01)

    A = build_pose_matrix(0.3, -0.4, -0.3)
    tic = time.time()
    reachy_sdk_zeroed.r_arm.goto(A, duration=2.0, wait=True)
    elapsed_time = time.time() - tic
    assert np.isclose(elapsed_time, 2.0, 1e-01)
    B = build_pose_matrix(0.3, 0.4, 0)
    tic = time.time()
    reachy_sdk_zeroed.l_arm.goto(B, duration=2.0)
    elapsed_time = time.time() - tic
    assert elapsed_time < 0.1

    tic = time.time()
    reachy_sdk_zeroed.goto_posture("default", duration=1.0)
    reachy_sdk_zeroed.r_arm.goto(A, wait=True, duration=1.0)
    elapsed_time = time.time() - tic
    assert elapsed_time >= 2.0


@pytest.mark.online
def test_is_goto_finished(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.head.goto([30, 0, 0], duration=2)
    req2 = reachy_sdk_zeroed.l_arm.goto([10, 10, 15, -20, 15, -15, -10], duration=3, interpolation_mode="linear")
    req3 = reachy_sdk_zeroed.r_arm.goto([0, 10, 20, -40, 10, 10, -15], duration=4)

    time.sleep(1)
    assert not reachy_sdk_zeroed.is_goto_finished(req1)
    assert not reachy_sdk_zeroed.is_goto_finished(req2)
    assert not reachy_sdk_zeroed.is_goto_finished(req3)

    req4 = reachy_sdk_zeroed.head.goto([0, 0, 0], duration=1)
    req5 = reachy_sdk_zeroed.l_arm.goto([0, 0, 0, 0, 0, 0, 0], duration=1)
    req6 = reachy_sdk_zeroed.r_arm.goto([0, 0, 0, 0, 0, 0, 0], duration=4)

    time.sleep(1)
    assert reachy_sdk_zeroed._get_goto_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_goto_state(req2).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed._get_goto_state(req3).goal_status == GoalStatus.STATUS_EXECUTING
    assert reachy_sdk_zeroed.is_goto_finished(req1)
    assert not reachy_sdk_zeroed.is_goto_finished(req2)
    assert not reachy_sdk_zeroed.is_goto_finished(req3)
    assert not reachy_sdk_zeroed.is_goto_finished(req4)
    assert not reachy_sdk_zeroed.is_goto_finished(req5)
    assert not reachy_sdk_zeroed.is_goto_finished(req6)

    cancel = reachy_sdk_zeroed.l_arm.cancel_all_goto()
    assert cancel.ack
    assert reachy_sdk_zeroed.is_goto_finished(req2)

    time.sleep(2)
    assert reachy_sdk_zeroed._get_goto_state(req1).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed._get_goto_state(req2).goal_status == GoalStatus.STATUS_CANCELED
    assert reachy_sdk_zeroed._get_goto_state(req3).goal_status == GoalStatus.STATUS_SUCCEEDED
    assert reachy_sdk_zeroed.is_goto_finished(req1)
    assert reachy_sdk_zeroed.is_goto_finished(req2)
    assert reachy_sdk_zeroed.is_goto_finished(req3)
    assert reachy_sdk_zeroed.is_goto_finished(req4)
    assert reachy_sdk_zeroed.is_goto_finished(req5)
    assert not reachy_sdk_zeroed.is_goto_finished(req6)

    with pytest.raises(TypeError):
        reachy_sdk_zeroed.is_goto_finished(3)


@pytest.mark.online
def test_single_joint_goto(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.r_arm.elbow.pitch.goto(-90, duration=3)
    time.sleep(0.5)

    while not is_goto_finished(reachy_sdk_zeroed, req1):
        time.sleep(0.1)

    assert np.allclose(
        reachy_sdk_zeroed.r_arm.get_current_positions(),
        [0, 0, 0, -90, 0, 0, 0],
        atol=1e-01,
    )

    req2 = reachy_sdk_zeroed.r_arm.elbow.pitch.goto(0, duration=1)
    time.sleep(0.5)

    while not is_goto_finished(reachy_sdk_zeroed, req2):
        time.sleep(0.1)

    assert np.allclose(
        reachy_sdk_zeroed.r_arm.get_current_positions(),
        [0, 0, 0, 0, 0, 0, 0],
        atol=1e-01,
    )

    req3 = reachy_sdk_zeroed.l_arm.shoulder.pitch.goto(-10, duration=1)
    req4 = reachy_sdk_zeroed.l_arm.elbow.yaw.goto(20, duration=1)
    req5 = reachy_sdk_zeroed.l_arm.wrist.pitch.goto(15, duration=1)
    time.sleep(0.5)

    assert len(reachy_sdk_zeroed.l_arm.get_goto_queue()) == 2
    while not is_goto_finished(reachy_sdk_zeroed, req5):
        time.sleep(0.1)

    assert np.allclose(
        reachy_sdk_zeroed.l_arm.get_current_positions(),
        [-10, 0, 20, 0, 0, 15, 0],
        atol=1e-01,
    )

    req6 = reachy_sdk_zeroed.head.neck.roll.goto(15, duration=1)
    req7 = reachy_sdk_zeroed.head.neck.yaw.goto(10, duration=1)
    time.sleep(0.5)

    assert len(reachy_sdk_zeroed.head.get_goto_queue()) == 1
    while not is_goto_finished(reachy_sdk_zeroed, req7):
        time.sleep(0.1)

    assert np.allclose(reachy_sdk_zeroed.head.get_current_positions(), [15, 0, 10], atol=1e-01)

    reachy_sdk_zeroed.turn_off()
    assert reachy_sdk_zeroed.l_arm.shoulder.roll.goto(10).id == -1
    assert reachy_sdk_zeroed.head.neck.roll.goto(10).id == -1
    reachy_sdk_zeroed.turn_on()


@pytest.mark.online
def test_get_translation_by(reachy_sdk_zeroed: ReachySDK) -> None:
    pose1 = reachy_sdk_zeroed.r_arm.forward_kinematics([0, -15, -15, -90, 0, 0, 0])
    pose2 = reachy_sdk_zeroed.r_arm.get_translation_by(0.1, 0, 0, initial_pose=pose1, frame="robot")

    assert np.allclose(pose1[:3, :3], pose2[:3, :3], atol=1e-03)
    assert np.isclose(pose1[0, 3] + 0.1, pose2[0, 3], atol=1e-03)
    assert np.isclose(pose1[1, 3], pose2[1, 3], atol=1e-03)
    assert np.isclose(pose1[2, 3], pose2[2, 3], atol=1e-03)

    pose3 = reachy_sdk_zeroed.r_arm.forward_kinematics([-10, -15, -15, -100, 0, 0, 0])
    pose4 = reachy_sdk_zeroed.r_arm.get_translation_by(-0.1, -0.1, 0.1, initial_pose=pose3, frame="robot")
    assert np.allclose(pose3[:3, :3], pose4[:3, :3], atol=1e-03)
    assert np.isclose(pose3[0, 3] - 0.1, pose4[0, 3], atol=1e-03)
    assert np.isclose(pose3[1, 3] - 0.1, pose4[1, 3], atol=1e-03)
    assert np.isclose(pose3[2, 3] + 0.1, pose4[2, 3], atol=1e-03)

    pose5 = reachy_sdk_zeroed.r_arm.get_translation_by(0.1, 0.1, -0.1, initial_pose=pose3, frame="gripper")
    translation5 = np.eye(4)
    translation5[0, 3] = 0.1
    translation5[1, 3] = 0.1
    translation5[2, 3] = -0.1
    assert np.allclose(pose3 @ translation5, pose5, atol=1e-03)

    pose6 = reachy_sdk_zeroed.r_arm.get_translation_by(0, -0.2, -0.2, initial_pose=pose3, frame="gripper")
    translation6 = np.eye(4)
    translation6[0, 3] = 0
    translation6[1, 3] = -0.2
    translation6[2, 3] = -0.2
    assert np.allclose(pose3 @ translation6, pose6, atol=1e-03)


@pytest.mark.online
def test_get_rotation_by(reachy_sdk_zeroed: ReachySDK) -> None:
    pose1 = reachy_sdk_zeroed.r_arm.forward_kinematics([0, -15, -15, -90, 0, 0, 0])
    pose2 = reachy_sdk_zeroed.r_arm.get_rotation_by(10, 0, 0, initial_pose=pose1, frame="gripper")

    rotation2 = matrix_from_euler_angles(10, 0, 0)
    assert np.allclose(pose1 @ rotation2, pose2, atol=1e-03)
    pose2_expected = np.array(
        [
            [-0.08682488, -0.0639791, -0.99416705, 0.40057661],
            [0.49240448, 0.86475723, -0.0986548, -0.35801578],
            [0.86602498, -0.498098, -0.04357885, -0.19331117],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(pose2_expected, pose2, atol=1e-03)

    pose3 = reachy_sdk_zeroed.r_arm.forward_kinematics([-10, -15, -15, -100, 0, 0, 0])
    pose4 = reachy_sdk_zeroed.r_arm.get_rotation_by(15, 10, -15, initial_pose=pose3, frame="gripper")

    rotation4 = matrix_from_euler_angles(15, 10, -15)
    assert np.allclose(pose3 @ rotation4, pose4, atol=1e-03)

    pose5 = reachy_sdk_zeroed.r_arm.get_rotation_by(-15, -10, 15, initial_pose=pose3, frame="robot")
    rotation5 = matrix_from_euler_angles(-15, -10, 15)
    expected_pose5_rot = rotation5[:3, :3] @ pose3[:3, :3]
    expected_pose5 = np.eye(4)
    expected_pose5[:3, :3] = expected_pose5_rot
    expected_pose5[:3, 3] = pose3[:3, 3]
    assert np.allclose(expected_pose5, pose5, atol=1e-03)

    pose6 = reachy_sdk_zeroed.r_arm.get_rotation_by(5, 0, -10, initial_pose=pose3, frame="robot")
    rotation6 = matrix_from_euler_angles(5, 0, -10)
    expected_pose6_rot = rotation6[:3, :3] @ pose3[:3, :3]
    expected_pose6 = np.eye(4)
    expected_pose6[:3, :3] = expected_pose6_rot
    expected_pose6[:3, 3] = pose3[:3, 3]
    assert np.allclose(expected_pose6, pose6, atol=1e-03)


@pytest.mark.online
def test_translate_by_robot_frame(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.r_arm.goto_posture("elbow_90")
    while not is_goto_finished(reachy_sdk_zeroed, req1):
        time.sleep(0.1)

    pose1 = reachy_sdk_zeroed.r_arm.forward_kinematics()
    req2 = reachy_sdk_zeroed.r_arm.translate_by(0.1, 0, 0, frame="robot")
    while not is_goto_finished(reachy_sdk_zeroed, req2):
        time.sleep(0.1)

    pose2 = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(pose1[:3, :3], pose2[:3, :3], atol=1e-03)
    assert np.isclose(pose1[0, 3] + 0.1, pose2[0, 3], atol=1e-03)
    assert np.isclose(pose1[1, 3], pose2[1, 3], atol=1e-03)
    assert np.isclose(pose1[2, 3], pose2[2, 3], atol=1e-03)

    req3 = reachy_sdk_zeroed.r_arm.goto([-10, -15, -15, -100, 0, 0, 0])
    req4 = reachy_sdk_zeroed.r_arm.goto([-10, -15, 20, -110, 0, 0, 0])
    req5 = reachy_sdk_zeroed.r_arm.translate_by(0.1, -0.1, -0.1)

    get_req4 = reachy_sdk_zeroed.get_goto_request(req4)
    if get_req4.request.target.joints is not None:
        pose4 = reachy_sdk_zeroed.r_arm.forward_kinematics(get_req4.request.target.joints)
    else:
        pose4 = get_req4.request.target.pose
    get_req5 = reachy_sdk_zeroed.get_goto_request(req5)
    if get_req5.request.target.joints is not None:
        pose5 = reachy_sdk_zeroed.r_arm.forward_kinematics(get_req5.request.target.joints)
    else:
        pose5 = get_req5.request.target.pose
    assert np.allclose(pose4[:3, :3], pose5[:3, :3], atol=1e-03)
    assert np.isclose(pose4[0, 3] + 0.1, pose5[0, 3], atol=1e-03)
    assert np.isclose(pose4[1, 3] - 0.1, pose5[1, 3], atol=1e-03)
    assert np.isclose(pose4[2, 3] - 0.1, pose5[2, 3], atol=1e-03)

    req6 = reachy_sdk_zeroed.r_arm.translate_by(0, 0, -0.1)

    with pytest.raises(ValueError):
        reachy_sdk_zeroed.r_arm.forward_kinematics(reachy_sdk_zeroed.get_goto_request(req6).request.target.pose)


@pytest.mark.online
def test_translate_by_gripper_frame(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.r_arm.goto_posture("elbow_90")
    while not is_goto_finished(reachy_sdk_zeroed, req1):
        time.sleep(0.1)

    pose1 = reachy_sdk_zeroed.r_arm.forward_kinematics()
    req2 = reachy_sdk_zeroed.r_arm.translate_by(0.1, 0, 0, frame="gripper", duration=3.0)
    tic = time.time()
    while not is_goto_finished(reachy_sdk_zeroed, req2):
        time.sleep(0.1)
    elapsed_time = time.time() - tic
    assert np.isclose(elapsed_time, 3.0, 1e-01)

    pose2 = reachy_sdk_zeroed.r_arm.forward_kinematics()
    translation2 = np.eye(4)
    translation2[0, 3] = 0.1
    assert np.allclose(pose1 @ translation2, pose2, atol=1e-03)

    req3 = reachy_sdk_zeroed.r_arm.goto([-10, -15, -15, -100, 0, 0, 0])
    req4 = reachy_sdk_zeroed.r_arm.goto([-10, -15, 20, -110, 0, 0, 0])
    req5 = reachy_sdk_zeroed.r_arm.translate_by(0.1, -0.1, -0.1, frame="gripper")

    get_req4 = reachy_sdk_zeroed.get_goto_request(req4)
    if get_req4.request.target.joints is not None:
        pose4 = reachy_sdk_zeroed.r_arm.forward_kinematics(get_req4.request.target.joints)
    else:
        pose4 = get_req4.request.target.pose
    get_req5 = reachy_sdk_zeroed.get_goto_request(req5)
    if get_req5.request.target.joints is not None:
        pose5 = reachy_sdk_zeroed.r_arm.forward_kinematics(get_req5.request.target.joints)
    else:
        pose5 = get_req5.request.target.pose
    translation5 = np.eye(4)
    translation5[0, 3] = 0.1
    translation5[1, 3] = -0.1
    translation5[2, 3] = -0.1
    assert np.allclose(pose4 @ translation5, pose5, atol=1e-03)

    while not is_goto_finished(reachy_sdk_zeroed, req5):
        time.sleep(0.1)

    req6 = reachy_sdk_zeroed.r_arm.goto([-10, -15, 30, -70, 0, 10, 0])
    req7 = reachy_sdk_zeroed.r_arm.translate_by(0.15, 0, 0.05, frame="gripper")

    get_req6 = reachy_sdk_zeroed.get_goto_request(req6)
    if get_req6.request.target.joints is not None:
        pose6 = reachy_sdk_zeroed.r_arm.forward_kinematics(get_req6.request.target.joints)
    else:
        pose6 = get_req6.request.target.pose
    get_req7 = reachy_sdk_zeroed.get_goto_request(req7)
    if get_req7.request.target.joints is not None:
        pose7 = reachy_sdk_zeroed.r_arm.forward_kinematics(get_req7.request.target.joints)
    else:
        pose7 = get_req7.request.target.pose
    translation7 = np.eye(4)
    translation7[0, 3] = 0.15
    translation7[2, 3] = 0.05
    assert np.allclose(pose6 @ translation7, pose7, atol=1e-03)

    req8 = reachy_sdk_zeroed.r_arm.translate_by(0, -0.3, -0.3, frame="gripper", wait=True)

    assert not np.allclose(
        reachy_sdk_zeroed.get_goto_request(req8).request.target.pose,
        reachy_sdk_zeroed.r_arm.forward_kinematics(),
    )


@pytest.mark.online
def test_rotate_by_robot_frame(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.r_arm.goto_posture("elbow_90")
    while not is_goto_finished(reachy_sdk_zeroed, req1):
        time.sleep(0.1)

    req2 = reachy_sdk_zeroed.r_arm.rotate_by(10, 0, 0, frame="robot")
    while not is_goto_finished(reachy_sdk_zeroed, req2):
        time.sleep(0.1)
    pose2_expected = np.array(
        [
            [-1.51352471e-02, 6.51403207e-04, -9.99885243e-01, 3.84194035e-01],
            [-8.84590707e-02, 9.96077829e-01, 1.98792631e-03, -2.23785617e-01],
            [9.95964817e-01, 8.84790072e-02, -1.50182615e-02, -2.73183117e-01],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pose2 = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(pose2_expected, pose2, atol=1e-03)

    req3 = reachy_sdk_zeroed.r_arm.goto([-10, 10, -15, -100, 0, 0, 0])
    req4 = reachy_sdk_zeroed.r_arm.goto([-10, 10, 20, -110, 0, 0, 0])
    req5 = reachy_sdk_zeroed.r_arm.rotate_by(15, -10, -5, frame="robot")

    get_req5 = reachy_sdk_zeroed.get_goto_request(req5)
    if get_req5.request.target.joints is not None:
        pose5 = reachy_sdk_zeroed.r_arm.forward_kinematics(get_req5.request.target.joints)
    else:
        pose5 = get_req5.request.target.pose
    expected_pose5 = np.array(
        [
            [-0.62069375, -0.4494677, -0.64243136, 0.32550951],
            [-0.30411506, 0.89323893, -0.33111666, -0.0165214],
            [0.72267095, -0.01014899, -0.69111772, -0.10196663],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(expected_pose5, pose5, atol=1e-03)

    while not is_goto_finished(reachy_sdk_zeroed, req5):
        time.sleep(0.1)

    req6 = reachy_sdk_zeroed.r_arm.goto([-10, 10, 30, -70, 0, 10, 0])
    req7 = reachy_sdk_zeroed.r_arm.rotate_by(0.15, 0, 0.05)

    get_req7 = reachy_sdk_zeroed.get_goto_request(req7)
    if get_req7.request.target.joints is not None:
        pose7 = reachy_sdk_zeroed.r_arm.forward_kinematics(get_req7.request.target.joints)
    else:
        pose7 = get_req7.request.target.pose
    expected_pose7 = np.array(
        [
            [0.20722949, -0.6570895, -0.72476846, 0.33380879],
            [0.38013816, 0.73671839, -0.5592325, 0.01343369],
            [0.90141606, -0.15962268, 0.40245457, -0.38287815],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(expected_pose7, pose7, atol=1e-03)

    req8 = reachy_sdk_zeroed.r_arm.rotate_by(50, -20, 70, frame="robot")

    with pytest.raises(ValueError):
        reachy_sdk_zeroed.r_arm.forward_kinematics(reachy_sdk_zeroed.get_goto_request(req8).request.target.joints)


@pytest.mark.online
def test_rotate_by_gripper_frame(reachy_sdk_zeroed: ReachySDK) -> None:
    req1 = reachy_sdk_zeroed.r_arm.goto_posture("elbow_90")
    while not is_goto_finished(reachy_sdk_zeroed, req1):
        time.sleep(0.1)

    req2 = reachy_sdk_zeroed.r_arm.rotate_by(10, 0, 0, frame="gripper", duration=3.0)
    tic = time.time()
    while not is_goto_finished(reachy_sdk_zeroed, req2):
        time.sleep(0.05)
    elapsed_time = time.time() - tic
    assert np.isclose(elapsed_time, 3.0, 1e-01)
    pose2_expected = np.array(
        [
            [-1.51352471e-02, -1.72986743e-01, -9.84807855e-01, 3.84194035e-01],
            [8.58322969e-02, 9.81060308e-01, -1.73647600e-01, -2.23785617e-01],
            [9.96194630e-01, -8.71565195e-02, -7.40616532e-07, -2.73183117e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    pose2 = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(pose2_expected, pose2, atol=1e-03)

    req3 = reachy_sdk_zeroed.r_arm.goto([-10, 10, -10, -100, 0, 0, 0])
    req4 = reachy_sdk_zeroed.r_arm.goto([-10, 10, 10, -110, 0, 0, 0])
    req5 = reachy_sdk_zeroed.r_arm.rotate_by(15, -10, -5, frame="gripper")

    get_req5 = reachy_sdk_zeroed.get_goto_request(req5)
    if get_req5.request.target.joints is not None:
        pose5 = reachy_sdk_zeroed.r_arm.forward_kinematics(get_req5.request.target.joints)
    else:
        pose5 = get_req5.request.target.pose
    expected_pose5 = np.array(
        [
            [-0.58122566, -0.57576842, -0.57503692, 0.35318658],
            [-0.22426798, 0.79263062, -0.56695729, -0.07168226],
            [0.78222798, -0.20056775, -0.58982368, -0.09387122],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(expected_pose5, pose5, atol=1e-03)

    while not is_goto_finished(reachy_sdk_zeroed, req5):
        time.sleep(0.1)

    req6 = reachy_sdk_zeroed.r_arm.goto([-10, 10, 15, -70, 0, 10, 0])
    req7 = reachy_sdk_zeroed.r_arm.rotate_by(0.15, 0, 0.05, frame="gripper")

    get_req7 = reachy_sdk_zeroed.get_goto_request(req7)
    if get_req7.request.target.joints is not None:
        pose7 = reachy_sdk_zeroed.r_arm.forward_kinematics(get_req7.request.target.joints)
    else:
        pose7 = get_req7.request.target.pose
    expected_pose7 = np.array(
        [
            [0.27965445, -0.44712639, -0.84963015, 0.38450476],
            [0.27624796, 0.88498932, -0.37480792, -0.06129458],
            [0.91950011, -0.12989189, 0.37100895, -0.36945202],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(expected_pose7, pose7, atol=1e-03)

    req8 = reachy_sdk_zeroed.r_arm.rotate_by(50, -80, 70, frame="gripper")

    with pytest.raises(ValueError):
        reachy_sdk_zeroed.r_arm.forward_kinematics(reachy_sdk_zeroed.get_goto_request(req8).request.target.joints)


@pytest.mark.online
def test_head_rotation(reachy_sdk_zeroed: ReachySDK) -> None:
    reachy_sdk_zeroed.head.goto([0, 0, 90], duration=1, wait=True)
    time.sleep(0.05)
    current_orientation = reachy_sdk_zeroed.head.get_current_positions()
    assert np.allclose(current_orientation, [0, 0, 90], atol=1e-03)

    reachy_sdk_zeroed.head.rotate_by(roll=30, duration=1, frame="head", wait=True)
    time.sleep(0.05)
    current_orientation = reachy_sdk_zeroed.head.get_current_positions()
    assert np.allclose(current_orientation, [0, 30, 90], atol=1e-03)

    reachy_sdk_zeroed.head.rotate_by(roll=-30, duration=1, frame="head", wait=True)
    time.sleep(0.05)
    current_orientation = reachy_sdk_zeroed.head.get_current_positions()
    assert np.allclose(current_orientation, [0, 0, 90], atol=1e-03)

    reachy_sdk_zeroed.head.rotate_by(roll=30, duration=1, frame="robot", wait=True)
    time.sleep(0.05)
    current_orientation = reachy_sdk_zeroed.head.get_current_positions()
    assert np.allclose(current_orientation, [29.507, 1.313, 94.981], atol=1e-03)

    reachy_sdk_zeroed.head.rotate_by(roll=-30, duration=1, frame="robot", wait=True)
    time.sleep(0.05)
    reachy_sdk_zeroed.head.rotate_by(pitch=30, duration=1, frame="head", wait=True)
    time.sleep(0.05)
    current_orientation = reachy_sdk_zeroed.head.get_current_positions()
    assert np.allclose(current_orientation, [-30, 0, 90], atol=1e-03)

    reachy_sdk_zeroed.head.rotate_by(pitch=-30, duration=1, frame="head", wait=True)
    time.sleep(0.05)
    reachy_sdk_zeroed.head.rotate_by(pitch=30, duration=1, frame="robot", wait=True)
    time.sleep(0.05)
    current_orientation = reachy_sdk_zeroed.head.get_current_positions()
    assert np.allclose(current_orientation, [0.000, 30.000, 89.998], atol=1e-03)

    reachy_sdk_zeroed.head.rotate_by(yaw=30, duration=1, frame="head", wait=True)
    time.sleep(0.05)
    current_orientation = reachy_sdk_zeroed.head.get_current_positions()
    assert np.allclose(current_orientation, [0, 30, 120], atol=1e-03)

    with pytest.raises(ValueError):
        reachy_sdk_zeroed.head.rotate_by(roll=30, frame="head", duration=0)

    with pytest.raises(ValueError):
        reachy_sdk_zeroed.head.rotate_by(roll=30, frame="coucou")

    with pytest.raises(ValueError):
        reachy_sdk_zeroed.head.rotate_by(roll=30, interpolation_mode="coucou")


@pytest.mark.online
def test_waiting_goto(reachy_sdk_zeroed: ReachySDK) -> None:
    def look_at_current_position() -> None:
        # sleep before reading to avoid outdated buffer
        # TODO implement unbeffered read
        time.sleep(0.05)
        x, y, z = reachy_sdk_zeroed.l_arm.forward_kinematics()[:3, 3]
        reachy_sdk_zeroed.head.look_at(x=x, y=y, z=z, duration=0.1, wait=True)
        assert reachy_sdk_zeroed.head.get_goto_playing().id == -1
        assert reachy_sdk_zeroed.head.get_goto_queue() == []

    reachy_sdk_zeroed.goto_posture("elbow_90", duration=0.5, wait=True)
    look_at_current_position()
    time.sleep(0.05)
    head_joints = reachy_sdk_zeroed.head.get_current_positions()
    first_gotoid = reachy_sdk_zeroed.l_arm.translate_by(x=0.2, y=0.0, z=0.1, duration=0.5, wait=False)
    reachy_sdk_zeroed.l_arm.translate_by(x=-0.2, y=0.0, z=-0.1, duration=0.5, wait=False)
    reachy_sdk_zeroed.l_arm.translate_by(x=0.2, y=0.0, z=0.1, duration=0.5, wait=False)
    last_gotoid = reachy_sdk_zeroed.l_arm.translate_by(x=-0.2, y=0.0, z=-0.1, duration=0.5, wait=False)
    assert reachy_sdk_zeroed.l_arm.get_goto_playing().id == first_gotoid.id
    assert len(reachy_sdk_zeroed.l_arm.get_goto_queue()) == 3

    while not reachy_sdk_zeroed.is_goto_finished(last_gotoid):
        look_at_current_position()

    # one last look after the movement is finished
    look_at_current_position()
    time.sleep(0.05)

    assert np.allclose(reachy_sdk_zeroed.head.get_current_positions(), head_joints, atol=1e-03)
