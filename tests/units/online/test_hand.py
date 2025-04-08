import time

import numpy as np
import pytest

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
    assert reachy_sdk_zeroed.l_arm.is_on(check_gripper=False) == True
    assert reachy_sdk_zeroed.l_arm.is_on() == False
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.pitch.present_position, -60, 10)

    reachy_sdk_zeroed.turn_on()


@pytest.mark.online
def test_gripper_goto(reachy_sdk_zeroed: ReachySDK) -> None:
    reachy_sdk_zeroed.l_arm.gripper.goto(40, percentage=True)
    reachy_sdk_zeroed.r_arm.gripper.goto(60)
    time.sleep(2.1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.gripper.opening, 40, 1e-01)
    assert np.isclose(reachy_sdk_zeroed.l_arm.gripper.present_position, 49, 1e-01)
    assert np.isclose(reachy_sdk_zeroed.r_arm.gripper.opening, 48.15, 1e-01)
    assert np.isclose(reachy_sdk_zeroed.r_arm.gripper.present_position, 60, 1e-01)

    tic = time.time()
    reachy_sdk_zeroed.l_arm.gripper.goto(25, duration=1.0, wait=True)
    duration = time.time() - tic
    assert np.isclose(duration, 1.0, 0.2)

    req1 = reachy_sdk_zeroed.r_arm.gripper.goto(75, duration=1.0, interpolation_mode="linear")
    req2 = reachy_sdk_zeroed.l_arm.gripper.goto(55, duration=3.0)
    req3 = reachy_sdk_zeroed.l_arm.gripper.goto(30, duration=3.0, percentage=True)

    ans1 = reachy_sdk_zeroed.get_goto_request(req1)
    assert ans1.part == "r_hand"
    assert np.allclose(ans1.request.target.joints, 75, atol=1e-03)
    assert ans1.request.duration == 1
    assert ans1.request.mode == "linear"

    ans2 = reachy_sdk_zeroed.get_goto_request(req2)
    assert ans2.part == "l_hand"
    assert np.allclose(ans2.request.target.joints, 55, atol=1e-03)
    assert ans2.request.duration == 3
    assert ans2.request.mode == "minimum_jerk"

    ans3 = reachy_sdk_zeroed.get_goto_request(req3)
    assert ans3.part == "l_hand"
    assert np.allclose(ans3.request.target.joints, 35.5, atol=1e-03)
    assert ans3.request.duration == 3
    assert ans3.request.mode == "minimum_jerk"

    req4 = reachy_sdk_zeroed.l_arm.gripper.goto(40)
    reachy_sdk_zeroed.l_arm.gripper.goto(60)
    reachy_sdk_zeroed.l_arm.gripper.goto(25, percentage=True)

    while not reachy_sdk_zeroed.is_goto_finished(req2):
        time.sleep(0.1)

    assert len(reachy_sdk_zeroed.l_arm.gripper.get_goto_queue()) == 3
    reachy_sdk_zeroed.cancel_goto_by_id(req4)
    assert len(reachy_sdk_zeroed.l_arm.gripper.get_goto_queue()) == 2

    reachy_sdk_zeroed.l_arm.gripper.cancel_all_goto()
    assert len(reachy_sdk_zeroed.l_arm.gripper.get_goto_queue()) == 0

    goto_id = reachy_sdk_zeroed.l_arm.gripper.goto(40)
    reachy_sdk_zeroed.l_arm.gripper.goto(60)
    reachy_sdk_zeroed.l_arm.gripper.goto(25, percentage=True)

    time.sleep(0.1)
    assert reachy_sdk_zeroed.l_arm.gripper.get_goto_playing() == goto_id
    assert len(reachy_sdk_zeroed.l_arm.gripper.get_goto_queue()) == 2
    reachy_sdk_zeroed.cancel_all_goto()
    assert len(reachy_sdk_zeroed.l_arm.gripper.get_goto_queue()) == 0


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

    reachy_sdk_zeroed.l_arm.gripper.goto(0, percentage=True, interpolation_mode="linear")
    assert reachy_sdk_zeroed.l_arm.gripper.is_moving()
    time.sleep(1.0)
    assert reachy_sdk_zeroed.l_arm.gripper.is_moving()
    time.sleep(1.1)
    assert not reachy_sdk_zeroed.l_arm.gripper.is_moving()

    reachy_sdk_zeroed.l_arm.gripper.goto(100, duration=3.0, percentage=True)
    assert reachy_sdk_zeroed.l_arm.gripper.is_moving()
    time.sleep(3.1)
    assert not reachy_sdk_zeroed.l_arm.gripper.is_moving()

    reachy_sdk_zeroed.l_arm.gripper.goto(0, percentage=True)
    reachy_sdk_zeroed.l_arm.gripper.goto(100, percentage=True)
    tic = time.time()
    assert reachy_sdk_zeroed.l_arm.gripper.is_moving()
    while reachy_sdk_zeroed.l_arm.gripper.get_goto_playing().id != -1:
        time.sleep(0.1)
    assert not reachy_sdk_zeroed.l_arm.gripper.is_moving()
    assert np.isclose(time.time() - tic, 4.0, 0.1)
