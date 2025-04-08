import time
from typing import Dict, List

import numpy as np
import pytest

from reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.online
def test_reachy_send_goal_positions(reachy_sdk_zeroed: ReachySDK) -> None:
    r_arm_present_positions = reachy_sdk_zeroed.r_arm.get_current_positions()
    l_arm_present_positions = reachy_sdk_zeroed.l_arm.get_current_positions()

    reachy_sdk_zeroed.r_arm.shoulder.pitch.goal_position = 20
    reachy_sdk_zeroed.r_arm.shoulder.roll.goal_position = -10
    reachy_sdk_zeroed.r_arm.elbow.pitch.goal_position = -50
    reachy_sdk_zeroed.r_arm.wrist.roll.goal_position = 30

    reachy_sdk_zeroed.l_arm.elbow.yaw.goal_position = -20
    reachy_sdk_zeroed.l_arm.wrist.roll.goal_position = 5
    reachy_sdk_zeroed.l_arm.wrist.pitch.goal_position = -5
    reachy_sdk_zeroed.l_arm.wrist.yaw.goal_position = 10

    reachy_sdk_zeroed.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.r_arm.shoulder.pitch.present_position, 20, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.shoulder.roll.present_position, -10, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.elbow.yaw.present_position, r_arm_present_positions[2], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.elbow.pitch.present_position, -50, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.wrist.roll.present_position, 30, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.wrist.pitch.present_position, r_arm_present_positions[5], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.wrist.yaw.present_position, r_arm_present_positions[6], 1e-03)

    assert np.isclose(reachy_sdk_zeroed.l_arm.shoulder.pitch.present_position, l_arm_present_positions[0], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.shoulder.roll.present_position, l_arm_present_positions[1], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.yaw.present_position, -20, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.pitch.present_position, l_arm_present_positions[3], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.roll.present_position, 5, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.pitch.present_position, -5, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.yaw.present_position, 10, 1e-03)

    r_arm_present_positions = reachy_sdk_zeroed.r_arm.get_current_positions()
    l_arm_present_positions = reachy_sdk_zeroed.l_arm.get_current_positions()
    neck_present_positions = reachy_sdk_zeroed.head.get_current_positions()
    antenna_left_present_positions = reachy_sdk_zeroed.head.l_antenna.present_position
    r_gripper_present_positions = reachy_sdk_zeroed.r_arm.gripper.present_position

    reachy_sdk_zeroed.r_arm.shoulder.roll.goal_position = 5
    reachy_sdk_zeroed.l_arm.shoulder.pitch.goal_position = 10
    reachy_sdk_zeroed.l_arm.gripper.goal_position = 68
    reachy_sdk_zeroed.head.neck.pitch.goal_position = 20
    reachy_sdk_zeroed.head.r_antenna.goal_position = 27

    r_arm_expected_positions = r_arm_present_positions.copy()
    r_arm_expected_positions[1] = 5
    l_arm_expected_positions = l_arm_present_positions.copy()
    l_arm_expected_positions[0] = 10

    reachy_sdk_zeroed.send_goal_positions()
    time.sleep(0.1)

    assert np.allclose(reachy_sdk_zeroed.r_arm.get_current_positions(), r_arm_expected_positions, 1e-03)
    assert np.allclose(reachy_sdk_zeroed.l_arm.get_current_positions(), l_arm_expected_positions, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.neck.pitch.present_position, 20, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.neck.roll.present_position, neck_present_positions[0], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.neck.yaw.present_position, neck_present_positions[2], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.l_antenna.present_position, antenna_left_present_positions, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.r_antenna.present_position, 27, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.gripper.present_position, r_gripper_present_positions, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.gripper.present_position, 68, 1e-03)

    reachy_sdk_zeroed.r_arm.turn_off()

    reachy_sdk_zeroed.r_arm.shoulder.roll.goal_position = -7
    reachy_sdk_zeroed.r_arm.elbow.pitch.goal_position = -30
    reachy_sdk_zeroed.l_arm.shoulder.pitch.goal_position = 8
    reachy_sdk_zeroed.l_arm.elbow.pitch.goal_position = -40

    reachy_sdk_zeroed.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.r_arm.shoulder.roll.present_position, r_arm_expected_positions[1], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.elbow.pitch.present_position, r_arm_present_positions[3], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.shoulder.pitch.present_position, 8, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.pitch.present_position, -40, 1e-03)

    l_gripper_position = reachy_sdk_zeroed.l_arm.gripper.present_position
    reachy_sdk_zeroed.l_arm.gripper.turn_off()
    time.sleep(0.1)
    reachy_sdk_zeroed.l_arm.elbow.pitch.goal_position = -50
    reachy_sdk_zeroed.l_arm.gripper.goal_position = 110
    reachy_sdk_zeroed.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.pitch.present_position, -50, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.gripper.present_position, l_gripper_position, 1e-03)


@pytest.mark.online
def test_arm_send_goal_positions(reachy_sdk_zeroed: ReachySDK) -> None:
    r_arm_present_positions = reachy_sdk_zeroed.r_arm.get_current_positions()
    l_arm_present_positions = reachy_sdk_zeroed.l_arm.get_current_positions()

    reachy_sdk_zeroed.r_arm.shoulder.pitch.goal_position = 20
    reachy_sdk_zeroed.r_arm.shoulder.roll.goal_position = -10
    reachy_sdk_zeroed.r_arm.elbow.pitch.goal_position = -50
    reachy_sdk_zeroed.r_arm.wrist.roll.goal_position = 30

    reachy_sdk_zeroed.l_arm.elbow.yaw.goal_position = -20
    reachy_sdk_zeroed.l_arm.wrist.roll.goal_position = 5
    reachy_sdk_zeroed.l_arm.wrist.pitch.goal_position = -5
    reachy_sdk_zeroed.l_arm.wrist.yaw.goal_position = 10

    reachy_sdk_zeroed.r_arm.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.r_arm.shoulder.pitch.present_position, 20, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.shoulder.roll.present_position, -10, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.elbow.yaw.present_position, r_arm_present_positions[2], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.elbow.pitch.present_position, -50, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.wrist.roll.present_position, 30, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.wrist.pitch.present_position, r_arm_present_positions[5], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.wrist.yaw.present_position, r_arm_present_positions[6], 1e-03)

    assert np.allclose(reachy_sdk_zeroed.l_arm.get_current_positions(), l_arm_present_positions, 1e-03)

    r_arm_present_positions = reachy_sdk_zeroed.r_arm.get_current_positions()
    reachy_sdk_zeroed.l_arm.send_goal_positions()
    time.sleep(0.1)

    assert np.allclose(reachy_sdk_zeroed.r_arm.get_current_positions(), r_arm_present_positions, 1e-03)

    assert np.isclose(reachy_sdk_zeroed.l_arm.shoulder.pitch.present_position, l_arm_present_positions[0], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.shoulder.roll.present_position, l_arm_present_positions[1], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.yaw.present_position, -20, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.pitch.present_position, l_arm_present_positions[3], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.roll.present_position, 5, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.pitch.present_position, -5, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.yaw.present_position, 10, 1e-03)

    reachy_sdk_zeroed.r_arm.turn_off()
    reachy_sdk_zeroed.r_arm.shoulder.pitch.goal_position = 2
    reachy_sdk_zeroed.r_arm.shoulder.roll.goal_position = 3
    reachy_sdk_zeroed.r_arm.elbow.yaw.goal_position = -4
    reachy_sdk_zeroed.r_arm.elbow.pitch.goal_position = -5

    reachy_sdk_zeroed.r_arm.send_goal_positions()
    time.sleep(0.1)

    assert np.allclose(reachy_sdk_zeroed.r_arm.get_current_positions(), r_arm_present_positions, 1e-03)

    reachy_sdk_zeroed.l_arm.wrist.roll.goal_position = 6
    reachy_sdk_zeroed.l_arm.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.roll.present_position, 6, 1e-03)

    l_arm_present_positions = reachy_sdk_zeroed.l_arm.get_current_positions()
    reachy_sdk_zeroed.l_arm.shoulder.turn_off()
    time.sleep(0.1)

    reachy_sdk_zeroed.l_arm.shoulder.roll.goal_position = 7
    reachy_sdk_zeroed.l_arm.wrist.pitch.goal_position = 8
    reachy_sdk_zeroed.l_arm.gripper.goal_position = 9
    reachy_sdk_zeroed.l_arm.send_goal_positions()
    assert reachy_sdk_zeroed.l_arm.gripper.is_moving()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.l_arm.shoulder.roll.present_position, l_arm_present_positions[1], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.pitch.present_position, 8, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.gripper.present_position, 9, 1e-03)

    reachy_sdk_zeroed.l_arm.gripper.turn_off()
    time.sleep(0.1)

    assert not reachy_sdk_zeroed.l_arm.gripper.is_moving()
    reachy_sdk_zeroed.l_arm.gripper.goal_position = 110
    reachy_sdk_zeroed.l_arm.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.l_arm.gripper.present_position, 9, 1e-03)


@pytest.mark.online
def test_head_send_goal_positions(reachy_sdk_zeroed: ReachySDK) -> None:
    reachy_sdk_zeroed.head.neck.pitch.goal_position = 20
    reachy_sdk_zeroed.head.neck.roll.goal_position = 10
    reachy_sdk_zeroed.head.neck.yaw.goal_position = -5
    reachy_sdk_zeroed.head.l_antenna.goal_position = 27
    reachy_sdk_zeroed.head.r_antenna.goal_position = 33

    reachy_sdk_zeroed.head.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.head.neck.pitch.present_position, 20, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.neck.roll.present_position, 10, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.neck.yaw.present_position, -5, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.l_antenna.present_position, 27, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.r_antenna.present_position, 33, 1e-03)

    reachy_sdk_zeroed.head.neck.turn_off()
    time.sleep(0.1)
    reachy_sdk_zeroed.head.neck.pitch.goal_position = 30
    reachy_sdk_zeroed.head.neck.roll.goal_position = 20
    reachy_sdk_zeroed.head.neck.yaw.goal_position = 10
    reachy_sdk_zeroed.head.l_antenna.goal_position = 37
    reachy_sdk_zeroed.head.r_antenna.goal_position = 43

    reachy_sdk_zeroed.head.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.head.neck.pitch.present_position, 20, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.neck.roll.present_position, 10, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.neck.yaw.present_position, -5, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.l_antenna.present_position, 37, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.r_antenna.present_position, 43, 1e-03)

    reachy_sdk_zeroed.head.neck.turn_on()
    time.sleep(0.1)
    reachy_sdk_zeroed.head.neck.pitch.goal_position = 40

    reachy_sdk_zeroed.head.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.head.neck.pitch.present_position, 40, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.neck.roll.present_position, 10, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.neck.yaw.present_position, -5, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.l_antenna.present_position, 37, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.r_antenna.present_position, 43, 1e-03)


@pytest.mark.online
def test_orbita_send_goal_positions(reachy_sdk_zeroed: ReachySDK) -> None:
    r_arm_present_positions = reachy_sdk_zeroed.r_arm.get_current_positions()

    reachy_sdk_zeroed.r_arm.shoulder.pitch.goal_position = 20
    reachy_sdk_zeroed.r_arm.shoulder.roll.goal_position = -10
    reachy_sdk_zeroed.r_arm.elbow.pitch.goal_position = -50

    reachy_sdk_zeroed.r_arm.shoulder.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.r_arm.shoulder.pitch.present_position, 20, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.shoulder.roll.present_position, -10, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.elbow.pitch.present_position, r_arm_present_positions[3], 1e-03)

    reachy_sdk_zeroed.r_arm.elbow.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.r_arm.shoulder.pitch.present_position, 20, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.shoulder.roll.present_position, -10, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.elbow.pitch.present_position, -50, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.elbow.yaw.present_position, r_arm_present_positions[2], 1e-03)

    reachy_sdk_zeroed.head.neck.pitch.goal_position = 3
    reachy_sdk_zeroed.head.neck.roll.goal_position = 5
    reachy_sdk_zeroed.head.neck.yaw.goal_position = 7

    l_arm_present_positions = reachy_sdk_zeroed.l_arm.get_current_positions()
    reachy_sdk_zeroed.l_arm.wrist.roll.goal_position = 6

    reachy_sdk_zeroed.head.neck.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.head.neck.pitch.present_position, 3, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.neck.roll.present_position, 5, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.neck.yaw.present_position, 7, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.roll.present_position, l_arm_present_positions[4], 1e-03)

    reachy_sdk_zeroed.l_arm.wrist.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.roll.present_position, 6, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.pitch.present_position, l_arm_present_positions[5], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.yaw.present_position, l_arm_present_positions[6], 1e-03)

    reachy_sdk_zeroed.r_arm.turn_off()
    time.sleep(0.1)
    r_arm_present_positions = reachy_sdk_zeroed.r_arm.get_current_positions()
    reachy_sdk_zeroed.r_arm.shoulder.pitch.goal_position = 2

    reachy_sdk_zeroed.r_arm.shoulder.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.r_arm.shoulder.pitch.present_position, r_arm_present_positions[0], 1e-03)
    assert np.isclose(reachy_sdk_zeroed.r_arm.shoulder.roll.present_position, r_arm_present_positions[1], 1e-03)


@pytest.mark.online
def test_gripper_send_goal_positions(reachy_sdk_zeroed: ReachySDK) -> None:
    l_gripper_present_position = reachy_sdk_zeroed.l_arm.gripper.present_position

    reachy_sdk_zeroed.r_arm.gripper.goal_position = 68
    reachy_sdk_zeroed.l_arm.gripper.goal_position = 110

    reachy_sdk_zeroed.r_arm.gripper.send_goal_positions()
    assert reachy_sdk_zeroed.r_arm.gripper.is_moving()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.r_arm.gripper.present_position, 68, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.gripper.present_position, l_gripper_present_position, 1e-03)
    time.sleep(0.1)

    reachy_sdk_zeroed.l_arm.gripper.send_goal_positions()
    assert reachy_sdk_zeroed.l_arm.gripper.is_moving()
    assert not reachy_sdk_zeroed.r_arm.gripper.is_moving()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.r_arm.gripper.present_position, 68, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.l_arm.gripper.present_position, 110, 1e-03)

    reachy_sdk_zeroed.r_arm.gripper.turn_off()
    time.sleep(0.1)
    r_gripper_present_position = reachy_sdk_zeroed.r_arm.gripper.present_position
    assert not reachy_sdk_zeroed.l_arm.gripper.is_moving()

    reachy_sdk_zeroed.r_arm.gripper.goal_position = 80
    reachy_sdk_zeroed.r_arm.gripper.send_goal_positions()
    assert not reachy_sdk_zeroed.r_arm.gripper.is_moving()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.r_arm.gripper.present_position, r_gripper_present_position, 1e-03)


@pytest.mark.online
def test_antenna_send_goal_positions(reachy_sdk_zeroed: ReachySDK) -> None:
    r_antenna_present_position = reachy_sdk_zeroed.head.r_antenna.present_position
    reachy_sdk_zeroed.head.l_antenna.goal_position = 27
    reachy_sdk_zeroed.head.r_antenna.goal_position = 33

    reachy_sdk_zeroed.head.l_antenna.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.head.l_antenna.present_position, 27, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.r_antenna.present_position, r_antenna_present_position, 1e-03)

    reachy_sdk_zeroed.head.r_antenna.turn_off()
    time.sleep(0.1)

    reachy_sdk_zeroed.head.r_antenna.goal_position = 37
    reachy_sdk_zeroed.head.r_antenna.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.head.r_antenna.present_position, r_antenna_present_position, 1e-03)

    reachy_sdk_zeroed.head.r_antenna.turn_on()
    time.sleep(0.1)

    reachy_sdk_zeroed.head.r_antenna.send_goal_positions()
    time.sleep(0.1)

    assert np.isclose(reachy_sdk_zeroed.head.r_antenna.present_position, 37, 1e-03)


@pytest.mark.online
def test_send_goal_positions_loop(reachy_sdk_zeroed: ReachySDK) -> None:
    freq = 100
    dt = 1 / freq
    data: Dict[str, List[float]] = {j_name: [] for j_name in reachy_sdk_zeroed.joints.keys()}
    data["timestamp"] = []

    reachy_sdk_zeroed.r_arm.goto_posture("elbow_90")
    reachy_sdk_zeroed.l_arm.goto_posture("elbow_90")
    reachy_sdk_zeroed.r_arm.gripper.goto(100, percentage=True)
    reachy_sdk_zeroed.l_arm.gripper.goto(100, percentage=True)
    reachy_sdk_zeroed.head.goto([20, 30, 20])
    reachy_sdk_zeroed.head.l_antenna.goto(60)
    reachy_sdk_zeroed.head.r_antenna.goto(-60)
    reachy_sdk_zeroed.r_arm.goto_posture("default")
    reachy_sdk_zeroed.l_arm.goto_posture("default")
    reachy_sdk_zeroed.r_arm.gripper.goto(0, percentage=True)
    reachy_sdk_zeroed.l_arm.gripper.goto(0, percentage=True)
    reachy_sdk_zeroed.head.goto([0, 0, 0])
    reachy_sdk_zeroed.head.l_antenna.goto(0)
    last_goto = reachy_sdk_zeroed.head.r_antenna.goto(0)

    t0 = time.time()
    while not reachy_sdk_zeroed.is_goto_finished(last_goto):
        data["timestamp"].append(time.time() - t0)
        for j_name, joint in reachy_sdk_zeroed.joints.items():
            data[j_name].append(joint.present_position)
        time.sleep(dt)
    duration = time.time() - t0

    t0_loop = time.time()
    for ite in range(len(data["timestamp"])):
        for j_name, joint in reachy_sdk_zeroed.joints.items():
            joint.goal_position = data[j_name][ite]

        reachy_sdk_zeroed.send_goal_positions()
        time.sleep(dt)
    assert np.isclose(time.time() - t0_loop, duration, 1e-01)
