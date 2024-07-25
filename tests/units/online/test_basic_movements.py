import time

import numpy as np
import numpy.typing as npt
import pytest
from pyquaternion import Quaternion
from reachy2_sdk_api.goto_pb2 import GoalStatus, GoToId

from reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.online
def test_basic(reachy_sdk_zeroed: ReachySDK) -> None:
    goal_position = -90
    reachy_sdk_zeroed.r_arm.elbow.pitch.goal_position = goal_position
    reachy_sdk_zeroed.send_goal_positions()
    time.sleep(1)
    assert reachy_sdk_zeroed.r_arm.elbow.pitch.present_position == goal_position


def build_pose_matrix(x: float, y: float, z: float) -> npt.NDArray[np.float64]:
    # The effector is always at the same orientation in the world frame
    return np.array(
        [
            [0, 0, -1, x],
            [0, 1, 0, y],
            [1, 0, 0, z],
            [0, 0, 0, 1],
        ]
    )


@pytest.mark.online
def test_triangle(reachy_sdk_zeroed: ReachySDK) -> None:
    # In A position, the effector is at (0.3, -0,4, -0.3) in the world frame
    # In B position, the effector is at (0.3, -0.4, 0) in the world frame
    # In C position, the effector is at (0.3, -0.1, -0.3) in the world frame

    A = build_pose_matrix(0.3, -0.4, -0.3)
    m1 = reachy_sdk_zeroed.r_arm.goto_from_matrix(A)

    while not is_goto_finished(reachy_sdk_zeroed, m1):
        time.sleep(0.1)

    current_pos = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(current_pos, A, atol=1e-03)

    B = build_pose_matrix(0.3, -0.4, 0)
    m2 = reachy_sdk_zeroed.r_arm.goto_from_matrix(B)

    while not is_goto_finished(reachy_sdk_zeroed, m2):
        time.sleep(0.1)

    current_pos = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(current_pos, B, atol=1e-03)

    C = build_pose_matrix(0.3, -0.2, -0.3)
    m3 = reachy_sdk_zeroed.r_arm.goto_from_matrix(C)

    while not is_goto_finished(reachy_sdk_zeroed, m3):
        time.sleep(0.1)

    current_pos = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(current_pos, C, atol=1e-03)

    A = build_pose_matrix(0.3, -0.4, -0.3)
    m1 = reachy_sdk_zeroed.r_arm.goto_from_matrix(A)

    while not is_goto_finished(reachy_sdk_zeroed, m1):
        time.sleep(0.1)

    current_pos = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(current_pos, A, atol=1e-03)


def is_goto_finished(reachy: ReachySDK, id: GoToId) -> bool:
    state = reachy._get_move_state(id)
    result = bool(
        state.goal_status == GoalStatus.STATUS_ABORTED
        or state.goal_status == GoalStatus.STATUS_CANCELED
        or state.goal_status == GoalStatus.STATUS_SUCCEEDED
    )
    return result


@pytest.mark.online
def test_head_movements(reachy_sdk_zeroed: ReachySDK) -> None:
    reachy_sdk_zeroed.head.turn_on()
    q0 = Quaternion(axis=[1, 0, 0], degrees=30)
    id = reachy_sdk_zeroed.head.orient(q0, duration=1)

    while not is_goto_finished(reachy_sdk_zeroed, id):
        time.sleep(0.1)

    q1 = reachy_sdk_zeroed.head.get_orientation()
    assert np.isclose(Quaternion.distance(q0, q1), 0, atol=1e-04)

    id = reachy_sdk_zeroed.head.goto_joints([0, 60, 0], duration=1)
    q2 = Quaternion(axis=[0, 1, 0], degrees=70)  # 10 degrees between joint and cartesian spaces

    while not is_goto_finished(reachy_sdk_zeroed, id):
        time.sleep(0.1)

    q3 = reachy_sdk_zeroed.head.get_orientation()
    assert np.isclose(Quaternion.distance(q2, q3), 0, atol=1e-04)

    id = reachy_sdk_zeroed.head.look_at(10000, 10000, 0, duration=1)
    q4 = Quaternion(axis=[0, 0, 1], degrees=45)

    while not is_goto_finished(reachy_sdk_zeroed, id):
        time.sleep(0.1)

    q5 = reachy_sdk_zeroed.head.get_orientation()
    assert np.isclose(Quaternion.distance(q4, q5), 0, atol=1e-04)

    id = reachy_sdk_zeroed.head.look_at(10000, 0, -10000, duration=1)
    q6 = Quaternion(axis=[0, 1, 0], degrees=45)

    while not is_goto_finished(reachy_sdk_zeroed, id):
        time.sleep(0.1)

    q7 = reachy_sdk_zeroed.head.get_orientation()
    assert np.isclose(Quaternion.distance(q6, q7), 0, atol=1e-04)


@pytest.mark.online
def test_basic_get_positions(reachy_sdk_zeroed: ReachySDK) -> None:
    expected_pos1 = [0, 0, 0, 0, 0, 0, 0]

    assert np.allclose(reachy_sdk_zeroed.l_arm.get_joints_positions(), expected_pos1, atol=1e-01)

    expected_pos2 = [15, 10, 20, -50, 10, 10, 20]
    id = reachy_sdk_zeroed.l_arm.goto_joints(expected_pos2, duration=3)
    while not is_goto_finished(reachy_sdk_zeroed, id):
        time.sleep(0.1)
    assert np.allclose(reachy_sdk_zeroed.l_arm.get_joints_positions(), expected_pos2, atol=1e-01)
