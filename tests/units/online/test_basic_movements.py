import time
from typing import List

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
    assert np.isclose(reachy_sdk_zeroed.r_arm.elbow.pitch.present_position, goal_position, 1e-03)


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

    assert np.allclose(reachy_sdk_zeroed.l_arm.get_joints_positions(), expected_pos1, atol=1e-03)

    expected_pos2 = [15, 10, 20, -50, 10, 10, 20]
    id = reachy_sdk_zeroed.l_arm.goto_joints(expected_pos2, duration=3)
    while not is_goto_finished(reachy_sdk_zeroed, id):
        time.sleep(0.1)
    assert np.allclose(reachy_sdk_zeroed.l_arm.get_joints_positions(), expected_pos2, atol=1e-03)


@pytest.mark.online
def test_send_goal_positions(reachy_sdk_zeroed: ReachySDK) -> None:
    def go_to_pose(reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str) -> List[float]:
        ik: List[float] = []
        if arm == "r_arm":
            ik = reachy.r_arm.inverse_kinematics(pose)
            for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
                joint.goal_position = goal_pos
        elif arm == "l_arm":
            ik = reachy.l_arm.inverse_kinematics(pose)
            for joint, goal_pos in zip(reachy.l_arm.joints.values(), ik):
                joint.goal_position = goal_pos
        reachy.send_goal_positions()
        return ik

    def make_circle(
        reachy: ReachySDK,
        center: npt.NDArray[np.float64],
        radius: float,
        nbr_points: int = 100,
    ) -> None:
        Y_r = center[1] + radius * np.cos(np.linspace(0, 2 * np.pi, nbr_points))
        Z = center[2] + radius * np.sin(np.linspace(0, 2 * np.pi, nbr_points))
        X = center[0] * np.ones(nbr_points)
        Y_l = -center[1] + radius * np.cos(np.linspace(0, 2 * np.pi, nbr_points))

        prev_goal = None
        prev_l_goal = None

        for i in range(nbr_points):
            if prev_goal is not None:
                assert np.allclose(reachy.r_arm.get_joints_positions(), prev_goal, atol=1e-03)
            if prev_l_goal is not None:
                assert np.allclose(reachy.l_arm.get_joints_positions(), prev_l_goal, atol=1e-03)
            pose = build_pose_matrix(X[i], Y_r[i], Z[i])
            prev_goal = go_to_pose(reachy, pose, "r_arm")

            l_pose = build_pose_matrix(X[i], Y_l[i], Z[i])
            prev_l_goal = go_to_pose(reachy, l_pose, "l_arm")

            time.sleep(0.05)

    center = np.array([0.4, -0.4, -0.2])
    radius = 0.15
    make_circle(reachy_sdk_zeroed, center, radius)
