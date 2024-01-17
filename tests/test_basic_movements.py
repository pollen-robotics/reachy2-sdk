import time

import numpy as np
import numpy.typing as npt
import pytest
from pyquaternion import Quaternion

from src.reachy2_sdk.reachy_sdk import ReachySDK


@pytest.fixture(scope="module")
def reachy_sdk() -> ReachySDK:
    reachy = ReachySDK(host="localhost")
    assert reachy.grpc_status == "connected"

    assert reachy.turn_on()

    yield reachy

    assert reachy.turn_off()
    print("teardown")
    reachy.disconnect()
    ReachySDK.clear()


@pytest.mark.online
def test_basic(reachy_sdk: ReachySDK) -> None:
    for joint in reachy_sdk.joints.values():
        joint.goal_position = 0
        time.sleep(0.01)

    goal_position = -90
    reachy_sdk.r_arm.elbow.pitch.goal_position = goal_position
    time.sleep(2)
    assert reachy_sdk.r_arm.elbow.pitch.present_position == goal_position


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
def test_square(reachy_sdk: ReachySDK) -> None:
    # reset
    for joint in reachy_sdk.joints.values():
        joint.goal_position = 0
        time.sleep(0.01)

    # In A position, the effector is at (0.3, -0,4, -0.3) in the world frame
    # In B position, the effector is at (0.3, -0.4, 0) in the world frame
    # In C position, the effector is at (0.3, -0.1, 0.0) in the world frame
    # In D position, the effector is at (0.3, -0.1, -0.3) in the world frame

    # Going from A to B
    for z in np.arange(-0.3, 0.01, 0.01):
        target_pose = build_pose_matrix(0.3, -0.4, z)
        ik = reachy_sdk.r_arm.inverse_kinematics(target_pose)

        for joint, goal_pos in zip(reachy_sdk.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        time.sleep(0.1)

    time.sleep(2)

    B = build_pose_matrix(0.3, -0.4, 0)
    current_pos = reachy_sdk.r_arm.forward_kinematics()
    assert np.allclose(current_pos, B, atol=1e-03)

    # Going from B to C
    for y in np.arange(-0.4, -0.1, 0.01):
        target_pose = build_pose_matrix(0.3, y, 0.0)
        ik = reachy_sdk.r_arm.inverse_kinematics(target_pose)

        for joint, goal_pos in zip(reachy_sdk.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        time.sleep(0.1)

    time.sleep(2)

    C = build_pose_matrix(0.3, -0.1, 0)
    current_pos = reachy_sdk.r_arm.forward_kinematics()
    assert np.allclose(current_pos, C, atol=1e-03)

    # Going from C to D
    for z in np.arange(0.0, -0.31, -0.01):
        target_pose = build_pose_matrix(0.3, -0.1, z)
        ik = reachy_sdk.r_arm.inverse_kinematics(target_pose)

        for joint, goal_pos in zip(reachy_sdk.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        time.sleep(0.1)

    time.sleep(2)

    D = build_pose_matrix(0.3, -0.1, -0.3)
    current_pos = reachy_sdk.r_arm.forward_kinematics()
    assert np.allclose(current_pos, D, atol=1e-03)

    # Going from D to A
    for y in np.arange(-0.1, -0.4, -0.01):
        target_pose = build_pose_matrix(0.3, y, -0.3)
        ik = reachy_sdk.r_arm.inverse_kinematics(target_pose)

        for joint, goal_pos in zip(reachy_sdk.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        time.sleep(0.1)

    time.sleep(2)

    A = build_pose_matrix(0.3, -0.4, -0.3)
    current_pos = reachy_sdk.r_arm.forward_kinematics()
    assert np.allclose(current_pos, A, atol=1e-03)


@pytest.mark.online
def test_head_movements(reachy_sdk: ReachySDK) -> None:
    q0 = Quaternion(axis=[1, 0, 0], angle=np.pi / 6.0)  # Rotate 30 about X
    reachy_sdk.head.orient(q0, duration=1)
    time.sleep(1.1)
    q1 = reachy_sdk.head.get_orientation()
    assert Quaternion.distance(q0, q1) < 1e-05

    reachy_sdk.head.rotate_to(roll=0, pitch=60, yaw=0, duration=1)
    q2 = Quaternion(axis=[0, 1, 0], angle=np.pi / 3.0)  # Rotate 60 about Y
    time.sleep(1.1)
    q3 = reachy_sdk.head.get_orientation()
    assert Quaternion.distance(q2, q3) < 1e-05


@pytest.mark.online
def test_cancel_goto(reachy_sdk: ReachySDK) -> None:
    reachy_sdk.head.rotate_to(0, 0, 0, duration=1.0)
    time.sleep(1)
    req = reachy_sdk.head.rotate_to(0, 40, 0, duration=10, interpolation_mode="linear")
    time.sleep(2)
    cancel = reachy_sdk.head.cancel_goto_by_id(req)
    assert cancel.ack
    assert abs(reachy_sdk.head.neck.pitch.present_position - 8.0) < 1
    assert reachy_sdk.head.neck.roll.present_position < 1e-04
    assert reachy_sdk.head.neck.yaw.present_position < 1e-04

    req2 = reachy_sdk.l_arm.goto_joints([0, 0, 0, 0, 0, 0, 0], duration=1, interpolation_mode="linear")
    time.sleep(1)
    req2 = reachy_sdk.l_arm.goto_joints([15, 10, 20, -50, 10, 10, 20], duration=10, interpolation_mode="linear")
    time.sleep(2)
    cancel2 = reachy_sdk.head.cancel_goto_by_id(req2)
    assert cancel2.ack
    assert abs(reachy_sdk.l_arm.shoulder.pitch.present_position - 3.0) < 1
    assert abs(reachy_sdk.l_arm.shoulder.roll.present_position - 2.0) < 1
    assert abs(reachy_sdk.l_arm.elbow.yaw.present_position - 4.0) < 1
    assert abs(reachy_sdk.l_arm.elbow.pitch.present_position + 10.0) < 1
    assert abs(reachy_sdk.l_arm.wrist.roll.present_position - 2.0) < 1
    assert abs(reachy_sdk.l_arm.wrist.pitch.present_position - 2.0) < 1
    assert abs(reachy_sdk.l_arm.wrist.yaw.present_position - 4.0) < 1
