import time

import numpy as np
import numpy.typing as npt
import pytest

from src.reachy2_sdk.reachy_sdk import ReachySDK
from pyquaternion import Quaternion


@pytest.mark.online
def test_basic() -> None:
    reachy = ReachySDK(host="localhost")
    assert reachy.grpc_status == "connected"

    assert reachy.turn_on()

    for joint in reachy.joints.values():
        joint.goal_position = 0
        time.sleep(0.01)

    goal_position = -90
    reachy.r_arm.elbow.pitch.goal_position = goal_position
    time.sleep(2)
    assert reachy.r_arm.elbow.pitch.present_position == goal_position

    assert reachy.turn_off()

    reachy.disconnect()
    ReachySDK.clear()


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
def test_square() -> None:
    reachy = ReachySDK(host="localhost")
    assert reachy.grpc_status == "connected"

    assert reachy.turn_on()

    # reset
    for joint in reachy.joints.values():
        joint.goal_position = 0
        time.sleep(0.01)

    # In A position, the effector is at (0.3, -0,4, -0.3) in the world frame
    # In B position, the effector is at (0.3, -0.4, 0) in the world frame
    # In C position, the effector is at (0.3, -0.1, 0.0) in the world frame
    # In D position, the effector is at (0.3, -0.1, -0.3) in the world frame

    # Going from A to B
    for z in np.arange(-0.3, 0.11, 0.01):
        jacobian = build_pose_matrix(0.3, -0.4, z)
        ik = reachy.r_arm.inverse_kinematics(jacobian)

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
            time.sleep(0.01)

    B = build_pose_matrix(0.3, -0.4, 0)
    current_pos = reachy.r_arm.forward_kinematics()
    assert np.allclose(current_pos, B, atol=1e-01)

    # Going from B to C
    for y in np.arange(-0.4, -0.1, 0.01):
        jacobian = build_pose_matrix(0.3, y, 0.0)
        ik = reachy.r_arm.inverse_kinematics(jacobian)

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
            time.sleep(0.01)

    C = build_pose_matrix(0.3, -0.1, 0)
    current_pos = reachy.r_arm.forward_kinematics()
    assert np.allclose(current_pos, C, atol=1e-01)

    # Going from C to D
    for z in np.arange(0.0, -0.31, -0.01):
        jacobian = build_pose_matrix(0.3, -0.1, z)
        ik = reachy.r_arm.inverse_kinematics(jacobian)

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
            time.sleep(0.01)

    D = build_pose_matrix(0.3, -0.1, -0.3)
    current_pos = reachy.r_arm.forward_kinematics()
    assert np.allclose(current_pos, D, atol=1e-01)

    # Going from D to A
    for y in np.arange(-0.1, -0.4, -0.01):
        jacobian = build_pose_matrix(0.3, y, -0.3)
        ik = reachy.r_arm.inverse_kinematics(jacobian)

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
            time.sleep(0.01)

    A = build_pose_matrix(0.3, -0.4, -0.3)
    current_pos = reachy.r_arm.forward_kinematics()
    assert np.allclose(current_pos, A, atol=1e-01)

    assert reachy.turn_off()

    reachy.disconnect()
    ReachySDK.clear()


@pytest.mark.online
def test_head_movements() -> None:
    reachy = ReachySDK(host="localhost")
    assert reachy.grpc_status == "connected"

    assert reachy.turn_on()

    q0 = Quaternion(axis=[1, 0, 0], angle=np.pi / 6.0)  # Rotate 30 about X
    reachy.head.orient(q0, duration=1)

    time.sleep(1)

    q1 = reachy.head.get_orientation()

    assert Quaternion.distance(q0, q1) < 1e-05

    reachy.head.rotate_to(roll=0, pitch=60, yaw=0, duration=1)
    q2 = Quaternion(axis=[0, 1, 0], angle=np.pi / 3.0)  # Rotate 60 about Y

    time.sleep(1)
    q3 = reachy.head.get_orientation()
    assert Quaternion.distance(q2, q3) < 1e-05

    assert reachy.turn_off()

    reachy.disconnect()
    ReachySDK.clear()
