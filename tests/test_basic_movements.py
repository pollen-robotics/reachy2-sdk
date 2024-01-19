import time

import numpy as np
import numpy.typing as npt
import pytest
from pyquaternion import Quaternion
from reachy2_sdk_api.goto_pb2 import GoalStatus

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
def test_basic(reachy_sdk_zeroed: ReachySDK) -> None:
    goal_position = -90
    reachy_sdk_zeroed.r_arm.elbow.pitch.goal_position = goal_position
    time.sleep(2)
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
def test_square(reachy_sdk_zeroed: ReachySDK) -> None:
    # In A position, the effector is at (0.3, -0,4, -0.3) in the world frame
    # In B position, the effector is at (0.3, -0.4, 0) in the world frame
    # In C position, the effector is at (0.3, -0.1, 0.0) in the world frame
    # In D position, the effector is at (0.3, -0.1, -0.3) in the world frame

    # Going from A to B
    for z in np.arange(-0.3, 0.01, 0.01):
        target_pose = build_pose_matrix(0.3, -0.4, z)
        ik = reachy_sdk_zeroed.r_arm.inverse_kinematics(target_pose)

        for joint, goal_pos in zip(reachy_sdk_zeroed.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        time.sleep(0.1)

    time.sleep(2)

    B = build_pose_matrix(0.3, -0.4, 0)
    current_pos = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(current_pos, B, atol=1e-03)

    # Going from B to C
    for y in np.arange(-0.4, -0.1, 0.01):
        target_pose = build_pose_matrix(0.3, y, 0.0)
        ik = reachy_sdk_zeroed.r_arm.inverse_kinematics(target_pose)

        for joint, goal_pos in zip(reachy_sdk_zeroed.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        time.sleep(0.1)

    time.sleep(2)

    C = build_pose_matrix(0.3, -0.1, 0)
    current_pos = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(current_pos, C, atol=1e-03)

    # Going from C to D
    for z in np.arange(0.0, -0.31, -0.01):
        target_pose = build_pose_matrix(0.3, -0.1, z)
        ik = reachy_sdk_zeroed.r_arm.inverse_kinematics(target_pose)

        for joint, goal_pos in zip(reachy_sdk_zeroed.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        time.sleep(0.1)

    time.sleep(2)

    D = build_pose_matrix(0.3, -0.1, -0.3)
    current_pos = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(current_pos, D, atol=1e-03)

    # Going from D to A
    for y in np.arange(-0.1, -0.4, -0.01):
        target_pose = build_pose_matrix(0.3, y, -0.3)
        ik = reachy_sdk_zeroed.r_arm.inverse_kinematics(target_pose)

        for joint, goal_pos in zip(reachy_sdk_zeroed.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        time.sleep(0.1)

    time.sleep(2)

    A = build_pose_matrix(0.3, -0.4, -0.3)
    current_pos = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(current_pos, A, atol=1e-03)


def is_goto_finished(reachy: ReachySDK, id: int) -> bool:
    state = reachy.r_arm.get_goto_state(id)
    result = bool(
        state.goal_status == GoalStatus.STATUS_ABORTED
        or state.goal_status == GoalStatus.STATUS_CANCELED
        or state.goal_status == GoalStatus.STATUS_SUCCEEDED
    )
    return result


@pytest.mark.online
def test_head_movements(reachy_sdk_zeroed: ReachySDK) -> None:
    q0 = Quaternion(axis=[1, 0, 0], angle=np.pi / 6.0)  # Rotate 30 about X
    id = reachy_sdk_zeroed.head.orient(q0, duration=1)

    while not is_goto_finished(reachy_sdk_zeroed, id):
        time.sleep(0.1)

    q1 = reachy_sdk_zeroed.head.get_orientation()
    assert np.isclose(Quaternion.distance(q0, q1), 0, atol=1e-04)

    id = reachy_sdk_zeroed.head.rotate_to(roll=0, pitch=60, yaw=0, duration=1)
    q2 = Quaternion(axis=[0, 1, 0], angle=np.pi / 3.0)  # Rotate 60 about Y

    while not is_goto_finished(reachy_sdk_zeroed, id):
        time.sleep(0.1)

    q3 = reachy_sdk_zeroed.head.get_orientation()
    assert np.isclose(Quaternion.distance(q2, q3), 0, atol=1e-04)


@pytest.mark.online
def test_cancel_goto(reachy_sdk_zeroed: ReachySDK) -> None:
    """
    # is already zeroed
    id = reachy_sdk_zeroed.head.rotate_to(0, 0, 0, duration=1.0)
    while not is_goto_finised(reachy_sdk_zeroed, id):
        time.sleep(0.1)
    """

    req = reachy_sdk_zeroed.head.rotate_to(0, 40, 0, duration=10, interpolation_mode="linear")
    time.sleep(2)
    cancel = reachy_sdk_zeroed.head.cancel_goto_by_id(req)
    assert cancel.ack

    # 40*2/10 -> 8° ideally. but timing is not precise
    assert np.isclose(reachy_sdk_zeroed.head.neck.pitch.present_position, 8, atol=1)  # ??
    assert np.isclose(reachy_sdk_zeroed.head.neck.roll.present_position, 0)
    assert np.isclose(reachy_sdk_zeroed.head.neck.yaw.present_position, 0)

    req2 = reachy_sdk_zeroed.l_arm.goto_joints([15, 10, 20, -50, 10, 10, 20], duration=10, interpolation_mode="linear")
    time.sleep(2)
    cancel2 = reachy_sdk_zeroed.head.cancel_goto_by_id(req2)
    assert cancel2.ack
    assert np.isclose(reachy_sdk_zeroed.l_arm.shoulder.pitch.present_position, 3, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.shoulder.roll.present_position, 2.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.yaw.present_position, 4.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.elbow.pitch.present_position, -10, atol=1)  # should be 10
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.roll.present_position, 2.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.pitch.present_position, 2.0, atol=1)
    assert np.isclose(reachy_sdk_zeroed.l_arm.wrist.yaw.present_position, 4.0, atol=1)
