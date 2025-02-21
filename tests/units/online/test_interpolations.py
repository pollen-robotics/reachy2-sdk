import time
from threading import Thread
from typing import List

import numpy as np
import numpy.typing as npt
import pytest

from reachy2_sdk.reachy_sdk import ReachySDK

from .test_basic_movements import build_pose_matrix


class LoopThread:
    def __init__(self, reachy: ReachySDK, arm: str):
        self.reachy = reachy
        self.arm = arm
        self.running = False
        self.forward_list = []
        self.thread = Thread(
            target=self.loop_forward_function,
            args=(
                self.reachy,
                self.arm,
            ),
        )

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def loop_forward_function(self, reachy: ReachySDK, arm: str):
        while self.running:
            if arm == "r_arm":
                self.forward_list.append(reachy.r_arm.forward_kinematics())
            else:
                self.forward_list.append(reachy.l_arm.forward_kinematics())
            time.sleep(0.5)


@pytest.mark.online
def test_send_cartesian_interpolation_linear(reachy_sdk_zeroed: ReachySDK) -> None:
    def test_pose_trajectory(
        A: npt.NDArray[np.float64], B: npt.NDArray[np.float64], duration: float
    ) -> List[npt.NDArray[np.float64]]:
        reachy_sdk_zeroed.r_arm.goto(A, wait=True)
        t = LoopThread(reachy_sdk_zeroed, "r_arm")
        t.start()
        tic = time.time()
        reachy_sdk_zeroed.r_arm.send_cartesian_interpolation(B, duration=duration, precision_distance_xyz=0.005)
        elapsed_time = time.time() - tic
        t.stop()
        assert np.isclose(elapsed_time, duration, 1)
        return t.forward_list

    xA = 0.3
    yA = -0.2
    zA = -0.3
    xB = xA
    yB = -0.4
    zB = zA
    A = build_pose_matrix(xA, yA, zA)
    B = build_pose_matrix(xB, yB, zB)
    inter_poses = test_pose_trajectory(A, B, 3.0)
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-03)
    for pose in inter_poses:
        assert np.isclose(pose[0, 3], xB, 1e-03)
        assert (pose[1, 3] <= max(yA, yB) or np.isclose(pose[1, 3], max(yA, yB), 1e-03)) and (
            pose[1, 3] >= min(yA, yB) or np.isclose(pose[1, 3], min(yA, yB), 1e-03)
        )
        assert np.isclose(pose[2, 3], zB, 1e-03)
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-03)

    xA = 0.3
    yA = -0.3
    zA = -0.3
    xB = 0.4
    yB = yA
    zB = -0.1
    A = build_pose_matrix(xA, yA, zA)
    B = build_pose_matrix(xB, yB, zB)
    inter_poses = test_pose_trajectory(A, B, 2.0)
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-03)
    for pose in inter_poses:
        assert (pose[0, 3] <= max(xA, xB) or np.isclose(pose[0, 3], max(xA, xB), 1e-03)) and (
            pose[0, 3] >= min(xA, xB) or np.isclose(pose[0, 3], min(xA, xB), 1e-03)
        )
        assert np.isclose(pose[1, 3], yB, 1e-03)
        assert (pose[2, 3] <= max(zA, zB) or np.isclose(pose[2, 3], max(zA, zB), 1e-03)) and (
            pose[2, 3] >= min(zA, zB) or np.isclose(pose[2, 3], max(zA, zB), 1e-03)
        )
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-03)


@pytest.mark.online
def test_send_cartesian_interpolation_circular(reachy_sdk_zeroed: ReachySDK) -> None:
    def test_pose_trajectory(
        A: npt.NDArray[np.float64], B: npt.NDArray[np.float64], duration: float, arc_direction: str
    ) -> List[npt.NDArray[np.float64]]:
        reachy_sdk_zeroed.r_arm.goto(A, wait=True)
        t = LoopThread(reachy_sdk_zeroed, "r_arm")
        t.start()
        tic = time.time()
        reachy_sdk_zeroed.r_arm.send_cartesian_interpolation(
            B, duration=duration, arc_direction=arc_direction, precision_distance_xyz=0.005
        )
        elapsed_time = time.time() - tic
        t.stop()
        assert np.isclose(elapsed_time, duration, 1)
        return t.forward_list

    xA = 0.3
    yA = -0.2
    zA = -0.3
    xB = xA
    yB = -0.4
    zB = zA
    A = build_pose_matrix(xA, yA, zA)
    B = build_pose_matrix(xB, yB, zB)

    # Test below
    inter_poses = test_pose_trajectory(A, B, 3.0, arc_direction="below")
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-03)
    went_down = False
    for pose in inter_poses:
        assert np.isclose(pose[0, 3], xB, 1e-03)
        assert (pose[1, 3] <= max(yA, yB) or np.isclose(pose[1, 3], max(yA, yB), 1e-03)) and (
            pose[1, 3] >= min(yA, yB) or np.isclose(pose[1, 3], min(yA, yB), 1e-03)
        )
        z_limit = zB - abs(yA - yB) / 2
        assert (pose[2, 3] <= zB or np.isclose(pose[2, 3], zB, 1e-03)) and (
            pose[2, 3] >= z_limit or np.isclose(pose[2, 3], z_limit, 1e-03)
        )
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-03)
        if pose[2, 3] < z_limit + 0.05:
            went_down = True
    assert went_down

    # Test above
    inter_poses = test_pose_trajectory(A, B, 3.0, arc_direction="above")
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-03)
    went_up = False
    for pose in inter_poses:
        assert np.isclose(pose[0, 3], xB, 1e-03)
        assert (pose[1, 3] <= max(yA, yB) or np.isclose(pose[1, 3], max(yA, yB), 1e-03)) and (
            pose[1, 3] >= min(yA, yB) or np.isclose(pose[1, 3], min(yA, yB), 1e-03)
        )
        z_limit = zB + abs(yA - yB) / 2
        assert (pose[2, 3] >= zB or np.isclose(pose[2, 3], zB, 1e-03)) and (
            pose[2, 3] <= z_limit or np.isclose(pose[2, 3], z_limit, 1e-03)
        )
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-03)
        if pose[2, 3] > z_limit - 0.05:
            went_up = True
    assert went_up

    # Test front
    inter_poses = test_pose_trajectory(A, B, 3.0, arc_direction="front")
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-03)
    went_front = False
    for pose in inter_poses:
        x_limit = xB + abs(yA - yB) / 2
        assert (pose[0, 3] >= xB or np.isclose(pose[0, 3], xB, 1e-03)) and (
            pose[0, 3] <= x_limit or np.isclose(pose[0, 3], x_limit, 1e-03)
        )
        if pose[0, 3] > x_limit - 0.05:
            went_front = True
        assert (pose[1, 3] <= max(yA, yB) or np.isclose(pose[1, 3], max(yA, yB), 1e-03)) and (
            pose[1, 3] >= min(yA, yB) or np.isclose(pose[1, 3], min(yA, yB), 1e-03)
        )
        assert np.isclose(pose[2, 3], zB, 1e-03)
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-03)
    assert went_front

    # Test back
    inter_poses = test_pose_trajectory(A, B, 3.0, arc_direction="back")
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-03)
    went_back = False
    for pose in inter_poses:
        x_limit = xB - abs(yA - yB) / 2
        assert (pose[0, 3] <= xB or np.isclose(pose[0, 3], xB, 1e-03)) and (
            pose[0, 3] >= x_limit or np.isclose(pose[0, 3], x_limit, 1e-03)
        )
        if pose[0, 3] < x_limit + 0.05:
            went_back = True
        assert (pose[1, 3] <= max(yA, yB) or np.isclose(pose[1, 3], max(yA, yB), 1e-03)) and (
            pose[1, 3] >= min(yA, yB) or np.isclose(pose[1, 3], min(yA, yB), 1e-03)
        )
        assert np.isclose(pose[2, 3], zB, 1e-03)
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-03)
    assert went_back

    # Test right
    inter_poses = test_pose_trajectory(A, B, 3.0, arc_direction="right")
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-03)
    for pose in inter_poses:
        assert np.isclose(pose[0, 3], xB, 1e-03)
        assert (pose[1, 3] <= max(yA, yB) or np.isclose(pose[1, 3], max(yA, yB), 1e-03)) and (
            pose[1, 3] >= min(yA, yB) or np.isclose(pose[1, 3], min(yA, yB), 1e-03)
        )
        assert (pose[2, 3] <= max(zA, zB) or np.isclose(pose[2, 3], max(zA, zB), 1e-03)) and (
            pose[2, 3] >= min(zA, zB) or np.isclose(pose[2, 3], max(zA, zB), 1e-03)
        )
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-03)

    # Test left
    inter_poses = test_pose_trajectory(A, B, 3.0, arc_direction="left")
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-03)
    for pose in inter_poses:
        assert np.isclose(pose[0, 3], xB, 1e-03)
        assert (pose[1, 3] <= max(yA, yB) or np.isclose(pose[1, 3], max(yA, yB), 1e-03)) and (
            pose[1, 3] >= min(yA, yB) or np.isclose(pose[1, 3], min(yA, yB), 1e-03)
        )
        assert (pose[2, 3] <= max(zA, zB) or np.isclose(pose[2, 3], max(zA, zB), 1e-03)) and (
            pose[2, 3] >= min(zA, zB) or np.isclose(pose[2, 3], max(zA, zB), 1e-03)
        )
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-03)
