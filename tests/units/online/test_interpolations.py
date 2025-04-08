import time
from threading import Thread
from typing import List

import numpy as np
import numpy.typing as npt
import pytest

from reachy2_sdk.reachy_sdk import ReachySDK

from .test_basic_movements import build_pose_matrix


@pytest.mark.online
def test_goto_cartesian_interpolation_linear(reachy_sdk_zeroed: ReachySDK) -> None:
    xA = 0.3
    yA = -0.2
    zA = -0.3
    xB = xA
    yB = -0.4
    zB = zA
    A = build_pose_matrix(xA, yA, zA)
    B = build_pose_matrix(xB, yB, zB)
    reachy_sdk_zeroed.r_arm.goto(A, wait=True)
    B_gotoid = reachy_sdk_zeroed.r_arm.goto(B, interpolation_space="cartesian_space", interpolation_mode="linear")
    inter_poses = []
    while not reachy_sdk_zeroed.is_goto_finished(B_gotoid):
        inter_poses.append(reachy_sdk_zeroed.r_arm.forward_kinematics())
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-02)
    for pose in inter_poses:
        assert np.isclose(pose[0, 3], xB, 1e-03)
        assert (pose[1, 3] <= max(yA, yB) or np.isclose(pose[1, 3], max(yA, yB), 1e-03)) and (
            pose[1, 3] >= min(yA, yB) or np.isclose(pose[1, 3], min(yA, yB), 1e-03)
        )
        assert np.isclose(pose[2, 3], zB, 1e-03)
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-02)

    xA = 0.3
    yA = -0.3
    zA = -0.3
    xB = 0.4
    yB = yA
    zB = -0.1
    A = build_pose_matrix(xA, yA, zA)
    B = build_pose_matrix(xB, yB, zB)
    reachy_sdk_zeroed.r_arm.goto(A, wait=True)
    B_gotoid = reachy_sdk_zeroed.r_arm.goto(B, interpolation_space="cartesian_space", interpolation_mode="linear")
    inter_poses = []
    while not reachy_sdk_zeroed.is_goto_finished(B_gotoid):
        inter_poses.append(reachy_sdk_zeroed.r_arm.forward_kinematics())
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-02)
    for pose in inter_poses:
        assert (pose[0, 3] <= max(xA, xB) or np.isclose(pose[0, 3], max(xA, xB), 1e-03)) and (
            pose[0, 3] >= min(xA, xB) or np.isclose(pose[0, 3], min(xA, xB), 1e-03)
        )
        assert np.isclose(pose[1, 3], yB, 1e-03)
        assert (pose[2, 3] <= max(zA, zB) or np.isclose(pose[2, 3], max(zA, zB), 1e-03)) and (
            pose[2, 3] >= min(zA, zB) or np.isclose(pose[2, 3], max(zA, zB), 1e-03)
        )
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-02)


@pytest.mark.online
def test_goto_cartesian_interpolation_elliptical(reachy_sdk_zeroed: ReachySDK) -> None:
    xA = 0.3
    yA = -0.2
    zA = -0.3
    xB = xA
    yB = -0.4
    zB = zA
    A = build_pose_matrix(xA, yA, zA)
    B = build_pose_matrix(xB, yB, zB)

    # Test below
    reachy_sdk_zeroed.r_arm.goto(A, wait=True)
    B_gotoid = reachy_sdk_zeroed.r_arm.goto(
        B, interpolation_space="cartesian_space", interpolation_mode="elliptical", arc_direction="below"
    )
    inter_poses = []
    while not reachy_sdk_zeroed.is_goto_finished(B_gotoid):
        inter_poses.append(reachy_sdk_zeroed.r_arm.forward_kinematics())
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-02)
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
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-02)
        if pose[2, 3] < z_limit + 0.05:
            went_down = True
    assert went_down

    # Test above
    reachy_sdk_zeroed.r_arm.goto(A, wait=True)
    B_gotoid = reachy_sdk_zeroed.r_arm.goto(
        B, interpolation_space="cartesian_space", interpolation_mode="elliptical", arc_direction="above"
    )
    inter_poses = []
    while not reachy_sdk_zeroed.is_goto_finished(B_gotoid):
        inter_poses.append(reachy_sdk_zeroed.r_arm.forward_kinematics())
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-02)
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
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-02)
        if pose[2, 3] > z_limit - 0.05:
            went_up = True
    assert went_up

    # Test front
    reachy_sdk_zeroed.r_arm.goto(A, wait=True)
    B_gotoid = reachy_sdk_zeroed.r_arm.goto(
        B, interpolation_space="cartesian_space", interpolation_mode="elliptical", arc_direction="front"
    )
    inter_poses = []
    while not reachy_sdk_zeroed.is_goto_finished(B_gotoid):
        inter_poses.append(reachy_sdk_zeroed.r_arm.forward_kinematics())
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-02)
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
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-02)
    assert went_front

    # Test back
    reachy_sdk_zeroed.r_arm.goto(A, wait=True)
    B_gotoid = reachy_sdk_zeroed.r_arm.goto(
        B, interpolation_space="cartesian_space", interpolation_mode="elliptical", arc_direction="back"
    )
    inter_poses = []
    while not reachy_sdk_zeroed.is_goto_finished(B_gotoid):
        inter_poses.append(reachy_sdk_zeroed.r_arm.forward_kinematics())
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-02)
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
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-02)
    assert went_back

    # Test right
    reachy_sdk_zeroed.r_arm.goto(A, wait=True)
    B_gotoid = reachy_sdk_zeroed.r_arm.goto(
        B, interpolation_space="cartesian_space", interpolation_mode="elliptical", arc_direction="right"
    )
    inter_poses = []
    while not reachy_sdk_zeroed.is_goto_finished(B_gotoid):
        inter_poses.append(reachy_sdk_zeroed.r_arm.forward_kinematics())
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-02)
    for pose in inter_poses:
        assert np.isclose(pose[0, 3], xB, 1e-03)
        assert (pose[1, 3] <= max(yA, yB) or np.isclose(pose[1, 3], max(yA, yB), 1e-03)) and (
            pose[1, 3] >= min(yA, yB) or np.isclose(pose[1, 3], min(yA, yB), 1e-03)
        )
        assert (pose[2, 3] <= max(zA, zB) or np.isclose(pose[2, 3], max(zA, zB), 1e-03)) and (
            pose[2, 3] >= min(zA, zB) or np.isclose(pose[2, 3], max(zA, zB), 1e-03)
        )
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-02)

    # Test left
    reachy_sdk_zeroed.r_arm.goto(A, wait=True)
    B_gotoid = reachy_sdk_zeroed.r_arm.goto(
        B, interpolation_space="cartesian_space", interpolation_mode="elliptical", arc_direction="left"
    )
    inter_poses = []
    while not reachy_sdk_zeroed.is_goto_finished(B_gotoid):
        inter_poses.append(reachy_sdk_zeroed.r_arm.forward_kinematics())
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-02)
    for pose in inter_poses:
        assert np.isclose(pose[0, 3], xB, 1e-03)
        assert (pose[1, 3] <= max(yA, yB) or np.isclose(pose[1, 3], max(yA, yB), 1e-03)) and (
            pose[1, 3] >= min(yA, yB) or np.isclose(pose[1, 3], min(yA, yB), 1e-03)
        )
        assert (pose[2, 3] <= max(zA, zB) or np.isclose(pose[2, 3], max(zA, zB), 1e-03)) and (
            pose[2, 3] >= min(zA, zB) or np.isclose(pose[2, 3], max(zA, zB), 1e-03)
        )
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-02)


@pytest.mark.online
def test_goto_cartesian_interpolation_elliptical_radius(reachy_sdk_zeroed: ReachySDK) -> None:
    xA = 0.3
    yA = -0.2
    zA = -0.2
    xB = xA
    yB = -0.4
    zB = zA
    A = build_pose_matrix(xA, yA, zA)
    B = build_pose_matrix(xB, yB, zB)

    # Test below
    reachy_sdk_zeroed.r_arm.goto(A, wait=True)
    secondary_radius = 0.2
    B_gotoid = reachy_sdk_zeroed.r_arm.goto(
        B,
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="below",
        secondary_radius=secondary_radius,
    )
    inter_poses = []
    while not reachy_sdk_zeroed.is_goto_finished(B_gotoid):
        inter_poses.append(reachy_sdk_zeroed.r_arm.forward_kinematics())
    time.sleep(0.1)
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-02)
    min_z = zA
    z_limit = zB - secondary_radius
    for pose in inter_poses:
        assert np.isclose(pose[0, 3], xB, 1e-03)
        assert (pose[1, 3] <= max(yA, yB) or np.isclose(pose[1, 3], max(yA, yB), 1e-03)) and (
            pose[1, 3] >= min(yA, yB) or np.isclose(pose[1, 3], min(yA, yB), 1e-03)
        )
        assert (pose[2, 3] <= zB or np.isclose(pose[2, 3], zB, 1e-03)) and (
            pose[2, 3] >= z_limit or np.isclose(pose[2, 3], z_limit, 1e-03)
        )
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-02)
        if pose[2, 3] < min_z:
            min_z = pose[2, 3]
    assert np.isclose(min_z, z_limit, 1e-03)

    # Test above
    reachy_sdk_zeroed.r_arm.goto(A, wait=True)
    secondary_radius = 0.05
    B_gotoid = reachy_sdk_zeroed.r_arm.goto(
        B,
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="above",
        secondary_radius=secondary_radius,
    )
    inter_poses = []
    while not reachy_sdk_zeroed.is_goto_finished(B_gotoid):
        inter_poses.append(reachy_sdk_zeroed.r_arm.forward_kinematics())
    time.sleep(0.1)
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-02)
    max_z = zA
    z_limit = zB + secondary_radius
    for pose in inter_poses:
        assert np.isclose(pose[0, 3], xB, 1e-03)
        assert (pose[1, 3] <= max(yA, yB) or np.isclose(pose[1, 3], max(yA, yB), 1e-03)) and (
            pose[1, 3] >= min(yA, yB) or np.isclose(pose[1, 3], min(yA, yB), 1e-03)
        )
        assert (pose[2, 3] >= zB or np.isclose(pose[2, 3], zB, 1e-03)) and (
            pose[2, 3] <= z_limit or np.isclose(pose[2, 3], z_limit, 1e-03)
        )
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-02)
        if pose[2, 3] > max_z:
            max_z = pose[2, 3]
    assert np.isclose(max_z, z_limit, 1e-03)

    xA = 0.4
    yA = -0.2
    zA = -0.3
    xB = xA
    yB = -0.4
    zB = zA
    A = build_pose_matrix(xA, yA, zA)
    B = build_pose_matrix(xB, yB, zB)

    # Test front
    reachy_sdk_zeroed.r_arm.goto(A, wait=True)
    B_gotoid = reachy_sdk_zeroed.r_arm.goto(
        B, interpolation_space="cartesian_space", interpolation_mode="elliptical", arc_direction="front"
    )
    inter_poses = []
    while not reachy_sdk_zeroed.is_goto_finished(B_gotoid):
        inter_poses.append(reachy_sdk_zeroed.r_arm.forward_kinematics())
    time.sleep(0.1)
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-02)
    max_x = xA
    x_limit = xB + abs(yA - yB) / 2
    for pose in inter_poses:
        assert (pose[0, 3] >= xB or np.isclose(pose[0, 3], xB, 1e-03)) and (
            pose[0, 3] <= x_limit or np.isclose(pose[0, 3], x_limit, 1e-03)
        )
        if pose[0, 3] > max_x:
            max_x = pose[0, 3]
        assert (pose[1, 3] <= max(yA, yB) or np.isclose(pose[1, 3], max(yA, yB), 1e-03)) and (
            pose[1, 3] >= min(yA, yB) or np.isclose(pose[1, 3], min(yA, yB), 1e-03)
        )
        assert np.isclose(pose[2, 3], zB, 1e-03)
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-02)
    assert np.isclose(max_x, x_limit, 1e-03)

    # Test back
    reachy_sdk_zeroed.r_arm.goto(A, wait=True)
    secondary_radius = 0.2
    B_gotoid = reachy_sdk_zeroed.r_arm.goto(
        B,
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="back",
        secondary_radius=secondary_radius,
    )
    inter_poses = []
    while not reachy_sdk_zeroed.is_goto_finished(B_gotoid):
        inter_poses.append(reachy_sdk_zeroed.r_arm.forward_kinematics())
    time.sleep(0.1)
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-02)
    min_x = xA
    x_limit = xB - secondary_radius
    for pose in inter_poses:
        assert (pose[0, 3] <= xB or np.isclose(pose[0, 3], xB, 1e-03)) and (
            pose[0, 3] >= x_limit or np.isclose(pose[0, 3], x_limit, 1e-03)
        )
        if pose[0, 3] < min_x:
            min_x = pose[0, 3]
        assert (pose[1, 3] <= max(yA, yB) or np.isclose(pose[1, 3], max(yA, yB), 1e-03)) and (
            pose[1, 3] >= min(yA, yB) or np.isclose(pose[1, 3], min(yA, yB), 1e-03)
        )
        assert np.isclose(pose[2, 3], zB, 1e-03)
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-02)
    assert np.isclose(min_x, x_limit, 1e-03)

    xA = 0.2
    yA = -0.3
    zA = -0.3
    xB = 0.4
    yB = yA
    zB = zA
    A = build_pose_matrix(xA, yA, zA)
    B = build_pose_matrix(xB, yB, zB)

    # Test right
    reachy_sdk_zeroed.r_arm.goto(A, wait=True)
    secondary_radius = 0.1
    B_gotoid = reachy_sdk_zeroed.r_arm.goto(
        B,
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="right",
        secondary_radius=secondary_radius,
    )
    inter_poses = []
    while not reachy_sdk_zeroed.is_goto_finished(B_gotoid):
        inter_poses.append(reachy_sdk_zeroed.r_arm.forward_kinematics())
    time.sleep(0.1)
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-02)
    min_y = yA
    y_limit = yB - secondary_radius
    for pose in inter_poses:
        assert np.isclose(pose[2, 3], zB, 1e-03)
        assert (pose[1, 3] >= y_limit or np.isclose(pose[1, 3], y_limit, 1e-03)) and (
            pose[1, 3] <= yB or np.isclose(pose[1, 3], yB, 1e-03)
        )
        assert (pose[0, 3] <= max(xA, xB) or np.isclose(pose[0, 3], max(xA, xB), 1e-03)) and (
            pose[0, 3] >= min(xA, xB) or np.isclose(pose[0, 3], max(xA, xB), 1e-03)
        )
        if pose[1, 3] < min_y:
            min_y = pose[1, 3]
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-02)
    assert np.isclose(min_y, y_limit, 1e-03)

    # Test left
    reachy_sdk_zeroed.r_arm.goto(A, wait=True)
    secondary_radius = 0.25
    B_gotoid = reachy_sdk_zeroed.r_arm.goto(
        B,
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="left",
        secondary_radius=secondary_radius,
    )
    inter_poses = []
    while not reachy_sdk_zeroed.is_goto_finished(B_gotoid):
        inter_poses.append(reachy_sdk_zeroed.r_arm.forward_kinematics())
    time.sleep(0.1)
    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    assert np.allclose(B_forward, B, atol=1e-02)
    max_y = yA
    y_limit = yB + secondary_radius
    for pose in inter_poses:
        assert np.isclose(pose[2, 3], zB, 1e-03)
        assert (pose[1, 3] >= yB or np.isclose(pose[1, 3], yB, 1e-03)) and (
            pose[1, 3] <= y_limit or np.isclose(pose[1, 3], y_limit, 1e-03)
        )
        assert (pose[0, 3] <= max(xA, xB) or np.isclose(pose[0, 3], max(xA, xB), 1e-03)) and (
            pose[0, 3] >= min(xA, xB) or np.isclose(pose[0, 3], max(xA, xB), 1e-03)
        )
        if pose[1, 3] > max_y:
            max_y = pose[1, 3]
        assert np.allclose(pose[:3, :3], B_forward[:3, :3], atol=1e-02)
    assert np.isclose(max_y, y_limit, 1e-03)
