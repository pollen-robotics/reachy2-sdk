import logging
import math
import time
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from reachy2_sdk import ReachySDK

# For scale 1
SIZE = 0.01


def circlePoints(r: float, n: int = 100, phase: float = 0) -> List[Tuple[float, float]]:
    return [(math.cos(2 * math.pi / n * x + phase) * r, math.sin(2 * math.pi / n * x + phase) * r) for x in range(0, n + 1)]


def ellipsePoints(
    r1: float, r2: float, n: int = 100, clockwise: bool = True, phase: float = math.pi / 2
) -> List[Tuple[float, float]]:
    if clockwise:
        return [
            (math.cos(2 * math.pi / n * x + phase) * r1, math.sin(2 * math.pi / n * x + phase) * r2) for x in range(n, -1, -1)
        ]
    else:
        return [
            (math.cos(2 * math.pi / n * x + phase) * r1, math.sin(2 * math.pi / n * x + phase) * r2) for x in range(0, n + 1)
        ]


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


def send_arm_position(reachy: ReachySDK, ik_sol: List[float]) -> None:
    reachy.r_arm.shoulder.pitch.goal_position = ik_sol[0]
    reachy.r_arm.shoulder.roll.goal_position = ik_sol[1]
    reachy.r_arm.elbow.yaw.goal_position = ik_sol[2]
    reachy.r_arm.elbow.pitch.goal_position = ik_sol[3]
    reachy.r_arm.wrist.roll.goal_position = ik_sol[4]
    reachy.r_arm.wrist.pitch.goal_position = ik_sol[5]
    reachy.r_arm.wrist.yaw.goal_position = ik_sol[6]
    reachy.send_goal_positions()


def write_A(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting A")

    size = SIZE * scale
    half_size = size / 2

    y_range = np.linspace(y, y - half_size, num=20)
    y2_range = np.linspace(y - half_size, y - size, num=20)
    z_range = np.linspace(z, z + size, num=20)
    line_1 = [(ya, za) for ya, za in zip(y_range, z_range)]
    line_2 = [(ya, za) for ya, za in zip(y2_range, reversed(z_range))]

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z), duration=0.7)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for ya, za in line_1:
        target_pose = build_pose_matrix(x, ya, za)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    for ya, za in line_2:
        target_pose = build_pose_matrix(x, ya, za)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    fk = reachy.r_arm.forward_kinematics()
    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, fk[1, 3], fk[2, 3]), 0.5)
    reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y_range[10], z_range[10]), 0.5)
    reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y2_range[10], z_range[10]), 0.5)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y2_range[10], z_range[10]), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("A finished")


def write_B(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting B")

    size = SIZE * scale

    z_range = np.linspace(z + size, z, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zp in z_range:
        target_pose = build_pose_matrix(x, y, zp)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), 0.5)
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), 0.5)
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)

    nb_points = 30
    points = ellipsePoints(size, size / 4, nb_points)
    for yp, zp in points[: nb_points // 2 + 1]:
        target_pose = build_pose_matrix(x, y - yp, z + zp + 3 * size / 4)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    for yp, zp in points[: nb_points // 2 + 1]:
        target_pose = build_pose_matrix(x, y - yp, z + zp + size / 4)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("B finished")


def write_C(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting C")

    half_size = (SIZE / 2) * scale

    nb_points = 30
    points = circlePoints(half_size, nb_points)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x - 0.02, y - points[nb_points // 6][0] - half_size, z + points[nb_points // 6][1] + half_size), starting_duration
    )
    first_pos = reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x, y - points[nb_points // 6][0] - half_size, z + points[nb_points // 6][1] + half_size), 0.5
    )
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for yc, zc in points[nb_points // 6 : -nb_points // 6]:
        target_pose = build_pose_matrix(x, y - yc - half_size, z + zc + half_size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x - 0.02, y - points[-nb_points // 6][0] - half_size, z + points[-nb_points // 6][1] + half_size),
        duration=0.5,
    )
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("C finished")


def write_D(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting D")

    size = SIZE * scale
    half_size = size / 2

    z_range = np.linspace(z + size, z, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zp in z_range:
        target_pose = build_pose_matrix(x, y, zp)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), 0.5)
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), 0.5)
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)

    nb_points = 30
    points = ellipsePoints(size, size / 2, nb_points)
    for yp, zp in points[: nb_points // 2 + 1]:
        target_pose = build_pose_matrix(x, y - yp, z + zp + half_size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("D finished")


def write_E(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting E")

    size = SIZE * scale
    half_size = size / 2

    z_range = np.linspace(z, z + size, num=20)
    y_range = np.linspace(y, y - size, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for ze in reversed(z_range):
        target_pose = build_pose_matrix(x, y, ze)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    for ye in y_range:
        target_pose = build_pose_matrix(x, ye, z)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), 0.5)
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), 0.5)
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)
    for ye in y_range:
        target_pose = build_pose_matrix(x, ye, z + size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + half_size), 0.5)
    inter_pos_2 = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + half_size), 0.5)
    while not reachy.is_move_finished(inter_pos_2):
        time.sleep(0.1)
    for ye in y_range:
        target_pose = build_pose_matrix(x, ye, z + half_size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z + half_size), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("E finished")


def write_F(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting F")

    size = SIZE * scale
    half_size = size / 2

    z_range = np.linspace(z, z + size, num=20)
    y_range = np.linspace(y, y - size, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for ze in reversed(z_range):
        target_pose = build_pose_matrix(x, y, ze)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), 0.5)
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), 0.5)
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)
    for ye in y_range:
        target_pose = build_pose_matrix(x, ye, z + size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + half_size), 0.5)
    inter_pos_2 = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + half_size), 0.5)
    while not reachy.is_move_finished(inter_pos_2):
        time.sleep(0.1)
    for ye in y_range:
        target_pose = build_pose_matrix(x, ye, z + half_size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z + half_size), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("F finished")


def write_G(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting G")

    size = SIZE * scale
    half_size = size / 2

    nb_points = 30
    points = circlePoints(half_size, nb_points)
    y_range = np.linspace(y - size, y - half_size, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x - 0.02, y - points[nb_points // 6][0] - half_size, z + points[nb_points // 6][1] + half_size), starting_duration
    )
    first_pos = reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x, y - points[nb_points // 6][0] - half_size, z + points[nb_points // 6][1] + half_size), duration=0.5
    )
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for yg, zg in points[nb_points // 6 :]:
        target_pose = build_pose_matrix(x, y - yg - half_size, z + zg + half_size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    for yg in y_range:
        target_pose = build_pose_matrix(x, yg, z + half_size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - half_size, z + half_size), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("G finished")


def write_H(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting H")

    size = SIZE * scale
    half_size = size / 2

    z_range = np.linspace(z, z + size, num=20)
    y_range = np.linspace(y, y - size, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zh in reversed(z_range):
        target_pose = build_pose_matrix(x, y, zh)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z + size), 0.5)
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y - size, z + size), 0.5)
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)
    for zh in reversed(z_range):
        target_pose = build_pose_matrix(x, y - size, zh)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + half_size), 0.5)
    inter_pos_2 = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + half_size), 0.5)
    while not reachy.is_move_finished(inter_pos_2):
        time.sleep(0.1)
    for yh in y_range:
        target_pose = build_pose_matrix(x, yh, z + half_size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z + half_size), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("H finished")


def write_I(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting I")

    size = SIZE * scale
    half_size = size / 2

    z_range = np.linspace(z + size, z, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - half_size, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y - half_size, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zi in z_range:
        target_pose = build_pose_matrix(x, y - half_size, zi)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - half_size, z), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("I finished")


def write_J(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting J")

    size = SIZE * scale
    half_size = size / 2
    quarter_size = half_size / 2

    z_range = np.linspace(z + size, z + half_size, num=20)
    y_range = np.linspace(y, y - size, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - half_size, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y - half_size, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zj in z_range:
        target_pose = build_pose_matrix(x, y - half_size, zj)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    nb_points = 30
    points = ellipsePoints(quarter_size, half_size, nb_points, phase=0)
    for yj, zj in points[: nb_points // 2 + 1]:
        target_pose = build_pose_matrix(x, y - yj - quarter_size, z + zj + half_size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), 0.5)
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), 0.5)
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)

    for yj in y_range:
        target_pose = build_pose_matrix(x, yj, z + size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z + size), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)

    print("J finished")


def write_K(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting K")

    size = SIZE * scale
    half_size = size / 2

    z_range = np.linspace(z + size, z, num=20)
    z2_range = np.linspace(z + size, z + half_size, num=20)
    y_range = np.linspace(y, y - size, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zk in z_range:
        target_pose = build_pose_matrix(x, y, zk)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z + size), 0.5)
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y - size, z + size), 0.5)
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)

    for yk, zk in zip(reversed(y_range), z2_range):
        target_pose = build_pose_matrix(x, yk, zk)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    for yk, zk in zip(y_range, z2_range - half_size):
        target_pose = build_pose_matrix(x, yk, zk)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)

    print("K finished")


def write_L(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting L")

    size = SIZE * scale

    z_range = np.linspace(z, z + size, num=20)
    y_range = np.linspace(y, y - size, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for ze in reversed(z_range):
        target_pose = build_pose_matrix(x, y, ze)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    for ye in y_range:
        target_pose = build_pose_matrix(x, ye, z)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("L finished")


def write_M(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting M")

    size = SIZE * scale
    half_size = size / 2

    z_range = np.linspace(z + size, z, num=20)
    z2_range = np.linspace(z + size, z + half_size, num=20)
    y_range = np.linspace(y, y - half_size, num=20)
    y2_range = np.linspace(y - half_size, y - size, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zn in z_range:
        target_pose = build_pose_matrix(x, y, zn)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), 0.5)
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), 0.5)
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)

    for yn, zn in zip(y_range, z2_range):
        target_pose = build_pose_matrix(x, yn, zn)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    for yn, zn in zip(y2_range, reversed(z2_range)):
        target_pose = build_pose_matrix(x, yn, zn)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    for zn in z_range:
        target_pose = build_pose_matrix(x, y - size, zn)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)

    print("M finished")


def write_N(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting N")

    size = SIZE * scale

    z_range = np.linspace(z, z + size, num=20)
    y_range = np.linspace(y, y - size, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zn in reversed(z_range):
        target_pose = build_pose_matrix(x, y, zn)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), 0.5)
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), 0.5)
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)

    for yn, zn in zip(y_range, reversed(z_range)):
        target_pose = build_pose_matrix(x, yn, zn)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    for zn in z_range:
        target_pose = build_pose_matrix(x, y - size, zn)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z + size), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)

    print("N finished")


def write_O(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting O")

    size = SIZE * scale
    half_size = size / 2

    nb_points = 30
    points = circlePoints(half_size, nb_points)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - points[0][0] - half_size, z + points[0][1] + half_size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x, y - points[0][0] - half_size, z + points[0][1] + half_size), duration=0.5
    )
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for yo, zo in points:
        target_pose = build_pose_matrix(x, y - yo - half_size, z + zo + half_size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x - 0.02, y - points[-1][0] - half_size, z + points[-1][1] + half_size), duration=0.5
    )
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)

    print("O finished")


def write_P(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting P")

    size = SIZE * scale
    half_size = size / 2
    quarter_size = half_size / 2

    z_range = np.linspace(z + size, z, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zp in z_range:
        target_pose = build_pose_matrix(x, y, zp)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), 0.5)
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), 0.5)
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)

    nb_points = 30
    points = ellipsePoints(size, quarter_size, nb_points)
    for yp, zp in points[: nb_points // 2 + 1]:
        target_pose = build_pose_matrix(x, y - yp, z + zp + 3 * quarter_size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + half_size), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)

    print("P finished")


def write_Q(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting Q")

    size = SIZE * scale
    half_size = size / 2
    quarter_size = half_size / 2

    y_range = np.linspace(y - 3 * quarter_size, y - size, num=20)
    z_range = np.linspace(z + quarter_size, z - quarter_size, num=20)

    nb_points = 30
    points = circlePoints(half_size, nb_points)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - points[0][0] - half_size, z + points[0][1] + half_size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x, y - points[0][0] - half_size, z + points[0][1] + half_size), duration=0.5
    )
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for yq, zq in points:
        target_pose = build_pose_matrix(x, y - yq - half_size, z + zq + half_size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - 3 * quarter_size, z + quarter_size), 0.5)
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y - 3 * quarter_size, z + quarter_size), 0.5)
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)

    for yq, zq in zip(y_range, z_range):
        target_pose = build_pose_matrix(x, yq, zq)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z - quarter_size), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("Q finished")


def write_R(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting R")

    size = SIZE * scale
    half_size = size / 2
    quarter_size = half_size / 2

    z_range = np.linspace(z + size, z, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zr in z_range:
        target_pose = build_pose_matrix(x, y, zr)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), 0.5)
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), 0.5)
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)

    nb_points = 30
    points = ellipsePoints(size, quarter_size, nb_points)
    for yr, zr in points[: nb_points // 2 + 1]:
        target_pose = build_pose_matrix(x, y - yr, z + zr + 3 * quarter_size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    y_range = np.linspace(y, y - size, num=10)
    z2_range = np.linspace(z + half_size, z, num=10)

    for yr, zr in zip(y_range, z2_range):
        target_pose = build_pose_matrix(x, yr, zr)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("R finished")


def write_S(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting S")

    size = SIZE * scale
    half_size = size / 2
    quarter_size = half_size / 2

    reachy.head.look_at(x, y, z, duration=1)

    nb_points = 36
    points_top = ellipsePoints(half_size, quarter_size, nb_points, clockwise=False, phase=math.pi / 6)
    points_bottom = ellipsePoints(half_size, quarter_size, nb_points)

    reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x - 0.02, y - points_top[0][0] - half_size, z + points_top[0][1] + 3 * quarter_size), starting_duration
    )
    first_pos = reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x, y - points_top[0][0] - half_size, z + points_top[0][1] + 3 * quarter_size), duration=0.5
    )
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for ys, zs in points_top[: 2 * nb_points // 3]:
        target_pose = build_pose_matrix(x, y - ys - half_size, z + zs + 3 * quarter_size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    for ys, zs in points_bottom[: 2 * nb_points // 3]:
        target_pose = build_pose_matrix(x, y - ys - half_size, z + zs + quarter_size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(
        build_pose_matrix(
            x - 0.02,
            y - points_bottom[2 * nb_points // 3][0] - half_size,
            z + points_bottom[2 * nb_points // 3][1] + quarter_size,
        ),
        duration=0.5,
    )
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("S finished")


def write_T(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting T")

    size = SIZE * scale
    half_size = size / 2

    z_range = np.linspace(z + size, z, num=20)
    y_range = np.linspace(y, y - size, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - half_size, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y - half_size, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zt in z_range:
        target_pose = build_pose_matrix(x, y - half_size, zt)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), 0.5)
    inter_pos_2 = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), 0.5)
    while not reachy.is_move_finished(inter_pos_2):
        time.sleep(0.1)
    for yt in y_range:
        target_pose = build_pose_matrix(x, yt, z + size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z + size), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("T finished")


def write_U(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting U")

    size = SIZE * scale
    half_size = size / 2

    z_range = np.linspace(z + size, z + half_size, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zu in z_range:
        target_pose = build_pose_matrix(x, y, zu)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    nb_points = 30
    points = circlePoints(half_size, nb_points, phase=math.pi)
    for yu, zu in points[: nb_points // 2 + 1]:
        target_pose = build_pose_matrix(x, y - yu - half_size, z + zu + half_size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    for zu in reversed(z_range):
        target_pose = build_pose_matrix(x, y - size, zu)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z + size), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)

    print("U finished")


def write_V(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting V")

    size = SIZE * scale
    half_size = size / 2

    y_range = np.linspace(y, y - half_size, num=20)
    y2_range = np.linspace(y - half_size, y - size, num=20)
    z_range = np.linspace(z, z + size, num=20)
    line_1 = [(ya, za) for ya, za in zip(y_range, reversed(z_range))]
    line_2 = [(ya, za) for ya, za in zip(y2_range, z_range)]

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for ya, za in line_1:
        target_pose = build_pose_matrix(x, ya, za)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    for ya, za in line_2:
        target_pose = build_pose_matrix(x, ya, za)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z + size), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)

    print("V finished")


def write_W(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting W")

    size = SIZE * scale
    half_size = size / 2
    quarter_size = half_size / 2

    y_range = np.linspace(y, y - quarter_size, num=10)
    z_range = np.linspace(z, z + size, num=10)
    z2_range = np.linspace(z, z + half_size, num=10)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for yw, zw in zip(y_range, reversed(z_range)):
        target_pose = build_pose_matrix(x, yw, zw)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    for yw, zw in zip(y_range - quarter_size, z2_range):
        target_pose = build_pose_matrix(x, yw, zw)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    for yw, zw in zip(y_range - half_size, reversed(z2_range)):
        target_pose = build_pose_matrix(x, yw, zw)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    for yw, zw in zip(y_range - 3 * quarter_size, z_range):
        target_pose = build_pose_matrix(x, yw, zw)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z + size), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)

    print("W finished")


def write_X(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting X")

    size = SIZE * scale

    y_range = np.linspace(y, y - size, num=10)
    z_range = np.linspace(z + size, z, num=10)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for yx, zx in zip(y_range, z_range):
        target_pose = build_pose_matrix(x, yx, zx)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z + size), 0.5)
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y - size, z + size), 0.5)
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)
    for yx, zx in zip(reversed(y_range), z_range):
        target_pose = build_pose_matrix(x, yx, zx)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)

    print("X finished")


def write_Y(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting Y")

    size = SIZE * scale
    half_size = size / 2

    y_range = np.linspace(y, y - half_size, num=10)
    y2_range = np.linspace(y - size, y, num=20)
    z_range = np.linspace(z + size, z + half_size, num=10)
    z2_range = np.linspace(z + size, z, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for yy, zy in zip(y_range, z_range):
        target_pose = build_pose_matrix(x, yy, zy)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z + size), 0.5)
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y - size, z + size), 0.5)
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)
    for yy, zy in zip(y2_range, z2_range):
        target_pose = build_pose_matrix(x, yy, zy)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("Y finished")


def write_Z(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    print("Starting Z")

    size = SIZE * scale

    z_range = np.linspace(z + size, z, num=20)
    y_range = np.linspace(y, y - size, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size), starting_duration)
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=0.5)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for yn in y_range:
        target_pose = build_pose_matrix(x, yn, z + size)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    for yn, zn in zip(reversed(y_range), z_range):
        target_pose = build_pose_matrix(x, yn, zn)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)
    for yn in y_range:
        target_pose = build_pose_matrix(x, yn, z)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.01)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - size, z), duration=0.5)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)

    print("Z finished")


def write_letter(reachy: ReachySDK, letter: str, x: float, y: float, z: float, scale: float = 1, starting_duration: float = 0.5) -> None:
    match letter:
        case "a":
            write_A(reachy, x, y, z, scale, starting_duration)
        case "b":
            write_B(reachy, x, y, z, scale, starting_duration)
        case "c":
            write_C(reachy, x, y, z, scale, starting_duration)
        case "d":
            write_D(reachy, x, y, z, scale, starting_duration)
        case "e":
            write_E(reachy, x, y, z, scale, starting_duration)
        case "f":
            write_F(reachy, x, y, z, scale, starting_duration)
        case "g":
            write_G(reachy, x, y, z, scale, starting_duration)
        case "h":
            write_H(reachy, x, y, z, scale, starting_duration)
        case "i":
            write_I(reachy, x, y, z, scale, starting_duration)
        case "j":
            write_J(reachy, x, y, z, scale, starting_duration)
        case "k":
            write_K(reachy, x, y, z, scale, starting_duration)
        case "l":
            write_L(reachy, x, y, z, scale, starting_duration)
        case "m":
            write_M(reachy, x, y, z, scale, starting_duration)
        case "n":
            write_N(reachy, x, y, z, scale, starting_duration)
        case "o":
            write_O(reachy, x, y, z, scale, starting_duration)
        case "p":
            write_P(reachy, x, y, z, scale, starting_duration)
        case "q":
            write_Q(reachy, x, y, z, scale, starting_duration)
        case "r":
            write_R(reachy, x, y, z, scale, starting_duration)
        case "s":
            write_S(reachy, x, y, z, scale, starting_duration)
        case "t":
            write_T(reachy, x, y, z, scale, starting_duration)
        case "u":
            write_U(reachy, x, y, z, scale, starting_duration)
        case "s":
            write_V(reachy, x, y, z, scale, starting_duration)
        case "w":
            write_W(reachy, x, y, z, scale, starting_duration)
        case "x":
            write_X(reachy, x, y, z, scale, starting_duration)
        case "y":
            write_Y(reachy, x, y, z, scale, starting_duration)
        case "z":
            write_Z(reachy, x, y, z, scale, starting_duration)


if __name__ == "__main__":
    print("Reachy SDK example: write name")

    logging.basicConfig(level=logging.INFO)
    reachy = ReachySDK(host="10.0.0.201")

    if not reachy.is_connected:
        exit("Reachy is not connected.")

    print("Turning on Reachy")
    # reachy.turn_on()

    time.sleep(0.2)

    print("Set to Elbow 90 pose ...")
    # r_arm_90 = reachy.r_arm.set_pose("elbow_90")
    r_arm_120 = reachy.r_arm.goto_joints([35, -15, -15, -120, 0, 0, 0])
    while not reachy.is_move_finished(r_arm_120):
        time.sleep(0.1)

    letters_space = SIZE + SIZE / 2

    starting_y = -0.35
    x = 0.45
    z = 0
    scale = 1

    word = input("Enter word to write: ")
    first_letter = True

    y = starting_y
    for char in word:
        if char.isalpha():
            char = char.lower()
            if first_letter:
                write_letter(reachy, char, x, y, z, scale, starting_duration=1.5)
                first_letter=False
            else:
                write_letter(reachy, char, x, y, z, scale)
            y -= SIZE * scale * 1.5
        if char == " ":
            y -= SIZE * scale * 0.5

    print("Set back to Elbow 90 pose ...")
    head_straight = reachy.head.set_pose("default")
    r_arm_120 = reachy.r_arm.goto_joints([35, -15, -15, -120, 0, 0, 0])
    while not reachy.is_move_finished(r_arm_120):
        time.sleep(0.1)
