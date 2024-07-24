import logging
import math
import time
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from reachy2_sdk import ReachySDK


# For scale 1
SIZE = 0.01


def circlePoints(r: float, n: int = 100) -> List[Tuple[float, float]]:
    return [(math.cos(2 * math.pi / n * x) * r, math.sin(2 * math.pi / n * x) * r) for x in range(0, n + 1)]


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


def write_A(reachy: ReachySDK, x: float, y: float, z: float, scale: float = 1) -> None:
    print("Starting A")

    half_size = (SIZE / 2) * scale
    size = SIZE * scale

    y_range = np.linspace(y, y - half_size, num=20)
    y2_range = np.linspace(y - half_size, y - size, num=20)
    z_range = np.linspace(z, z + size, num=20)
    line_1 = [(ya, za) for ya, za in zip(y_range, z_range)]
    line_2 = [(ya, za) for ya, za in zip(y2_range, reversed(z_range))]

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z))
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z), duration=1)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for ya, za in line_1:
        target_pose = build_pose_matrix(x, ya, za)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)
    for ya, za in line_2:
        target_pose = build_pose_matrix(x, ya, za)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)
    fk = reachy.r_arm.forward_kinematics()
    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, fk[1, 3], fk[2, 3]))
    reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y_range[10], z_range[10]))
    reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y2_range[10], z_range[10]))

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y2_range[10], z_range[10]), duration=1)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("A finished")


def write_B(reachy: ReachySDK, x: float, y: float, z: float) -> None:
    print("Starting B")

    half_size = (SIZE / 2) * scale
    size = SIZE * scale

    z_range = np.linspace(z + size, z, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size))
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size), duration=1)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zp in z_range:
        target_pose = build_pose_matrix(x, y, zp)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + size))
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + size))
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)

    nb_points = 30
    points = ellipsePoints(size, size / 4, nb_points)
    for yp, zp in points[: nb_points // 2 + 1]:
        target_pose = build_pose_matrix(x, y - yp, z + zp + 3 * size / 4)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)
    for yp, zp in points[: nb_points // 2 + 1]:
        target_pose = build_pose_matrix(x, y - yp, z + zp + size / 4)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z), duration=1)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("B finished")


def write_C(reachy: ReachySDK, x: float, y: float, z: float) -> None:
    print("Starting C")
    nb_points = 30
    points = circlePoints(0.005, nb_points)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x - 0.02, y - points[nb_points // 6][0] - 0.005, z + points[nb_points // 6][1] + 0.005)
    )
    first_pos = reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x, y - points[nb_points // 6][0] - 0.005, z + points[nb_points // 6][1] + 0.005), duration=1
    )
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for yc, zc in points[nb_points // 6 : -nb_points // 6]:
        target_pose = build_pose_matrix(x, y - yc - 0.005, z + zc + 0.005)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    last_pos = reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x - 0.02, y - points[-nb_points // 6][0] - 0.005, z + points[-nb_points // 6][1] + 0.005), duration=1
    )
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("C finished")


def write_E(reachy: ReachySDK, x: float, y: float, z: float) -> None:
    print("Starting E")

    z_range = np.linspace(z, z + 0.01, num=20)
    y_range = np.linspace(y, y - 0.01, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + 0.01))
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + 0.01), duration=1)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for ze in reversed(z_range):
        target_pose = build_pose_matrix(x, y, ze)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)
    for ye in y_range:
        target_pose = build_pose_matrix(x, ye, z)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + 0.01))
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + 0.01))
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)
    for ye in y_range:
        target_pose = build_pose_matrix(x, ye, z + 0.01)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + 0.005))
    inter_pos_2 = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + 0.005))
    while not reachy.is_move_finished(inter_pos_2):
        time.sleep(0.1)
    for ye in y_range:
        target_pose = build_pose_matrix(x, ye, z + 0.005)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - 0.01, z + 0.005), duration=1)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("E finished")


def write_G(reachy: ReachySDK, x: float, y: float, z: float) -> None:
    print("Starting G")
    nb_points = 30
    points = circlePoints(0.005, nb_points)
    y_range = np.linspace(y - 0.01, y - 0.005, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x - 0.02, y - points[nb_points // 6][0] - 0.005, z + points[nb_points // 6][1] + 0.005)
    )
    first_pos = reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x, y - points[nb_points // 6][0] - 0.005, z + points[nb_points // 6][1] + 0.005), duration=1
    )
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for yg, zg in points[nb_points // 6 :]:
        target_pose = build_pose_matrix(x, y - yg - 0.005, z + zg + 0.005)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)
    for yg in y_range:
        target_pose = build_pose_matrix(x, yg, z + 0.005)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - 0.005, z + 0.005), duration=1)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("G finished")


def write_H(reachy: ReachySDK, x: float, y: float, z: float) -> None:
    print("Starting H")

    z_range = np.linspace(z, z + 0.01, num=20)
    y_range = np.linspace(y, y - 0.01, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + 0.01))
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + 0.01), duration=1)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zh in reversed(z_range):
        target_pose = build_pose_matrix(x, y, zh)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - 0.01, z + 0.01))
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y - 0.01, z + 0.01))
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)
    for zh in reversed(z_range):
        target_pose = build_pose_matrix(x, y - 0.01, zh)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + 0.005))
    inter_pos_2 = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + 0.005))
    while not reachy.is_move_finished(inter_pos_2):
        time.sleep(0.1)
    for yh in y_range:
        target_pose = build_pose_matrix(x, yh, z + 0.005)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - 0.01, z + 0.005), duration=1)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("H finished")


def write_I(reachy: ReachySDK, x: float, y: float, z: float) -> None:
    print("Starting I")

    z_range = np.linspace(z + 0.01, z, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - 0.005, z + 0.01))
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y - 0.005, z + 0.01), duration=1)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zi in z_range:
        target_pose = build_pose_matrix(x, y - 0.005, zi)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - 0.005, z), duration=1)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("I finished")


def write_L(reachy: ReachySDK, x: float, y: float, z: float) -> None:
    print("Starting L")

    z_range = np.linspace(z, z + 0.01, num=20)
    y_range = np.linspace(y, y - 0.01, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + 0.01))
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + 0.01), duration=1)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for ze in reversed(z_range):
        target_pose = build_pose_matrix(x, y, ze)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)
    for ye in y_range:
        target_pose = build_pose_matrix(x, ye, z)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - 0.01, z), duration=1)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("L finished")


def write_N(reachy: ReachySDK, x: float, y: float, z: float) -> None:
    print("Starting N")

    z_range = np.linspace(z, z + 0.01, num=20)
    y_range = np.linspace(y, y - 0.01, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z))
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z), duration=1)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zn in z_range:
        target_pose = build_pose_matrix(x, y, zn)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)
    for yn, zn in zip(y_range, reversed(z_range)):
        target_pose = build_pose_matrix(x, yn, zn)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)
    for zn in z_range:
        target_pose = build_pose_matrix(x, y - 0.01, zn)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - 0.01, z + 0.01), duration=1)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("N finished")


def write_O(reachy: ReachySDK, x: float, y: float, z: float) -> None:
    print("Starting O")
    nb_points = 30
    points = circlePoints(0.005, nb_points)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - points[0][0] - 0.005, z + points[0][1] + 0.005))
    first_pos = reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x, y - points[0][0] - 0.005, z + points[0][1] + 0.005), duration=1
    )
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for yo, zo in points:
        target_pose = build_pose_matrix(x, y - yo - 0.005, z + zo + 0.005)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    last_pos = reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x - 0.02, y - points[-1][0] - 0.005, z + points[-1][1] + 0.005), duration=1
    )
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("O finished")


def write_P(reachy: ReachySDK, x: float, y: float, z: float) -> None:
    print("Starting P")
    z_range = np.linspace(z + 0.01, z, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + 0.01))
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + 0.01), duration=1)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zp in z_range:
        target_pose = build_pose_matrix(x, y, zp)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + 0.01))
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + 0.01))
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)

    nb_points = 30
    points = ellipsePoints(0.01, 0.0025, nb_points)
    for yp, zp in points[: nb_points // 2 + 1]:
        target_pose = build_pose_matrix(x, y - yp, z + zp + 0.0075)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + 0.005), duration=1)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("P finished")


def write_R(reachy: ReachySDK, x: float, y: float, z: float) -> None:
    print("Starting R")
    z_range = np.linspace(z + 0.01, z, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + 0.01))
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + 0.01), duration=1)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zr in z_range:
        target_pose = build_pose_matrix(x, y, zr)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + 0.01))
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + 0.01))
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)

    nb_points = 30
    points = ellipsePoints(0.01, 0.0025, nb_points)
    for yr, zr in points[: nb_points // 2 + 1]:
        target_pose = build_pose_matrix(x, y - yr, z + zr + 0.0075)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    y_range = np.linspace(y, y - 0.01, num=10)
    z2_range = np.linspace(z + 0.005, z, num=10)

    for yr, zr in zip(y_range, z2_range):
        target_pose = build_pose_matrix(x, yr, zr)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - 0.01, z), duration=1)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("R finished")


def write_S(reachy: ReachySDK, x: float, y: float, z: float) -> None:
    print("Starting S")

    reachy.head.look_at(x, y, z, duration=1)

    nb_points = 36
    points_top = ellipsePoints(0.005, 0.0025, nb_points, clockwise=False, phase=math.pi / 6)
    points_bottom = ellipsePoints(0.005, 0.0025, nb_points)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - points_top[0][0] - 0.005, z + points_top[0][1] + 0.0075))
    first_pos = reachy.r_arm.goto_from_matrix(
        build_pose_matrix(x, y - points_top[0][0] - 0.005, z + points_top[0][1] + 0.0075), duration=1
    )
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for ys, zs in points_top[: 2 * nb_points // 3]:
        target_pose = build_pose_matrix(x, y - ys - 0.005, z + zs + 0.0075)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)
    for ys, zs in points_bottom[: 2 * nb_points // 3]:
        target_pose = build_pose_matrix(x, y - ys - 0.005, z + zs + 0.0025)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    last_pos = reachy.r_arm.goto_from_matrix(
        build_pose_matrix(
            x - 0.02, y - points_bottom[2 * nb_points // 3][0] - 0.005, z + points_bottom[2 * nb_points // 3][1] + 0.0025
        ),
        duration=1,
    )
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("S finished")


def write_T(reachy: ReachySDK, x: float, y: float, z: float) -> None:
    print("Starting T")

    z_range = np.linspace(z + 0.01, z, num=20)
    y_range = np.linspace(y, y - 0.01, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - 0.005, z + 0.01))
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y - 0.005, z + 0.01), duration=1)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for zt in z_range:
        target_pose = build_pose_matrix(x, y - 0.005, zt)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + 0.01))
    inter_pos_2 = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + 0.01))
    while not reachy.is_move_finished(inter_pos_2):
        time.sleep(0.1)
    for yt in y_range:
        target_pose = build_pose_matrix(x, yt, z + 0.01)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - 0.01, z + 0.01), duration=1)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("T finished")


def write_Y(reachy: ReachySDK, x: float, y: float, z: float) -> None:
    print("Starting Y")

    y_range = np.linspace(y, y - 0.005, num=10)
    y2_range = np.linspace(y - 0.01, y, num=20)
    z_range = np.linspace(z + 0.01, z + 0.005, num=10)
    z2_range = np.linspace(z + 0.01, z, num=20)

    reachy.head.look_at(x, y, z, duration=1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z + 0.01))
    first_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y, z + 0.01), duration=1)
    while not reachy.is_move_finished(first_pos):
        time.sleep(0.1)

    for yy, zy in zip(y_range, z_range):
        target_pose = build_pose_matrix(x, yy, zy)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y - 0.01, z + 0.01))
    inter_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x, y - 0.01, z + 0.01))
    while not reachy.is_move_finished(inter_pos):
        time.sleep(0.1)
    for yy, zy in zip(y2_range, z2_range):
        target_pose = build_pose_matrix(x, yy, zy)
        ik = reachy.r_arm.inverse_kinematics(target_pose)
        send_arm_position(reachy, ik)
        time.sleep(0.1)

    last_pos = reachy.r_arm.goto_from_matrix(build_pose_matrix(x - 0.02, y, z), duration=1)
    while not reachy.is_move_finished(last_pos):
        time.sleep(0.1)
    print("Y finished")


if __name__ == "__main__":
    print("Reachy SDK example: write name")

    logging.basicConfig(level=logging.INFO)
    reachy = ReachySDK(host="localhost")

    if not reachy.is_connected:
        exit("Reachy is not connected.")

    print("Turning on Reachy")
    reachy.turn_on()

    time.sleep(0.2)

    print("Set to Elbow 90 pose ...")
    r_arm_90 = reachy.r_arm.set_pose("elbow_90")
    while not reachy.is_move_finished(r_arm_90):
        time.sleep(0.1)

    letters_space = SIZE + SIZE / 2

    starting_y = -0.35
    x = 0.45
    z = 0
    scale = 1

    write_R(reachy, x, starting_y, z)
    write_E(reachy, x, starting_y - 0.015, z)
    write_A(reachy, x, starting_y - 0.015 * 2, z)
    write_C(reachy, x, starting_y - 0.015 * 3, z)
    write_H(reachy, x, starting_y - 0.015 * 4, z)
    write_Y(reachy, x, starting_y - 0.015 * 5, z)

    # write_G(reachy, x, starting_y, z)
    # write_A(reachy, x, starting_y-0.015, z)
    # write_E(reachy, x, starting_y-0.015*2, z)
    # write_L(reachy, x, starting_y-0.015*3, z)
    # write_L(reachy, x, starting_y-0.015*4, z)
    # write_E(reachy, x, starting_y-0.015*5, z)

    # write_P(reachy, x, starting_y, z)
    # write_O(reachy, x, starting_y-0.015, z)
    # write_L(reachy, x, starting_y-0.015*2, z)
    # write_L(reachy, x, starting_y-0.015*3, z)
    # write_E(reachy, x, starting_y-0.015*4, z)
    # write_N(reachy, x, starting_y-0.015*5, z)

    # write_R(reachy, x, starting_y, z)
    # write_O(reachy, x, starting_y-0.015, z)
    # write_B(reachy, x, starting_y-0.015*2, z)
    # write_O(reachy, x, starting_y-0.015*3, z)
    # write_T(reachy, x, starting_y-0.015*4, z)
    # write_I(reachy, x, starting_y-0.015*5, z)
    # write_C(reachy, x, starting_y-0.015*6, z)
    # write_S(reachy, x, starting_y-0.015*7, z)

    print("Set back to Elbow 90 pose ...")
    r_arm_90 = reachy.head.set_pose("default")
    r_arm_90 = reachy.r_arm.set_pose("elbow_90")
    while not reachy.is_move_finished(r_arm_90):
        time.sleep(0.1)
