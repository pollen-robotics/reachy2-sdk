import logging
import math
import time
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from reachy2_sdk import ReachySDK
from reachy2_sdk.utils.utils import get_pose_matrix

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


def write_A(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting A")

    size = SIZE * scale
    half_size = size / 2

    # TODO : do for non-horizontal orientation

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + size, y - half_size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size / 2, y - half_size / 2, z + 0.02), interpolation_space="cartesian_space", duration=1
    )
    # Start writing
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size / 2, y - half_size / 2, z), interpolation_space="cartesian_space", duration=1
    )
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size / 2, y - 3 * half_size / 2, z), interpolation_space="cartesian_space", duration=1
    )

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size / 2, y - 3 * half_size / 2, z + 0.02),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )
    print("A finished")


def write_B(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting B")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size, y, z),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        duration=1,
        arc_direction="right",
        secondary_radius=half_size,
    )
    reachy.r_arm.goto(
        build_pose_matrix(x, y, z),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        duration=1,
        arc_direction="right",
        secondary_radius=size,
    )

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x, y, z + 0.02),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )
    print("B finished")


def write_C(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting C")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + size, y - half_size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(
        build_pose_matrix(x, y - half_size, z),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="left",
        duration=1,
    )
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait)

    print("C finished")


def write_D(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting D")

    size = SIZE * scale

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(
        build_pose_matrix(x, y, z),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="right",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x, y, z + 0.02), duration=1, wait=wait)

    print("D finished")


def write_E(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting E")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), interpolation_space="cartesian_space", duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y, z + 0.02), interpolation_space="cartesian_space", duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size, y, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait
    )

    print("E finished")


def write_F(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting F")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), interpolation_space="cartesian_space", duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y, z + 0.02), interpolation_space="cartesian_space", duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size, y, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait
    )

    print("F finished")


def write_G(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting G")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + size, y - half_size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(
        build_pose_matrix(x, y - half_size, z),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="left",
        duration=1,
    )
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y - size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y - half_size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size, y - half_size, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait
    )

    print("G finished")


def write_H(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting H")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z + 0.02), interpolation_space="cartesian_space", duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y, z + 0.02), interpolation_space="cartesian_space", duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size, y, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait
    )

    print("H finished")


def write_I(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting I")

    size = SIZE * scale
    half_size = size / 2
    quarter_size = size / 4

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y - half_size, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - half_size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - half_size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x + size, y - quarter_size, z + 0.02), interpolation_space="cartesian_space", duration=1
    )
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - quarter_size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + size, y - 3 * quarter_size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x, y - quarter_size, z + 0.02), interpolation_space="cartesian_space", duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x, y - quarter_size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - 3 * quarter_size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x, y - 3 * quarter_size, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait
    )

    print("I finished")


def write_J(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting J")

    size = SIZE * scale
    half_size = size / 2
    quarter_size = size / 4

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y - half_size, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - half_size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y - half_size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size, y, z),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="back",
        secondary_radius=half_size,
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x + size, y - quarter_size, z + 0.02), interpolation_space="cartesian_space", duration=1
    )
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - quarter_size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + size, y - 3 * quarter_size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x + size, y - 3 * quarter_size, z + 0.02),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("J finished")


def write_K(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting K")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z + 0.02), interpolation_space="cartesian_space", duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z + 0.02), interpolation_space="cartesian_space", duration=1)

    print("K finished")


def write_L(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting L")

    size = SIZE * scale

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait)

    print("L finished")


def write_M(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting M")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), interpolation_space="cartesian_space", duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y - half_size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x + size, y - size, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait
    )

    print("M finished")


def write_N(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting N")

    size = SIZE * scale

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), interpolation_space="cartesian_space", duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z + 0.02), interpolation_space="cartesian_space", duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait)

    print("N finished")


def write_O(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting O")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y - half_size, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - half_size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(
        build_pose_matrix(x, y - half_size, z),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="left",
        duration=1,
    )
    reachy.r_arm.goto(
        build_pose_matrix(x + size, y - half_size, z),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="right",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x + size, y - half_size, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait
    )

    print("O finished")


def write_P(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting P")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), interpolation_space="cartesian_space", duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size, y, z),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="right",
        secondary_radius=half_size,
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size, y, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait
    )

    print("P finished")


def write_Q(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting Q")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y - half_size, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - half_size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(
        build_pose_matrix(x, y - half_size, z),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="left",
        duration=1,
    )
    reachy.r_arm.goto(
        build_pose_matrix(x + size, y - half_size, z),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="right",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size, y - half_size, z + 0.02),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size, y - half_size, z),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        build_pose_matrix(x - half_size, y - size, z),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x - half_size, y - size, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait
    )

    print("Q finished")


def write_R(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting R")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), interpolation_space="cartesian_space", duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size, y, z),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="right",
        secondary_radius=half_size,
        duration=1,
    )
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait)

    print("R finished")


def write_S(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting S")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y - half_size, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - half_size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size, y - half_size, z),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="left",
        secondary_radius=half_size,
        duration=1,
    )
    reachy.r_arm.goto(
        build_pose_matrix(x, y - half_size, z),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="right",
        secondary_radius=half_size,
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x, y - half_size, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait
    )

    print("S finished")


def write_T(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting T")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y - half_size, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - half_size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - half_size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), interpolation_space="cartesian_space", duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x + size, y - size, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait
    )

    print("T finished")


def write_U(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting U")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(
        build_pose_matrix(x + half_size, y - size, z),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="back",
        duration=1,
    )
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x + size, y - size, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait
    )

    print("U finished")


def write_V(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting V")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - half_size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - half_size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x, y - half_size, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait
    )

    print("V finished")


def write_W(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting W")

    size = SIZE * scale
    half_size = size / 2
    quarter_size = size / 4

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - quarter_size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y - half_size, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y - half_size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - quarter_size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y - half_size, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y - half_size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - 3 * quarter_size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - 3 * quarter_size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(
        build_pose_matrix(x, y - 3 * quarter_size, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait
    )

    print("W finished")


def write_X(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting X")

    size = SIZE * scale

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x, y, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait)

    print("X finished")


def write_Y(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting Y")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + half_size, y - half_size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x, y, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait)
    print("Y finished")


def write_Z(
    reachy: ReachySDK,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting Z")

    size = SIZE * scale

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z + 0.02), duration=1)
    # Start writing
    reachy.r_arm.goto(build_pose_matrix(x + size, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x + size, y - size, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y, z), interpolation_space="cartesian_space", duration=1)
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z), interpolation_space="cartesian_space", duration=1)

    # Pen up
    reachy.r_arm.goto(build_pose_matrix(x, y - size, z + 0.02), interpolation_space="cartesian_space", duration=1, wait=wait)

    print("Z finished")


def write_letter(
    reachy: ReachySDK, letter: str, x: float, y: float, z: float, roll: float, pitch: float, yaw: float, scale: float = 1
) -> None:
    match letter:
        case "a":
            write_A(reachy, x, y, z, roll, pitch, yaw, scale)
        case "b":
            write_B(reachy, x, y, z, roll, pitch, yaw, scale)
        case "c":
            write_C(reachy, x, y, z, roll, pitch, yaw, scale)
        case "d":
            write_D(reachy, x, y, z, roll, pitch, yaw, scale)
        case "e":
            write_E(reachy, x, y, z, roll, pitch, yaw, scale)
        case "f":
            write_F(reachy, x, y, z, roll, pitch, yaw, scale)
        case "g":
            write_G(reachy, x, y, z, roll, pitch, yaw, scale)
        case "h":
            write_H(reachy, x, y, z, roll, pitch, yaw, scale)
        case "i":
            write_I(reachy, x, y, z, roll, pitch, yaw, scale)
        case "j":
            write_J(reachy, x, y, z, roll, pitch, yaw, scale)
        case "k":
            write_K(reachy, x, y, z, roll, pitch, yaw, scale)
        case "l":
            write_L(reachy, x, y, z, roll, pitch, yaw, scale)
        case "m":
            write_M(reachy, x, y, z, roll, pitch, yaw, scale)
        case "n":
            write_N(reachy, x, y, z, roll, pitch, yaw, scale)
        case "o":
            write_O(reachy, x, y, z, roll, pitch, yaw, scale)
        case "p":
            write_P(reachy, x, y, z, roll, pitch, yaw, scale)
        case "q":
            write_Q(reachy, x, y, z, roll, pitch, yaw, scale)
        case "r":
            write_R(reachy, x, y, z, roll, pitch, yaw, scale)
        case "s":
            write_S(reachy, x, y, z, roll, pitch, yaw, scale)
        case "t":
            write_T(reachy, x, y, z, roll, pitch, yaw, scale)
        case "u":
            write_U(reachy, x, y, z, roll, pitch, yaw, scale)
        case "s":
            write_V(reachy, x, y, z, roll, pitch, yaw, scale)
        case "w":
            write_W(reachy, x, y, z, roll, pitch, yaw, scale)
        case "x":
            write_X(reachy, x, y, z, roll, pitch, yaw, scale)
        case "y":
            write_Y(reachy, x, y, z, roll, pitch, yaw, scale)
        case "z":
            write_Z(reachy, x, y, z, roll, pitch, yaw, scale)


if __name__ == "__main__":
    print("Reachy SDK example: write name")

    logging.basicConfig(level=logging.INFO)
    reachy = ReachySDK(host="localhost")

    if not reachy.is_connected:
        exit("Reachy is not connected.")

    print("Turning on Reachy")
    reachy.turn_on()
    reachy.goto_posture()

    time.sleep(0.2)

    print("Set to Elbow 90 pose ...")
    r_arm_120 = reachy.r_arm.goto([35, -15, -15, -120, 0, 0, 0])
    while not reachy.is_goto_finished(r_arm_120):
        time.sleep(0.1)

    letters_space = SIZE + SIZE / 2

    starting_y = -0.35
    x = 0.45
    z = 0.0
    scale = 2

    # TEST
    starting_y = 0.1
    x = 0.3
    z = -0.2
    roll = 0
    pitch = -90
    scale = 2

    word = input("Enter word to write: ")

    y = starting_y
    for char in word:
        if char.isalpha():
            char = char.lower()
            yaw = ((y + 0.6) * 80) / 0.7 - 20
            write_letter(reachy, char, x, y, z, roll, pitch, yaw, scale)
            y -= SIZE * scale * 1.5
        if char == " ":
            y -= SIZE * scale * 0.5

    print("Set back to Elbow 90 pose ...")
    head_move = reachy.head.goto_posture("default")
    r_arm_120 = reachy.r_arm.goto([35, -15, -15, -120, 0, 0, 0])
    while not reachy.is_goto_finished(r_arm_120):
        time.sleep(0.1)
