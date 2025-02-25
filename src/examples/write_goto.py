import logging
import time
from typing import Any, List

from reachy2_sdk import ReachySDK
from reachy2_sdk.utils.utils import get_pose_matrix

# For scale 1
SIZE = 0.01


def get_oriented_pose_matrix(
    point: List[float],
    origin: List[float],
    orientation: str,
    pen_up: bool = False,
) -> Any:
    """Go to a specific pose with the arm."""
    if orientation == "horizontal":
        x = origin[0] + point[0]
        y = origin[1] - point[1]
        z = origin[2]
        roll = 0.0
        pitch = -90.0
        yaw = ((y + 0.6) * 80) / 0.7 - 20
        rotation = [roll, pitch, yaw]
        if pen_up:
            return get_pose_matrix([x, y, z + 0.02], rotation)
        return get_pose_matrix([x, y, z], rotation)
    elif orientation == "vertical":
        x = origin[0]
        y = origin[1] - point[1]
        z = origin[2] + point[0]
        roll = ((y + 0.6) * 80) / 0.7 - 20
        pitch = -180.0
        yaw = 0.0
        rotation = [roll, pitch, yaw]
        if pen_up:
            return get_pose_matrix([x - 0.02, y, z], rotation)
        return get_pose_matrix([x, y, z], rotation)
    else:
        raise ValueError("Invalid orientation")


def write_A(
    reachy: ReachySDK,
    origin: List[float],
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
    reachy.r_arm.goto(get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size / 2, half_size / 2], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size / 2, half_size / 2], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size / 2, 3 * half_size / 2], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size / 2, 3 * half_size / 2], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )
    print("A finished")


def write_B(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting B")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        duration=1,
        arc_direction="right",
        secondary_radius=3 * half_size / 2,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        duration=1,
        arc_direction="right",
        secondary_radius=size,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )
    print("B finished")


def write_C(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting C")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="left",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("C finished")


def write_D(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting D")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="right",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation, pen_up=True), duration=1, wait=wait
    )

    print("D finished")


def write_E(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting E")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("E finished")


def write_F(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting F")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("F finished")


def write_G(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting G")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="left",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, half_size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("G finished")


def write_H(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting H")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("H finished")


def write_I(
    reachy: ReachySDK,
    origin: List[float],
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
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation, pen_up=True), duration=1
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, quarter_size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, quarter_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 3 * quarter_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, quarter_size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, quarter_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 3 * quarter_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 3 * quarter_size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("I finished")


def write_J(
    reachy: ReachySDK,
    origin: List[float],
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
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation, pen_up=True), duration=1
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="back",
        secondary_radius=half_size,
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, quarter_size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, quarter_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 3 * quarter_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 3 * quarter_size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("J finished")


def write_K(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting K")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )

    print("K finished")


def write_L(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting L")

    size = SIZE * scale

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("L finished")


def write_M(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting M")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("M finished")


def write_N(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting N")

    size = SIZE * scale

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("N finished")


def write_O(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting O")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation, pen_up=True), duration=1
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="left",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="right",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("O finished")


def write_P(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting P")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="right",
        secondary_radius=size,
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("P finished")


def write_Q(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting Q")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation, pen_up=True), duration=1
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="left",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="right",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, half_size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0 - half_size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0 - half_size, size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("Q finished")


def write_R(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting R")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="right",
        secondary_radius=size,
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("R finished")


def write_S(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting S")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="left",
        secondary_radius=half_size,
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="right",
        secondary_radius=half_size,
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("S finished")


def write_T(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting T")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation, pen_up=True), duration=1
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("T finished")


def write_U(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting U")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        interpolation_mode="elliptical",
        arc_direction="back",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("U finished")


def write_V(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting V")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, half_size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("V finished")


def write_W(
    reachy: ReachySDK,
    origin: List[float],
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
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, quarter_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, half_size], origin=origin, orientation=orientation, pen_up=True), duration=1
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, quarter_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, half_size], origin=origin, orientation=orientation, pen_up=True), duration=1
    )
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 3 * quarter_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 3 * quarter_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 3 * quarter_size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("W finished")


def write_X(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting X")

    size = SIZE * scale

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("X finished")


def write_Y(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting Y")

    size = SIZE * scale
    half_size = size / 2

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([half_size, half_size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )
    print("Y finished")


def write_Z(
    reachy: ReachySDK,
    origin: List[float],
    scale: float = 1,
    orientation: str = "horizontal",
    wait: bool = True,
) -> None:
    print("Starting Z")

    size = SIZE * scale

    reachy.head.look_at(x, y, z, duration=1)

    # Pen up
    reachy.r_arm.goto(get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation, pen_up=True), duration=1)
    # Start writing
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([size, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, 0], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation),
        interpolation_space="cartesian_space",
        duration=1,
    )

    # Pen up
    reachy.r_arm.goto(
        get_oriented_pose_matrix([0, size], origin=origin, orientation=orientation, pen_up=True),
        interpolation_space="cartesian_space",
        duration=1,
        wait=wait,
    )

    print("Z finished")


def write_letter(reachy: ReachySDK, letter: str, origin: List[float], scale: float = 1) -> None:
    match letter:
        case "a":
            write_A(reachy, origin, scale)
        case "b":
            write_B(reachy, origin, scale)
        case "c":
            write_C(reachy, origin, scale)
        case "d":
            write_D(reachy, origin, scale)
        case "e":
            write_E(reachy, origin, scale)
        case "f":
            write_F(reachy, origin, scale)
        case "g":
            write_G(reachy, origin, scale)
        case "h":
            write_H(reachy, origin, scale)
        case "i":
            write_I(reachy, origin, scale)
        case "j":
            write_J(reachy, origin, scale)
        case "k":
            write_K(reachy, origin, scale)
        case "l":
            write_L(reachy, origin, scale)
        case "m":
            write_M(reachy, origin, scale)
        case "n":
            write_N(reachy, origin, scale)
        case "o":
            write_O(reachy, origin, scale)
        case "p":
            write_P(reachy, origin, scale)
        case "q":
            write_Q(reachy, origin, scale)
        case "r":
            write_R(reachy, origin, scale)
        case "s":
            write_S(reachy, origin, scale)
        case "t":
            write_T(reachy, origin, scale)
        case "u":
            write_U(reachy, origin, scale)
        case "v":
            write_V(reachy, origin, scale)
        case "w":
            write_W(reachy, origin, scale)
        case "x":
            write_X(reachy, origin, scale)
        case "y":
            write_Y(reachy, origin, scale)
        case "z":
            write_Z(reachy, origin, scale)


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

    print("Set to custom Elbow 120 pose ...")
    r_arm_120 = reachy.r_arm.goto([35, -15, -15, -120, 0, 0, 0], wait=True)

    letters_space = SIZE + SIZE / 2

    starting_y = -0.35
    x = 0.45
    z = 0.0
    scale = 2

    # TEST
    starting_y = 0.1
    x = 0.3
    z = -0.362
    scale = 2

    word = input("Enter word to write: ")

    y = starting_y
    for char in word:
        if char.isalpha():
            char = char.lower()
            write_letter(reachy, char, [x, y, z], scale)
            y -= SIZE * scale * 1.5
        if char == " ":
            y -= SIZE * scale * 0.5

    print("Set back to custom Elbow 120 pose ...")
    head_move = reachy.head.goto_posture("default")
    r_arm_120 = reachy.r_arm.goto([35, -15, -15, -120, 0, 0, 0], wait=True)
