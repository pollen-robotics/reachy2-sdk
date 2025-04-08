"""Example of how to draw a square with Reachy's right arm."""

import logging
import time

import numpy as np
import numpy.typing as npt

from reachy2_sdk import ReachySDK


def build_pose_matrix(x: float, y: float, z: float) -> npt.NDArray[np.float64]:
    """Build a 4x4 pose matrix for a given position in 3D space, with the effector at a fixed orientation.

    Args:
        x: The x-coordinate of the position.
        y: The y-coordinate of the position.
        z: The z-coordinate of the position.

    Returns:
        A 4x4 NumPy array representing the pose matrix.
    """
    # The effector is always at the same orientation in the world frame
    return np.array(
        [
            [0, 0, -1, x],
            [0, 1, 0, y],
            [1, 0, 0, z],
            [0, 0, 0, 1],
        ]
    )


def draw_square(reachy: ReachySDK) -> None:
    """Draw a square path with Reachy's right arm in 3D space.

    This function commands Reachy's right arm to move in a square pattern
    using four predefined positions (A, B, C, and D) in the world frame.
    The square is drawn by moving the arm sequentially through these positions:
    - A: (0.4, -0.5, -0.2)
    - B: (0.4, -0.5, 0)
    - C: (0.4, -0.3, 0)
    - D: (0.4, -0.3, -0.2)

    see https://pollen-robotics.github.io/reachy2-docs/developing-with-reachy-2/basics/4-use-arm-kinematics/
    for Reachy's coordinate system

    Each movement uses inverse kinematics to calculate the required joint
    positions to achieve the target pose and then sends the commands to
    Reachy's arm to execute the movements.

    Args:
        reachy: An instance of the ReachySDK used to control the robot.
    """
    # Going from A to B
    target_pose = build_pose_matrix(0.4, -0.5, 0)
    ik = reachy.r_arm.inverse_kinematics(target_pose)
    reachy.r_arm.goto(ik, duration=2.0, degrees=True)

    current_pos = reachy.r_arm.forward_kinematics()
    print("Pose B: ", current_pos)

    # Going from B to C
    target_pose = build_pose_matrix(0.4, -0.3, 0)
    ik = reachy.r_arm.inverse_kinematics(target_pose)
    reachy.r_arm.goto(ik, duration=2.0, degrees=True)

    current_pos = reachy.r_arm.forward_kinematics()
    print("Pose C: ", current_pos)

    # Going from C to D
    target_pose = build_pose_matrix(0.4, -0.3, -0.2)
    ik = reachy.r_arm.inverse_kinematics(target_pose)
    reachy.r_arm.goto(ik, duration=2.0, degrees=True)

    current_pos = reachy.r_arm.forward_kinematics()
    print("Pose D: ", current_pos)

    # Going from D to A
    target_pose = build_pose_matrix(0.4, -0.5, -0.2)
    ik = reachy.r_arm.inverse_kinematics(target_pose)
    reachy.r_arm.goto(ik, duration=2.0, degrees=True, wait=True)

    current_pos = reachy.r_arm.forward_kinematics()
    print("Pose A: ", current_pos)


def goto_to_point_A(reachy: ReachySDK) -> None:
    """Move Reachy's right arm to Point A in 3D space.

    This function commands Reachy's right arm to move to a specified target position
    (Point A) in the world frame, which is located at (0.4, -0.5, -0.2).

    Args:
        reachy: An instance of the ReachySDK used to control the robot.
    """
    # position of point A in space
    target_pose = build_pose_matrix(0.4, -0.5, -0.2)
    # get the position in the joint space
    joints_positions = reachy.r_arm.inverse_kinematics(target_pose)
    # move Reachy's right arm to this point
    reachy.r_arm.goto(joints_positions, duration=2, wait=True)


if __name__ == "__main__":
    print("Reachy SDK example: draw square")

    logging.basicConfig(level=logging.INFO)
    reachy = ReachySDK(host="localhost")

    if not reachy.is_connected:
        exit("Reachy is not connected.")

    print("Turning on Reachy")
    reachy.turn_on()

    time.sleep(0.2)

    print("Set to Elbow 90 pose ...")
    goto_ids = reachy.goto_posture("elbow_90", wait=True)
    # wait_for_pose_to_finish(goto_ids)

    print("Move to point A")
    goto_to_point_A(reachy)

    print("Draw a square with the right arm ...")
    draw_square(reachy)

    print("Set to Zero pose ...")
    goto_ids = reachy.goto_posture("default", wait=True)
    # wait_for_pose_to_finish(goto_ids)

    time.sleep(0.2)

    exit("Exiting example")
