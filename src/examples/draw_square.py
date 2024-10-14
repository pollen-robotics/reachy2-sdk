import logging
import time

import numpy as np
import numpy.typing as npt

from reachy2_sdk import ReachySDK


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


def draw_square(reachy: ReachySDK) -> None:
    # In A position, the effector is at (0.4, -0,5, -0.2) in the world frame
    # In B position, the effector is at (0.4, -0.5, 0) in the world frame
    # In C position, the effector is at (0.4, -0.3, 0.0) in the world frame
    # In D position, the effector is at (0.4, -0.3, -0.2) in the world frame
    # see https://docs.pollen-robotics.com/sdk/first-moves/kinematics/ for Reachy's coordinate system

    # Going from A to B
    target_pose = build_pose_matrix(0.4, -0.5, 0)
    ik = reachy.r_arm.inverse_kinematics(target_pose)
    reachy.r_arm.goto_joints(ik, duration=2.0, degrees=True)

    current_pos = reachy.r_arm.forward_kinematics()
    print("Pose B: ", current_pos)

    # Going from B to C
    target_pose = build_pose_matrix(0.4, -0.3, 0)
    ik = reachy.r_arm.inverse_kinematics(target_pose)
    reachy.r_arm.goto_joints(ik, duration=2.0, degrees=True)

    current_pos = reachy.r_arm.forward_kinematics()
    print("Pose C: ", current_pos)

    # Going from C to D
    target_pose = build_pose_matrix(0.4, -0.3, -0.2)
    ik = reachy.r_arm.inverse_kinematics(target_pose)
    reachy.r_arm.goto_joints(ik, duration=2.0, degrees=True)

    current_pos = reachy.r_arm.forward_kinematics()
    print("Pose D: ", current_pos)

    # Going from D to A
    target_pose = build_pose_matrix(0.4, -0.5, -0.2)
    ik = reachy.r_arm.inverse_kinematics(target_pose)
    reachy.r_arm.goto_joints(ik, duration=2.0, degrees=True, wait=True)

    current_pos = reachy.r_arm.forward_kinematics()
    print("Pose A: ", current_pos)


def goto_to_point_A(reachy: ReachySDK) -> None:
    # position of point A in space
    target_pose = build_pose_matrix(0.4, -0.5, -0.2)
    # get the position in the joint space
    joints_positions = reachy.r_arm.inverse_kinematics(target_pose)
    # move Reachy's right arm to this point
    reachy.r_arm.goto_joints(joints_positions, duration=2)


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

    print("Turning off Reachy")
    reachy.turn_off()

    time.sleep(0.2)

    exit("Exiting example")
