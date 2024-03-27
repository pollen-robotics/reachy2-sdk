import time

import numpy as np

from reachy2_sdk import ReachySDK


def build_pose_matrix(x: float, y: float, z: float):
    # The effector is always at the same orientation in the world frame
    return np.array(
        [
            [0, 0, -1, x],
            [0, 1, 0, y],
            [1, 0, 0, z],
            [0, 0, 0, 1],
        ]
    )


def follow_square(reachy: ReachySDK):
    # In A position, the effector is at (0.3, -0,4, -0.3) in the world frame
    # In B position, the effector is at (0.3, -0.4, 0) in the world frame
    # In C position, the effector is at (0.3, -0.1, 0.0) in the world frame
    # In D position, the effector is at (0.3, -0.1, -0.3) in the world frame

    # Going from A to B
    for z in np.arange(-0.3, 0.1, 0.01):
        jacobian = build_pose_matrix(0.3, -0.4, z)
        ik = reachy.r_arm.inverse_kinematics(jacobian)

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        time.sleep(0.1)

    # Going from B to C
    for y in np.arange(-0.4, -0.1, 0.01):
        jacobian = build_pose_matrix(0.3, y, 0.0)
        ik = reachy.r_arm.inverse_kinematics(jacobian)

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        time.sleep(0.1)

    # Going from C to D
    for z in np.arange(0.0, -0.3, -0.01):
        jacobian = build_pose_matrix(0.3, -0.1, z)
        ik = reachy.r_arm.inverse_kinematics(jacobian)

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        time.sleep(0.1)

    # Going from D to A
    for y in np.arange(-0.1, -0.4, -0.01):
        jacobian = build_pose_matrix(0.3, y, -0.3)
        ik = reachy.r_arm.inverse_kinematics(jacobian)

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        time.sleep(0.1)


def main_test():
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy.grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()
    print("Putting each joint at 0 degrees angle")
    time.sleep(0.5)
    for joint in reachy.joints.values():
        joint.goal_position = 0

    print("Putting the right arm at 90 degrees angle")
    time.sleep(1.0)
    reachy.r_arm.elbow.pitch.goal_position = -90

    time.sleep(2.0)

    print("Putting the right arm at 0 degrees angle")
    time.sleep(0.5)
    reachy.r_arm.elbow.pitch.goal_position = 0

    print("Reproducing the square movement without using goto")
    time.sleep(1.0)
    follow_square(reachy)

    print("Going back to initial position")
    time.sleep(0.5)
    for joint in reachy.joints.values():
        joint.goal_position = 0

    print("Finished testing, disconnecting from Reachy...")
    time.sleep(0.5)
    reachy.disconnect()


if __name__ == "__main__":
    main_test()
