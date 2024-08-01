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
    # In A position, the effector is at (0.4, -0.5, -0.2) in the world frame
    # In B position, the effector is at (0.4, -0.5, 0) in the world frame
    # In C position, the effector is at (0.4, -0.3, 0.0) in the world frame
    # In D position, the effector is at (0.4, -0.3, -0.2) in the world frame

    # Going from A to B
    for z in np.arange(-0.2, 0.01, 0.01):
        jacobian = build_pose_matrix(0.4, -0.5, z)
        ik = reachy.r_arm.inverse_kinematics(jacobian)

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        reachy.send_goal_positions()
        time.sleep(0.1)

    # Going from B to C
    for y in np.arange(-0.5, -0.29, 0.01):
        jacobian = build_pose_matrix(0.4, y, 0.0)
        ik = reachy.r_arm.inverse_kinematics(jacobian)

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        reachy.send_goal_positions()
        time.sleep(0.1)

    # Going from C to D
    for z in np.arange(0.0, -0.21, -0.01):
        jacobian = build_pose_matrix(0.4, -0.3, z)
        ik = reachy.r_arm.inverse_kinematics(jacobian)

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        reachy.send_goal_positions()
        time.sleep(0.1)

    # Going from D to A
    for y in np.arange(-0.3, -0.51, -0.01):
        jacobian = build_pose_matrix(0.4, y, -0.2)
        ik = reachy.r_arm.inverse_kinematics(jacobian)

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        reachy.send_goal_positions()
        time.sleep(0.1)


def main_test():
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if not reachy.is_connected():
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()
    print("Putting each joint at 0 degrees angle")
    time.sleep(0.5)
    for joint in reachy.joints.values():
        joint.goal_position = 0
    reachy.send_goal_positions()

    print("Putting the right arm in default pose")
    time.sleep(1.0)
    reachy.r_arm.shoulder.roll.goal_position = -10
    reachy.r_arm.elbow.yaw.goal_position = -15
    reachy.send_goal_positions()

    print("Putting the right elbow pitch at -90 degrees angle")
    time.sleep(1.0)
    reachy.r_arm.elbow.pitch.goal_position = -90
    reachy.send_goal_positions()

    time.sleep(2.0)

    print("Putting back the right elbow pitch at 0 degrees angle")
    time.sleep(0.5)
    reachy.r_arm.elbow.pitch.goal_position = 0
    reachy.send_goal_positions()

    print("Reproducing the square movement without using goto")
    time.sleep(1.0)
    follow_square(reachy)

    print("Going back to default position")
    time.sleep(0.5)
    for joint in reachy.joints.values():
        joint.goal_position = 0
    reachy.r_arm.shoulder.roll.goal_position = -10
    reachy.r_arm.elbow.yaw.goal_position = -15
    reachy.send_goal_positions()

    print("Finished testing, disconnecting from Reachy...")
    time.sleep(0.5)
    reachy.disconnect()


if __name__ == "__main__":
    main_test()
