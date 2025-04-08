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


def goto_A(reachy: ReachySDK):
    jacobian = build_pose_matrix(0.4, -0.5, -0.2)
    ik = reachy.r_arm.inverse_kinematics(jacobian)

    reachy.r_arm.goto(ik, wait=True)


def follow_square(reachy: ReachySDK):
    # In A position, the effector is at (0.4, -0.5, -0.2) in the world frame
    # In B position, the effector is at (0.4, -0.5, 0) in the world frame
    # In C position, the effector is at (0.4, -0.3, 0.0) in the world frame
    # In D position, the effector is at (0.4, -0.3, -0.2) in the world frame

    # Going from A to B
    for z in np.arange(-0.2, 0.01, 0.005):
        jacobian = build_pose_matrix(0.4, -0.5, z)
        ik = reachy.r_arm.inverse_kinematics(jacobian)

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        reachy.send_goal_positions(check_positions=False)
        time.sleep(0.05)

    # Going from B to C
    for y in np.arange(-0.5, -0.29, 0.005):
        jacobian = build_pose_matrix(0.4, y, 0.0)
        ik = reachy.r_arm.inverse_kinematics(jacobian)

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        reachy.send_goal_positions(check_positions=False)
        time.sleep(0.05)

    # Going from C to D
    for z in np.arange(0.0, -0.21, -0.005):
        jacobian = build_pose_matrix(0.4, -0.3, z)
        ik = reachy.r_arm.inverse_kinematics(jacobian)

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        reachy.send_goal_positions(check_positions=False)
        time.sleep(0.05)

    # Going from D to A
    for y in np.arange(-0.3, -0.51, -0.005):
        jacobian = build_pose_matrix(0.4, y, -0.2)
        ik = reachy.r_arm.inverse_kinematics(jacobian)

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        reachy.send_goal_positions(check_positions=False)
        time.sleep(0.05)


def main_test():
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="10.0.0.201")

    time.sleep(1.0)
    if not reachy.is_connected():
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()
    print("Putting each joint at 0 degrees angle")
    reachy.r_arm.goto([0, 0, 0, 0, 0, 0, 0])
    reachy.l_arm.goto([0, 0, 0, 0, 0, 0, 0])
    reachy.head.goto([0, 0, 0], wait=True)

    print("Putting both arms in default pose")
    time.sleep(1.0)
    reachy.goto_posture(wait=True)

    print("Putting the right elbow pitch at -90 degrees angle")
    time.sleep(1.0)
    reachy.r_arm.goto_posture("elbow_90", wait=True)

    time.sleep(2.0)

    print("Putting back the right elbow pitch at 0 degrees angle")
    time.sleep(0.5)
    reachy.r_arm.goto_posture(wait=True)

    print("Ready to do the square movement")
    time.sleep(0.5)

    print("Going to first position")
    time.sleep(0.5)
    goto_A(reachy)

    print("Reproducing the square movement without using goto")
    time.sleep(1.0)
    follow_square(reachy)

    print("Going back to default position")
    time.sleep(0.5)
    reachy.goto_posture(wait=True)

    print("Finished testing, disconnecting from Reachy...")
    time.sleep(0.5)
    reachy.disconnect()


if __name__ == "__main__":
    main_test()
