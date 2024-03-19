import time

import numpy as np
import numpy.typing as npt
from reachy2_sdk import ReachySDK
from reachy2_sdk_api.goto_pb2 import GoalStatus, GoToId


def connect_to_reachy():
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")
    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return
    return reachy


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


def turn_on():
    reachy = connect_to_reachy()

    print("Turning on...")

    reachy.turn_on()

    reachy.disconnect()
    ReachySDK.clear()


def turn_on_debug():
    reachy = connect_to_reachy()

    print("Turning on...")

    reachy.turn_on()

    print("The arms should move now")

    pose = build_pose_matrix(0.3, -0.45, -0.3)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)
    pose = build_pose_matrix(0.3, 0.45, -0.3)
    ik = reachy.l_arm.inverse_kinematics(pose)
    reachy.l_arm.goto_joints(ik, 2.0, degrees=True)
    reachy.head.rotate_to(np.pi / 3, np.pi / 3, np.pi / 3, duration=1.5, interpolation_mode="minimum_jerk", degrees=False)
    time.sleep(2.0)

    print("The arm should NOT move now !")
    reachy.turn_off()
    for joint in reachy.joints.values():
        joint.goal_position = 0
    time.sleep(3.0)

    print("The arm should NOT move now either !")
    reachy.turn_on()
    time.sleep(2.0)

    reachy.disconnect()
    ReachySDK.clear()


if __name__ == "__main__":
    # turn_on()
    turn_on_debug()
