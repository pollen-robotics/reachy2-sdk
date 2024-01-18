import time
from math import degrees

import numpy as np
import numpy.typing as npt
from reachy2_sdk_api.goto_pb2 import GoalStatus

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


def test_goto_joint(reachy: ReachySDK) -> None:
    # In A position, the effector is at (0.3, -0,4, -0.3) in the world frame
    # In B position, the effector is at (0.3, -0.4, 0) in the world frame
    # In C position, the effector is at (0.3, -0.1, 0.0) in the world frame
    # In D position, the effector is at (0.3, -0.1, -0.3) in the world frame

    print("With interpolation_mode='minimum_jerk'")

    pose = build_pose_matrix(0.3, -0.4, -0.3)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True, interpolation_mode="minimum_jerk")

    pose = build_pose_matrix(0.3, -0.4, 0.0)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True, interpolation_mode="minimum_jerk")

    pose = build_pose_matrix(0.3, -0.1, 0.0)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True, interpolation_mode="minimum_jerk")

    pose = build_pose_matrix(0.3, -0.1, -0.3)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True, interpolation_mode="minimum_jerk")

    time.sleep(1.0)
    init_pose(reachy)

    print("With interpolation_mode='linear'")

    pose = build_pose_matrix(0.3, -0.4, -0.3)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True, interpolation_mode="linear")

    pose = build_pose_matrix(0.3, -0.4, 0.0)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True, interpolation_mode="linear")

    pose = build_pose_matrix(0.3, -0.1, 0.0)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True, interpolation_mode="linear")

    pose = build_pose_matrix(0.3, -0.1, -0.3)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True, interpolation_mode="linear")

    time.sleep(1.0)
    init_pose(reachy)


def test_both_arms(reachy: ReachySDK) -> None:
    # In A position, the effector is at (0.3, -0,4, -0.3) in the world frame
    # In B position, the effector is at (0.3, -0.4, 0) in the world frame
    # In C position, the effector is at (0.3, -0.1, 0.0) in the world frame
    # In D position, the effector is at (0.3, -0.1, -0.3) in the world frame

    pose = build_pose_matrix(0.3, -0.45, -0.3)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)
    time.sleep(1.0)
    pose = build_pose_matrix(0.3, 0.45, -0.3)
    ik = reachy.l_arm.inverse_kinematics(pose)
    reachy.l_arm.goto_joints(ik, 2.0, degrees=True)

    pose = build_pose_matrix(0.3, -0.45, 0.0)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)
    pose = build_pose_matrix(0.3, 0.45, 0.0)
    ik = reachy.l_arm.inverse_kinematics(pose)
    reachy.l_arm.goto_joints(ik, 2.0, degrees=True)

    pose = build_pose_matrix(0.3, -0.25, 0.0)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)
    pose = build_pose_matrix(0.3, 0.25, 0.0)
    ik = reachy.l_arm.inverse_kinematics(pose)
    reachy.l_arm.goto_joints(ik, 2.0, degrees=True)

    pose = build_pose_matrix(0.3, -0.15, -0.3)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)
    pose = build_pose_matrix(0.3, 0.15, -0.3)
    ik = reachy.l_arm.inverse_kinematics(pose)
    id = reachy.l_arm.goto_joints(ik, 2.0, degrees=True)

    while is_goto_finised(reachy, id) is False:
        time.sleep(0.1)
    init_pose(reachy)


def is_goto_finised(reachy: ReachySDK, id: int, verbose: bool = False) -> bool:
    state = reachy.r_arm.get_goto_state(id)
    if verbose:
        print(f"State fo goal {id}: {state}")
    result = bool(
        state.goal_status == GoalStatus.STATUS_ABORTED
        or state.goal_status == GoalStatus.STATUS_CANCELED
        or state.goal_status == GoalStatus.STATUS_SUCCEEDED
    )
    return result


def test_state(reachy: ReachySDK) -> None:
    pose = build_pose_matrix(0.3, -0.4, -0.3)
    ik = reachy.r_arm.inverse_kinematics(pose)
    id = reachy.r_arm.goto_joints(ik, 2.0, degrees=True)
    print(f"goto id={id}")
    while is_goto_finised(reachy, id, verbose=True) is False:
        time.sleep(0.1)

    time.sleep(1.0)
    init_pose(reachy)


def init_pose(reachy: ReachySDK) -> None:
    print("Putting each joint at 0 degrees angle with a goto")
    id1 = reachy.r_arm.goto_joints([0, 0, 0, 0, 0, 0, 0], 2.0, degrees=True)
    id2 = reachy.l_arm.goto_joints([0, 0, 0, 0, 0, 0, 0], 2.0, degrees=True)
    while is_goto_finised(reachy, id1) is False or is_goto_finised(reachy, id2) is False:
        time.sleep(0.1)


def test_goto_cancel(reachy: ReachySDK) -> None:
    pose = build_pose_matrix(0.3, -0.4, -0.3)
    ik = reachy.r_arm.inverse_kinematics(pose)
    id = reachy.r_arm.goto_joints(ik, 2.0, degrees=True)

    time.sleep(1.0)
    print("Canceling the goto goal!")
    reachy.r_arm.cancel_goto_by_id(id)
    print("The arm should be stopped now.")
    time.sleep(1.0)
    init_pose(reachy)
    time.sleep(0.5)

    print("Setting up a lot of gotos!")
    pose = build_pose_matrix(0.3, -0.4, -0.3)
    reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)

    pose = build_pose_matrix(0.3, -0.4, 0.0)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)

    pose = build_pose_matrix(0.3, -0.1, 0.0)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)

    pose = build_pose_matrix(0.3, -0.1, -0.3)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)

    time.sleep(1.0)
    print("Canceling all gotos!")
    reachy.cancel_all_goto()
    print("The arm should be stopped now.")
    time.sleep(1.0)
    init_pose(reachy)

    print("End of cancel test!")


def test_goto_cartesian(reachy: ReachySDK) -> None:
    id = reachy.r_arm.goto_from_matrix(build_pose_matrix(0.3, -0.4, -0.3), 2.0)
    while is_goto_finised(reachy, id) is False:
        time.sleep(0.1)

    print("Trying with weird q0")
    q0 = [np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2]
    id = reachy.r_arm.goto_from_matrix(build_pose_matrix(0.3, -0.4, -0.3), 2.0, q0=q0)
    while is_goto_finised(reachy, id) is False:
        time.sleep(0.1)

    time.sleep(1.0)
    init_pose(reachy)

    print("End of cartesian goto test!")


def test_goto_rejection(reachy: ReachySDK) -> None:
    print("Trying a goto with duration 0.0")
    id = reachy.r_arm.goto_from_matrix(build_pose_matrix(0.3, -0.4, -0.3), 0.0)
    print(f"goto id={id}")
    if id.id < 0:
        print("The goto was rejected as expected!")
    else:
        print("The goto was not rejected, this is NOT expected...")
    init_pose(reachy)


def test_head_orient(reachy: ReachySDK) -> None:
    id = reachy.head.rotate_to(0, 0, 0.5, duration=1.0, interpolation_mode="minimum_jerk", degrees=False)
    while is_goto_finised(reachy, id, verbose=True) is False:
        time.sleep(0.1)
    id = reachy.head.rotate_to(0, 0.5, 0.5, duration=1.0, interpolation_mode="minimum_jerk", degrees=False)
    while is_goto_finised(reachy, id, verbose=True) is False:
        time.sleep(0.1)
    id = reachy.head.rotate_to(0.5, 0.5, 0.5, duration=1.0, interpolation_mode="minimum_jerk", degrees=False)
    while is_goto_finised(reachy, id, verbose=True) is False:
        time.sleep(0.1)
    id = reachy.head.rotate_to(0.5, 0.5, 0.0, duration=1.0, interpolation_mode="minimum_jerk", degrees=False)
    while is_goto_finised(reachy, id, verbose=True) is False:
        time.sleep(0.1)
    id = reachy.head.rotate_to(0.5, 0.0, 0.0, duration=1.0, interpolation_mode="minimum_jerk", degrees=False)
    while is_goto_finised(reachy, id, verbose=True) is False:
        time.sleep(0.1)
    id = reachy.head.rotate_to(0.0, 0.0, 0.0, duration=1.0, interpolation_mode="minimum_jerk", degrees=False)
    while is_goto_finised(reachy, id, verbose=True) is False:
        time.sleep(0.1)


def test_head_look_at(reachy: ReachySDK) -> None:
    list_of_poses = [[1.0, 0.0, 0.0], [1.0, 0.2, -0.5], [1.0, -0.5, 0.3]]
    for pose in list_of_poses:
        id = reachy.head.look_at(pose[0], pose[1], pose[2], duration=1.0, interpolation_mode="minimum_jerk")
        if id.id < 0:
            print("The goto was rejected!")
            return
        while is_goto_finised(reachy, id, verbose=True) is False:
            time.sleep(0.1)


def main_test() -> None:
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy.grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()
    init_pose(reachy)

    print("\n###1)Testing the goto_joints function, drawing a square")
    test_goto_joint(reachy)

    print("\n###2)Testing the get_goto_state function")
    test_state(reachy)

    print("\n###3)Testing the goto_cancel function")
    test_goto_cancel(reachy)

    print("\n###4)Testing both arms")
    test_both_arms(reachy)

    print("\n###5)Testing the goto_cartesian function")
    test_goto_cartesian(reachy)
    print("\n###6)Testing goto REJECTION")
    test_goto_rejection(reachy)

    print("\n###7)Testing the goto_head function")
    test_head_orient(reachy)

    while True:
        print("\n###X)Testing both arms ad vitam eternam")
        test_both_arms(reachy)

    # print("\n###8)Testing the look_at function")
    # test_head_look_at(reachy)

    print("Finished testing, disconnecting from Reachy...")
    time.sleep(0.5)

    reachy.disconnect()


def head_test() -> None:
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy.grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    print("Turning on...")

    reachy.turn_on()
    print("Init pose...")

    init_pose(reachy)

    # print("\n###7)Testing the goto_head function")
    test_head_orient(reachy)

    print("\n###8)Testing the look_at function")
    test_head_look_at(reachy)

    print("Finished testing, disconnecting from Reachy...")
    time.sleep(0.5)

    reachy.disconnect()
    ReachySDK.clear()


def deco_test() -> None:
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)

    reachy.disconnect()
    ReachySDK.clear()

    print("Trying AGAIN to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)

    reachy.disconnect()
    ReachySDK.clear()


if __name__ == "__main__":
    head_test()
    # main_test()
    # deco_test()
