from cgi import test
import time

import numpy as np

from reachy2_sdk import ReachySDK

from reachy2_sdk_api.arm_pb2 import Arm as Arm_proto
from reachy2_sdk_api.arm_pb2 import (
    ArmCartesianGoal,
    ArmEndEffector,
    ArmFKRequest,
    ArmIKRequest,
    ArmJointGoal,
    ArmLimits,
    ArmPosition,
    ArmState,
    ArmTemperatures,
)
from reachy2_sdk_api.arm_pb2_grpc import ArmServiceStub
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.kinematics_pb2 import (
    ExtEulerAngles,
    ExtEulerAnglesTolerances,
    Matrix3x3,
    Matrix4x4,
    Point,
    PointDistanceTolerances,
    Quaternion,
    Rotation3d,
)
from reachy2_sdk_api.goto_pb2 import (
    CartesianGoal,
    JointsGoal,
    GoToId,
    GoToGoalStatus,
    GoToAck,
    GoalStatus,
)
from reachy2_sdk_api.orbita2d_pb2 import Pose2d
from reachy2_sdk_api.part_pb2 import PartId


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


def test_goto_joint(reachy: ReachySDK):
    # In A position, the effector is at (0.3, -0,4, -0.3) in the world frame
    # In B position, the effector is at (0.3, -0.4, 0) in the world frame
    # In C position, the effector is at (0.3, -0.1, 0.0) in the world frame
    # In D position, the effector is at (0.3, -0.1, -0.3) in the world frame

    pose = build_pose_matrix(0.3, -0.4, -0.3)
    ik = reachy.r_arm.inverse_kinematics(pose)
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
    init_pose(reachy)


def test_both_arms(reachy: ReachySDK):
    # In A position, the effector is at (0.3, -0,4, -0.3) in the world frame
    # In B position, the effector is at (0.3, -0.4, 0) in the world frame
    # In C position, the effector is at (0.3, -0.1, 0.0) in the world frame
    # In D position, the effector is at (0.3, -0.1, -0.3) in the world frame

    pose = build_pose_matrix(0.3, -0.4, -0.3)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)
    time.sleep(1.0)
    pose = build_pose_matrix(0.3, 0.4, -0.3)
    ik = reachy.l_arm.inverse_kinematics(pose)
    reachy.l_arm.goto_joints(ik, 2.0, degrees=True)

    pose = build_pose_matrix(0.3, -0.4, 0.0)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)
    pose = build_pose_matrix(0.3, 0.4, 0.0)
    ik = reachy.l_arm.inverse_kinematics(pose)
    reachy.l_arm.goto_joints(ik, 2.0, degrees=True)

    pose = build_pose_matrix(0.3, -0.1, 0.0)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)
    pose = build_pose_matrix(0.3, 0.1, 0.0)
    ik = reachy.l_arm.inverse_kinematics(pose)
    reachy.l_arm.goto_joints(ik, 2.0, degrees=True)

    pose = build_pose_matrix(0.3, -0.1, -0.3)
    ik = reachy.r_arm.inverse_kinematics(pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)
    pose = build_pose_matrix(0.3, 0.1, -0.3)
    ik = reachy.l_arm.inverse_kinematics(pose)
    id = reachy.l_arm.goto_joints(ik, 2.0, degrees=True)

    while is_goto_finised(reachy, id) == False:
        time.sleep(0.1)
    init_pose(reachy)


def is_goto_finised(reachy: ReachySDK, id: int, verbose: bool = False):
    state = reachy.r_arm.get_goto_state(id)
    if verbose:
        print(f"State fo goal {id}: {state}")
    return (
        state.goal_status == GoalStatus.STATUS_ABORTED
        or state.goal_status == GoalStatus.STATUS_CANCELED
        or state.goal_status == GoalStatus.STATUS_SUCCEEDED
    )


def test_state(reachy: ReachySDK):
    pose = build_pose_matrix(0.3, -0.4, -0.3)
    ik = reachy.r_arm.inverse_kinematics(pose)
    id = reachy.r_arm.goto_joints(ik, 2.0, degrees=True)
    print(f"goto id={id}")
    while is_goto_finised(reachy, id, verbose=True) == False:
        time.sleep(0.1)

    time.sleep(1.0)
    init_pose(reachy)


def init_pose(reachy: ReachySDK):
    print("Putting each joint at 0 degrees angle with a goto")
    id1 = reachy.r_arm.goto_joints([0, 0, 0, 0, 0, 0, 0], 1.0, degrees=True)
    id2 = reachy.l_arm.goto_joints([0, 0, 0, 0, 0, 0, 0], 1.0, degrees=True)
    while is_goto_finised(reachy, id1) == False or is_goto_finised(reachy, id2) == False:
        time.sleep(0.1)


def test_goto_cancel(reachy: ReachySDK):
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
    reachy.r_arm.cancel_all_goto()
    print("The arm should be stopped now.")
    time.sleep(1.0)
    init_pose(reachy)

    print("End of cancel test!")


def test_goto_cartesian(reachy: ReachySDK):
    id = reachy.r_arm.goto_from_matrix(build_pose_matrix(0.3, -0.4, -0.3), 2.0)
    while is_goto_finised(reachy, id) is False:
        time.sleep(0.1)
    init_pose(reachy)

    # id = reachy.r_arm.goto([0.3, -0.4, -0.3], [0, 0, 0], duration=2.0)
    # while is_goto_finised(reachy, id) is False:
    #     time.sleep(0.1)
    # init_pose(reachy)
    print("End of cartesian goto test!")


def test_goto_rejection(reachy: ReachySDK):
    print("Trying a goto with duration 0.0")
    id = reachy.r_arm.goto_from_matrix(build_pose_matrix(0.3, -0.4, -0.3), 0.0)
    print(f"goto id={id}")
    if id.id < 0:
        print("The goto was rejected as expected!")
    else:
        print("The goto was not rejected, this is NOT expected...")
    init_pose(reachy)


def test_goto_head(reachy: ReachySDK):
    id = reachy.head.goto_joints([0, 0, 30], 1.0, degrees=True)
    while is_goto_finised(reachy, id, verbose=True) is False:
        time.sleep(0.1)
    # reachy.head.look_at(0.3, 0.0, 0.0, 1.0)


def main_test():
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy.grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

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

    # print("\n###7)Testing the goto_head function")
    # test_goto_head(reachy)

    print("Finished testing, disconnecting from Reachy...")
    time.sleep(0.5)
    reachy.disconnect()


if __name__ == "__main__":
    main_test()
