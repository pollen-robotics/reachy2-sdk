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


def is_goto_finised(reachy: ReachySDK, id: int):
    state = reachy.r_arm.get_goto_state(id)
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
    while True:
        state = reachy.r_arm.get_goto_state(id)
        print(f"{state}")
        if (
            state.goal_status == GoalStatus.STATUS_ABORTED
            or state.goal_status == GoalStatus.STATUS_CANCELED
            or state.goal_status == GoalStatus.STATUS_SUCCEEDED
        ):
            print("End of status test!")
            break
        time.sleep(0.1)

    time.sleep(1.0)
    init_pose(reachy)


def init_pose(reachy: ReachySDK):
    print("Putting each joint at 0 degrees angle with a goto")
    reachy.r_arm.goto_joints([0, 0, 0, 0, 0, 0, 0], 1.0, degrees=True)
    reachy.l_arm.goto_joints([0, 0, 0, 0, 0, 0, 0], 1.0, degrees=True)
    time.sleep(1.01)


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


def main_test():
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy.grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    init_pose(reachy)

    # print("Testing the goto_joints function")
    # test_goto_joint(reachy)

    # print("Testing the get_goto_state function")
    # test_state(reachy)

    # print("Testing the goto_cancel function")
    # test_goto_cancel(reachy)

    print("Testing both arms")
    test_both_arms(reachy)

    print("Finished testing, disconnecting from Reachy...")
    time.sleep(0.5)
    reachy.disconnect()


if __name__ == "__main__":
    main_test()
