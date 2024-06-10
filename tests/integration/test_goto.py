import time

import numpy as np
import numpy.typing as npt
from reachy2_sdk_api.goto_pb2 import GoalStatus, GoToId
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from sympy import li

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


def get_homogeneous_matrix_msg_from_euler(
    position: tuple = (0, 0, 0),  # (x, y, z)
    euler_angles: tuple = (0, 0, 0),  # (roll, pitch, yaw)
    degrees: bool = False,
):
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = R.from_euler("xyz", euler_angles, degrees=degrees).as_matrix()
    homogeneous_matrix[:3, 3] = position
    return homogeneous_matrix


def decompose_matrix(matrix):
    """Decompose a homogeneous 4x4 matrix into rotation (quaternion) and translation components."""
    rotation_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]
    rotation = R.from_matrix(rotation_matrix)
    return rotation, translation


def recompose_matrix(rotation, translation):
    """Recompose a homogeneous 4x4 matrix from rotation (quaternion) and translation components."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation  # .as_matrix()
    matrix[:3, 3] = translation
    return matrix


def interpolate_matrices(matrix1, matrix2, t):
    """Interpolate between two 4x4 matrices at time t [0, 1]."""
    rot1, trans1 = decompose_matrix(matrix1)
    rot2, trans2 = decompose_matrix(matrix2)

    # Linear interpolation for translation
    trans_interpolated = (1 - t) * trans1 + t * trans2

    # SLERP for rotation interpolation
    q1 = Quaternion(matrix=rot1.as_matrix())
    q2 = Quaternion(matrix=rot2.as_matrix())
    q_interpolated = Quaternion.slerp(q1, q2, t)
    rot_interpolated = q_interpolated.rotation_matrix

    # Recompose the interpolated matrix
    interpolated_matrix = recompose_matrix(rot_interpolated, trans_interpolated)
    return interpolated_matrix


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

    while is_goto_finished(reachy, id) is False:
        time.sleep(0.1)
    init_pose(reachy)


def is_goto_finished(reachy: ReachySDK, id: GoToId, verbose=False) -> bool:
    state = reachy.get_goto_state(id)
    if verbose:
        print(f"Goal status: {state.goal_status}")
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
    while is_goto_finished(reachy, id, verbose=True) is False:
        time.sleep(0.1)

    time.sleep(1.0)
    init_pose(reachy)


def init_pose(reachy: ReachySDK) -> None:
    print("Putting each joint at 0 degrees angle with a goto")
    id1 = reachy.r_arm.goto_joints([0, 0, 0, 0, 0, 0, 0], 2.0, degrees=True)
    id2 = reachy.l_arm.goto_joints([0, 0, 0, 0, 0, 0, 0], 2.0, degrees=True)
    while is_goto_finished(reachy, id1) is False or is_goto_finished(reachy, id2) is False:
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
    delay = 2.0
    # goal = ([x, y, z], [roll, pitch, yaw])
    goals = []
    goals.extend(
        [([0.3, -0.0, -0.3], [45.0, -60, 0.0]), ([0.3, -0.0, -0.3], [45.0, -30, 0.0]), ([0.4, -0.0, -0.3], [45.0, -30, 10.0])]
    )
    goals.extend(
        [
            ([0.3, -0.4, -0.3], [0.0, -90, 0.0]),
            ([0.5, -0.4, -0.3], [0.0, -90, 0.0]),
            ([0.9, -0.4, -0.3], [0.0, -90, 0.0]),
            ([0.3, -0.4, -0.1], [0.0, -90, 0.0]),
        ]
    )
    goals.extend(
        [
            ([0.3, -0.4, -0.3], [0.0, -90, 0.0]),
            ([0.3, -0.4, -0.3], [0.0, -60, 0.0]),
            ([0.3, -0.4, -0.3], [0.0, -30, 0.0]),
            ([0.3, -0.4, -0.3], [0.0, 0, 0.0]),
        ]
    )
    # Problematic goal :
    # goals = [([0.3, -0.4, -0.3], [0.0,-30,0.0]), ([0.3, -0.4, -0.3], [0.0,0,0.0])]
    for goal in goals:
        id = reachy.r_arm.goto_from_matrix(
            get_homogeneous_matrix_msg_from_euler(position=goal[0], euler_angles=goal[1], degrees=True), delay
        )
        if id.id < 0:
            print("The goto was rejected! Unreachable pose.")
            time.sleep(1.0)
        else:
            while not reachy.is_move_finished(id):
                time.sleep(0.1)


def test_goto_single_pose(reachy: ReachySDK) -> None:
    list_of_mats = []
    list_of_mats.append(
        np.array(
            [
                [0.056889, 0.99439, -0.089147, 0.25249],
                [-0.14988, 0.096786, 0.98395, -0.099362],
                [0.98707, -0.042614, 0.15455, -0.32934],
                [0, 0, 0, 1],
            ]
        )
    )
    list_of_mats.append(
        np.array(
            [
                [0.36861, 0.089736, -0.92524, 0.37213],
                [-0.068392, 0.99525, 0.069279, -0.028012],
                [0.92706, 0.037742, 0.373, -0.38572],
                [0, 0, 0, 1],
            ]
        )
    )

    list_of_mats.append(
        np.array(
            [
                [0.29157, 0.95649, -0.010922, 0.39481],
                [-0.27455, 0.094617, 0.95691, -0.065782],
                [0.9163, -0.276, 0.29019, -0.27771],
                [0, 0, 0, 1],
            ]
        )
    )

    list_of_mats.append(
        np.array(
            [
                [0.30741, 0.95003, -0.054263, 0.38327],
                [-0.77787, 0.28373, 0.56073, -0.059699],
                [0.54811, -0.13017, 0.82622, -0.2948],
                [0, 0, 0, 1],
            ]
        )
    )

    list_of_mats.append(
        np.array(
            [
                [0.46129, 0.27709, -0.84288, 0.38394],
                [-0.22793, 0.95511, 0.18924, -0.0087053],
                [0.85747, 0.10482, 0.50373, -0.28038],
                [0, 0, 0, 1],
            ]
        )
    )
    list_of_mats.append(
        np.array(
            [
                [-0.030111, 0.99613, -0.082526, 0.31861],
                [-0.18163, 0.075737, 0.98045, 0.035154],
                [0.98291, 0.044511, 0.17864, -0.25577],
                [0, 0, 0, 1],
            ]
        )
    )

    for mat in list_of_mats:
        input("press enter to go to the next pose!")
        id = reachy.l_arm.goto_from_matrix(mat)
        if id.id < 0:
            print("The goto was rejected! Unreachable pose.")
            time.sleep(1.0)
        else:
            l_ik_sol = reachy.l_arm.inverse_kinematics(mat)
            goal_pose = reachy.l_arm.forward_kinematics(l_ik_sol)
            precision_distance_xyz_to_sol = np.linalg.norm(goal_pose[:3, 3] - mat[:3, 3])
            print(f"l2 xyz distance Ik SOL vs goal pose: {precision_distance_xyz_to_sol} with joints: {l_ik_sol}")
            while not reachy.is_move_finished(id):
                time.sleep(0.1)

    list_of_mats = []
    list_of_mats.append(
        np.array(
            [
                [-0.12009, 0.94787, -0.29517, 0.42654],
                [-0.28736, -0.31779, -0.90357, -0.069494],
                [-0.95026, -0.023686, 0.31054, -0.36334],
                [0, 0, 0, 1],
            ]
        )
    )
    list_of_mats.append(
        np.array(
            [
                [-0.23603, 0.91326, -0.33203, 0.35339],
                [-0.57781, -0.40662, -0.70767, -0.17652],
                [-0.7813, 0.02482, 0.62367, -0.40001],
                [0, 0, 0, 1],
            ]
        )
    )

    for mat in list_of_mats:
        input("press enter to go to the next pose!")
        id = reachy.r_arm.goto_from_matrix(mat)
        if id.id < 0:
            print("The goto was rejected! Unreachable pose.")
            time.sleep(1.0)
        else:
            r_ik_sol = reachy.r_arm.inverse_kinematics(mat)
            goal_pose = reachy.r_arm.forward_kinematics(r_ik_sol)
            precision_distance_xyz_to_sol = np.linalg.norm(goal_pose[:3, 3] - mat[:3, 3])
            print(f"l2 xyz distance Ik SOL vs goal pose: {precision_distance_xyz_to_sol} with joints: {r_ik_sol}")
            while not reachy.is_move_finished(id):
                time.sleep(0.1)


def test_task_space_interpolation_goto(reachy: ReachySDK) -> None:
    delay = 2.0
    # goal = ([x, y, z], [roll, pitch, yaw])
    # Problematic goal :
    # goals = [([0.3, -0.4, -0.3], [0.0,-30,0.0]), ([0.3, -0.4, -0.3], [0.0,0,0.0])]
    # mat2 = np.array(
    #     [
    #         [0.056889, 0.99439, -0.089147, 0.25249],
    #         [-0.14988, 0.096786, 0.98395, -0.099362],
    #         [0.98707, -0.042614, 0.15455, -0.32934],
    #         [0, 0, 0, 1],
    #     ]
    # )
    # mat1 = np.array([[0, 0, -1.0, 0.25249], [0, 1.0, 0, -0.099362], [1.0, 0, 0, -0.32934], [0, 0, 0, 1]])

    # r_arm
    # mat1 = build_pose_matrix(0.3, -0.45, 0.0)
    # mat2 = build_pose_matrix(0.3, -0.45, -0.3)
    # SimSim probelmatic pose on l_arm:
    mat1 = build_pose_matrix(0.3, 0.45, 0.0)
    mat2 = np.array(
        [
            [0.36861, 0.089736, -0.92524, 0.37213],
            [-0.068392, 0.99525, 0.069279, -0.028012],
            [0.92706, 0.037742, 0.373, -0.38572],
            [0, 0, 0, 1],
        ]
    )

    # mat2 = np.array([[    0.29157,     0.95649,   -0.010922,     0.39481],
    #    [   -0.27455,    0.094617,     0.95691,   -0.065782],
    #    [     0.9163,      -0.276,     0.29019,    -0.27771],
    #    [          0,           0,           0,           1]])

    # mat2 = np.array([[    0.30741,     0.95003,   -0.054263,     0.38327],
    #     [   -0.77787,     0.28373,     0.56073,   -0.059699],
    #     [    0.54811,    -0.13017,     0.82622,     -0.2948],
    #     [          0,           0,           0,           1]])
    # l2 distance between the two matrices in x, y, z only
    l2_distance_xyz = np.linalg.norm(mat1[:3, 3] - mat2[:3, 3])
    # distance in orientation TODO
    speed = 0.1
    nb_points = 20
    duration = (l2_distance_xyz / speed) / nb_points
    try:
        l_ik_sol = reachy.l_arm.inverse_kinematics(mat2)
        goal_pose = reachy.l_arm.forward_kinematics(l_ik_sol)
        precision_distance_xyz_to_sol = np.linalg.norm(goal_pose[:3, 3] - mat2[:3, 3])
        print(f"l2 xyz distance Ik SOL vs goal pose: {precision_distance_xyz_to_sol}")
    except:
        print("Goal pose is not reachable!")
    for t in np.linspace(0, 1, nb_points):
        interpolated_matrix = interpolate_matrices(mat1, mat2, t)
        id = reachy.l_arm.goto_from_matrix(interpolated_matrix, duration, interpolation_mode="linear")
        if id.id < 0:
            print(f"The goto t={t} was rejected! Unreachable pose.")
            time.sleep(duration)
        else:
            print(f"The goto t={t} was accepted.")
            while not reachy.is_move_finished(id):
                time.sleep(0.01)

    # time.sleep(duration*nb_points + 1)
    time.sleep(0.5)
    current_pose = reachy.l_arm.forward_kinematics()
    precision_distance_xyz = np.linalg.norm(current_pose[:3, 3] - mat2[:3, 3])
    print(f"l2 xyz distance to goal: {precision_distance_xyz}")


def test_goto_cartesian_with_interpolation(reachy: ReachySDK) -> None:
    pose = build_pose_matrix(0.3, -0.4, -0.3)
    reachy.r_arm.goto_from_matrix(pose, 2.0, interpolation_mode="minimum_jerk", with_cartesian_interpolation=True)

    pose = build_pose_matrix(0.3, -0.4, 0.0)
    reachy.r_arm.goto_from_matrix(pose, 2.0, interpolation_mode="minimum_jerk", with_cartesian_interpolation=True)

    pose = build_pose_matrix(0.3, -0.1, 0.0)
    reachy.r_arm.goto_from_matrix(pose, 2.0, interpolation_mode="minimum_jerk", with_cartesian_interpolation=True)

    pose = build_pose_matrix(0.3, -0.1, -0.3)
    reachy.r_arm.goto_from_matrix(pose, 2.0, interpolation_mode="minimum_jerk", with_cartesian_interpolation=True)


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
    while is_goto_finished(reachy, id, verbose=True) is False:
        time.sleep(0.1)
    id = reachy.head.rotate_to(0, 0.5, 0.5, duration=1.0, interpolation_mode="minimum_jerk", degrees=False)
    while is_goto_finished(reachy, id, verbose=True) is False:
        time.sleep(0.1)
    id = reachy.head.rotate_to(0.5, 0.5, 0.5, duration=1.0, interpolation_mode="minimum_jerk", degrees=False)
    while is_goto_finished(reachy, id, verbose=True) is False:
        time.sleep(0.1)
    id = reachy.head.rotate_to(0.5, 0.5, 0.0, duration=1.0, interpolation_mode="minimum_jerk", degrees=False)
    while is_goto_finished(reachy, id, verbose=True) is False:
        time.sleep(0.1)
    id = reachy.head.rotate_to(0.5, 0.0, 0.0, duration=1.0, interpolation_mode="minimum_jerk", degrees=False)
    while is_goto_finished(reachy, id, verbose=True) is False:
        time.sleep(0.1)
    id = reachy.head.rotate_to(0.0, 0.0, 0.0, duration=1.0, interpolation_mode="minimum_jerk", degrees=False)
    while is_goto_finished(reachy, id, verbose=True) is False:
        time.sleep(0.1)


def test_head_look_at(reachy: ReachySDK) -> None:
    list_of_poses = [[1.0, 0.0, 0.0], [1.0, 0.2, -0.5], [1.0, -0.5, 0.3]]
    for pose in list_of_poses:
        id = reachy.head.look_at(pose[0], pose[1], pose[2], duration=1.0, interpolation_mode="minimum_jerk")
        if id.id < 0:
            print("The goto was rejected!")
            return
        while is_goto_finished(reachy, id, verbose=True) is False:
            time.sleep(0.1)


def main_test() -> None:
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")

    while not reachy.is_connected():
        print("Waiting for Reachy to be ready...")
        time.sleep(0.2)
    print(reachy.info)
    reachy.turn_on()
    time.sleep(1.0)

    reachy.l_arm.goto_joints([0, 20, 0, 0, 0, 0, 0], duration=2)
    id = reachy.r_arm.goto_joints([0, -20, 0, 0, 0, 0, 0], duration=2)
    while not reachy.is_move_finished(id):
        time.sleep(0.1)

    # init_pose(reachy)

    # print("\n###1)Testing the goto_joints function, drawing a square")
    # test_goto_joint(reachy)

    # print("\n###2)Testing the get_goto_state function")
    # test_state(reachy)

    # print("\n###3)Testing the goto_cancel function")
    # test_goto_cancel(reachy)

    # print("\n###4)Testing both arms")
    # while True:
    #     test_both_arms(reachy)

    # print("\n###5)Testing the goto_cartesian function")
    # test_goto_single_pose(reachy)
    while True:
        # test_goto_cartesian(reachy)
        # test_task_space_interpolation_goto(reachy)
        break
    # print("\n###6)Testing goto REJECTION")
    # test_goto_rejection(reachy)

    # print("\n###7)Testing the goto_head function")
    # test_head_orient(reachy)

    # while True:
    #     print("\n###X)Testing both arms ad vitam eternam")
    #     test_both_arms(reachy)

    # print("\n###8)Testing the look_at function")
    # test_head_look_at(reachy)

    print("\n###9)Testing the goto_cartesian_with_interpolation function")
    test_goto_cartesian_with_interpolation(reachy)

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


def multi_test():
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy.grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    print("Turning on...")

    reachy.turn_on()
    for joint in reachy.joints.values():
        joint.goal_position = 0

    reachy.l_arm.goto_joints([15, 10, 20, -50, 10, 10, 20], duration=3, interpolation_mode="linear")
    time.sleep(4)

    for joint in reachy.joints.values():
        joint.goal_position = 0
    time.sleep(1)

    reachy.head.rotate_to(0, 40, 0)

    reachy.disconnect()
    ReachySDK.clear()


if __name__ == "__main__":
    # head_test()
    main_test()
    # deco_test()
    # multi_test()
