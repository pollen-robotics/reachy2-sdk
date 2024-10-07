"""Reachy utils module.

This module contains various useful functions especially:
- angle conversion from/to degree/radian
- enum conversion to string
"""

from collections import namedtuple
from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from google.protobuf.wrappers_pb2 import FloatValue
from pyquaternion import Quaternion
from reachy2_sdk_api.arm_pb2 import ArmPosition
from reachy2_sdk_api.goto_pb2 import GoToInterpolation, InterpolationMode
from reachy2_sdk_api.kinematics_pb2 import ExtEulerAngles, Rotation3d
from reachy2_sdk_api.orbita2d_pb2 import Pose2d

SimplifiedRequest = namedtuple("SimplifiedRequest", ["part", "goal_positions", "duration", "mode"])
"""Named tuple for easy access to request variables"""


def convert_to_radians(my_list: List[float]) -> Any:
    """Convert a list of angles from degrees to radians."""
    a = np.array(my_list)
    a = np.deg2rad(a)

    return a.tolist()


def convert_to_degrees(my_list: List[float]) -> Any:
    """Convert a list of angles from radians to degrees."""
    a = np.array(my_list)
    a = np.rad2deg(a)

    return a.tolist()


def list_to_arm_position(positions: List[float], degrees: bool = True) -> ArmPosition:
    """Convert a list of joint positions to an ArmPosition message.

    This is used to send a joint position to the arm's gRPC server and to compute the forward
    and inverse kinematics.
    """
    if degrees:
        positions = convert_to_radians(positions)
    arm_pos = ArmPosition(
        shoulder_position=Pose2d(
            axis_1=FloatValue(value=positions[0]),
            axis_2=FloatValue(value=positions[1]),
        ),
        elbow_position=Pose2d(
            axis_1=FloatValue(value=positions[2]),
            axis_2=FloatValue(value=positions[3]),
        ),
        wrist_position=Rotation3d(
            rpy=ExtEulerAngles(
                roll=FloatValue(value=positions[4]),
                pitch=FloatValue(value=positions[5]),
                yaw=FloatValue(value=positions[6]),
            )
        ),
    )

    return arm_pos


def arm_position_to_list(arm_pos: ArmPosition, degrees: bool = True) -> List[float]:
    """Convert an ArmPosition message to a list of joint positions.

    It is used to convert the result of the inverse kinematics.
    By default, it will return the result in degrees.
    """
    positions = []

    for _, value in arm_pos.shoulder_position.ListFields():
        positions.append(value.value)
    for _, value in arm_pos.elbow_position.ListFields():
        positions.append(value.value)
    for _, value in arm_pos.wrist_position.rpy.ListFields():
        positions.append(value.value)

    if degrees:
        positions = convert_to_degrees(positions)

    return positions


def ext_euler_angles_to_list(euler_angles: ExtEulerAngles, degrees: bool = True) -> List[float]:
    """Convert an ExtEulerAngles 3D rotation message to a list of joint positions.

    By default, it will return the result in degrees.
    """
    positions = [euler_angles.roll.value, euler_angles.pitch.value, euler_angles.yaw.value]

    if degrees:
        positions = convert_to_degrees(positions)

    return positions


def get_grpc_interpolation_mode(interpolation_mode: str) -> GoToInterpolation:
    """Convert interpolation mode given as string to GoToInterpolation"""
    if interpolation_mode not in ["minimum_jerk", "linear"]:
        raise ValueError(f"Interpolation mode {interpolation_mode} not supported! Should be 'minimum_jerk' or 'linear'")

    if interpolation_mode == "minimum_jerk":
        interpolation_mode = InterpolationMode.MINIMUM_JERK
    else:
        interpolation_mode = InterpolationMode.LINEAR
    return GoToInterpolation(interpolation_type=interpolation_mode)


def get_interpolation_mode(interpolation_mode: InterpolationMode) -> str:
    """Convert interpolation mode given as GoToInterpolation to string"""
    if interpolation_mode not in [InterpolationMode.MINIMUM_JERK, InterpolationMode.LINEAR]:
        raise ValueError(f"Interpolation mode {interpolation_mode} not supported! Should be 'minimum_jerk' or 'linear'")

    if interpolation_mode == InterpolationMode.MINIMUM_JERK:
        mode = "minimum_jerk"
    else:
        mode = "linear"
    return mode


def decompose_matrix(matrix: npt.NDArray[np.float64]) -> Tuple[Quaternion, npt.NDArray[np.float64]]:
    """Decompose a homogeneous 4x4 matrix into rotation (quaternion) and translation components."""
    rotation_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]

    # increase tolerance to avoid errors when checking if the matrix is a valid rotation matrix
    # See https://github.com/KieranWynn/pyquaternion/pull/44
    rotation = Quaternion(matrix=rotation_matrix, atol=1e-05, rtol=1e-05)
    return rotation, translation


def recompose_matrix(rotation: npt.NDArray[np.float64], translation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Recompose a homogeneous 4x4 matrix from rotation (quaternion) and translation components."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation  # .as_matrix()
    matrix[:3, 3] = translation
    return matrix


def matrix_from_euler_angles(roll: float, pitch: float, yaw: float, degrees: bool = True) -> npt.NDArray[np.float64]:
    """Create a homogeneous rotation matrix 4x4 from roll, pitch, yaw angles."""
    if degrees:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    R_x = np.array(
        [[1, 0, 0, 0], [0, np.cos(roll), -np.sin(roll), 0], [0, np.sin(roll), np.cos(roll), 0], [0, 0, 0, 1]], dtype=np.float64
    )

    R_y = np.array(
        [[np.cos(pitch), 0, np.sin(pitch), 0], [0, 1, 0, 0], [-np.sin(pitch), 0, np.cos(pitch), 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )

    R_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0, 0], [np.sin(yaw), np.cos(yaw), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
    )

    rotation_matrix = R_z @ R_y @ R_x
    return rotation_matrix


def get_pose_matrix(position: List[float], rotation: List[float], degrees: bool = True) -> npt.NDArray[np.float64]:
    """Creates the 4x4 pose matrix from a position vector and \"roll pitch yaw\" angles (rotation).
    Arguments :
        position : a list of size 3. It is the requested position of the end effector in the robot coordinate system
        rotation : a list of size 3. It it the requested orientation of the end effector in the robot coordinate system.
                   Rotation is given as intrinsic angles, that are executed in roll, pitch, yaw order.
        degrees  : True if angles are provided in degrees, False if they are in radians.
    Returns :
        pose : the constructed pose matrix. This is a 4x4 numpy array
    """
    if not (isinstance(position, np.ndarray) or isinstance(position, list)) or not all(
        isinstance(pos, float | int) for pos in position
    ):
        raise TypeError(f"position should be a list of float, got {position}")
    if not (isinstance(rotation, np.ndarray) or isinstance(rotation, list)) or not all(
        isinstance(rot, float | int) for rot in rotation
    ):
        raise TypeError(f"rotation should be a list of float, got {rotation}")
    if len(position) != 3:
        raise ValueError("position should be a list of len 3")
    if len(rotation) != 3:
        raise ValueError("rotation should be a list of len 3")

    pose = matrix_from_euler_angles(rotation[0], rotation[1], rotation[2], degrees=degrees)
    pose[:3, 3] = np.array(position)
    return pose


def rotate_in_self(_frame: npt.NDArray[np.float64], rotation: List[float], degrees: bool = True) -> npt.NDArray[np.float64]:
    """
    Returns a new frame that is the input frame rotated in itself.
    Arguments :
        _frame   : the input frame
        rotation : the rotation to be applied [x, y, z]
        degrees  : are the angles of the rotation in degrees or radians ?

    """
    frame = _frame.copy()

    toOrigin = np.eye(4)
    toOrigin[:3, :3] = frame[:3, :3]
    toOrigin[:3, 3] = frame[:3, 3]
    toOrigin = np.linalg.inv(toOrigin)

    frame = toOrigin @ frame
    frame = get_pose_matrix([0.0, 0.0, 0.0], rotation, degrees=degrees) @ frame
    frame = np.linalg.inv(toOrigin) @ frame

    return frame


def translate_in_self(_frame: npt.NDArray[np.float64], translation: List[float]) -> npt.NDArray[np.float64]:
    """
    Returns a new frame that is the input frame translated along its own axes
    Arguments :
        _frame      : the input frame
        translation : the translation to be applied
    """
    frame = _frame.copy()

    toOrigin = np.eye(4)
    toOrigin[:3, :3] = frame[:3, :3]
    toOrigin[:3, 3] = frame[:3, 3]
    toOrigin = np.linalg.inv(toOrigin)

    frame = toOrigin @ frame
    frame = get_pose_matrix(translation, [0, 0, 0]) @ frame
    frame = np.linalg.inv(toOrigin) @ frame

    return frame


def invert_affine_transformation_matrix(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Invert an homegeneous transformation matrix"""
    """with matrix M = [R t] , returns M^-1 = [R.T -R.T * t]"""
    """                [0 1]                  [0          1]"""
    if matrix.shape != (4, 4):
        raise ValueError("matrix should be 4x4")

    inv_matrix = np.eye(4)
    inv_matrix[:3, :3] = matrix[:3, :3].T
    inv_matrix[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inv_matrix


def get_normal_vector(vector: npt.NDArray[np.float64], arc_direction: str) -> Optional[npt.NDArray[np.float64]]:
    """Get a normal vector to the given vector in the desired direction.

    direction can be: 'above', 'below', 'front', 'back', 'right' or 'left'.
    """
    match arc_direction:
        case "above":
            if abs(vector[0]) < 0.001 and abs(vector[1]) < 0.001:
                return None
            normal = np.cross(vector, [0, 0, -1])
        case "below":
            if abs(vector[0]) < 0.001 and abs(vector[1]) < 0.001:
                return None
            normal = np.cross(vector, [0, 0, 1])
        case "left":
            if abs(vector[0]) < 0.001 and abs(vector[2]) < 0.001:
                return None
            normal = np.cross(vector, [0, -1, 0])
        case "right":
            if abs(vector[0]) < 0.001 and abs(vector[2]) < 0.001:
                return None
            normal = np.cross(vector, [0, 1, 0])
        case "front":
            if abs(vector[1]) < 0.001 and abs(vector[2]) < 0.001:
                return None
            normal = np.cross(vector, [-1, 0, 0])
        case "back":
            if abs(vector[1]) < 0.001 and abs(vector[2]) < 0.001:
                return None
            normal = np.cross(vector, [1, 0, 0])
        case _:
            raise ValueError(
                f"arc_direction '{arc_direction}' not supported! Should be one of: "
                "'above', 'below', 'front', 'back', 'right' or 'left'"
            )

    if np.linalg.norm(normal) == 0:
        # Return None if the vector is in the requested arc_direction
        return None

    normal = normal / np.linalg.norm(normal)
    return normal
