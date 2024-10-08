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


def convert_to_radians(angles_list: List[float]) -> Any:
    """
    Converts a list of angles from degrees to radians.

    Args:
      angles_list (List[float]): a list of angles in degrees that you want to convert to radians.

    Returns:
      a list of angles in radians.
    """
    a = np.array(angles_list)
    a = np.deg2rad(a)

    return a.tolist()


def convert_to_degrees(angles_list: List[float]) -> Any:
    """
    Converts a list of angles from radians to degrees.

    Args:
      angles_list (List[float]): a list of angles in radians that you want to convert to degrees.

    Returns:
      a list of angles in degrees.
    """
    a = np.array(angles_list)
    a = np.rad2deg(a)

    return a.tolist()


def list_to_arm_position(positions: List[float], degrees: bool = True) -> ArmPosition:
    """
    Converts a list of joint positions to an ArmPosition message, considering whether the positions are in degrees or radians.

    Args:
      positions (List[float]): a list of float values representing joint positions. The length of the list should be 7, with
    the values in the following order:
    [shoulder.pitch, shoulder.yaw, elbow.pitch, elbow.yaw, wrist.roll, wrist.pitch, wrist.yaw]
      degrees (bool): a boolean flag that indicates whether the input joint positions are in degrees or radians. If `degrees`
    is set to `True`, it means that the input joint positions are in degrees. Defaults to True.

    Returns:
      a ArmPosition message is being returned, which contains the shoulder position, elbow position,
    and wrist position based on the input list of joint positions.
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
    """
    Converts an ArmPosition message to a list of joint positions, with an option to return the result in degrees.

    Args:
      arm_pos (ArmPosition): an ArmPosition message containing shoulder, elbow, and wrist positions.
      degrees (bool): a boolean parameter that specifies whether the joint positions should be returned in degrees or not.
    Defaults to True

    Returns:
      a list of joint positions based on the ArmPosition, returned in the following order:
    [shoulder.pitch, shoulder.yaw, elbow.pitch, elbow.yaw, wrist.roll, wrist.pitch, wrist.yaw]
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
    """
    Converts ExtEulerAngles 3D rotation message to a list of joint positions, with an option to return the result in degrees.

    Args:
      euler_angles (ExtEulerAngles): an `ExtEulerAngles` object representing a 3D rotation message.
      degrees (bool): a boolean parameter that specifies whether the output should be in degrees or not. If `degrees` is set to
    `True`, the function will convert the angles to degrees before returning the list of joint positions. Defaults to True.

    Returns:
      a list of joint positions representing the Euler angles in order [roll, pitch, yaw].
    """
    positions = [euler_angles.roll.value, euler_angles.pitch.value, euler_angles.yaw.value]

    if degrees:
        positions = convert_to_degrees(positions)

    return positions


def get_grpc_interpolation_mode(interpolation_mode: str) -> GoToInterpolation:
    """
    Converts a given interpolation mode string to a corresponding GoToInterpolation object.

    Args:
      interpolation_mode (str): a string that represents the type of interpolation to be used. It can have two possible values:
    "minimum_jerk" or "linear".

    Returns:
      an instance of the `GoToInterpolation` class with the interpolation type set based on the input `interpolation_mode`
    string.
    """
    if interpolation_mode not in ["minimum_jerk", "linear"]:
        raise ValueError(f"Interpolation mode {interpolation_mode} not supported! Should be 'minimum_jerk' or 'linear'")

    if interpolation_mode == "minimum_jerk":
        interpolation_mode = InterpolationMode.MINIMUM_JERK
    else:
        interpolation_mode = InterpolationMode.LINEAR
    return GoToInterpolation(interpolation_type=interpolation_mode)


def get_interpolation_mode(interpolation_mode: InterpolationMode) -> str:
    """
    Converts an interpolation mode enum to a string representation.

    Args:
      interpolation_mode (InterpolationMode): interpolation mode given as `InterpolationMode`. The supported
    interpolation modes are `MINIMUM_JERK` and `LINEAR`

    Returns:
      a string representing the interpolation mode based on the input `interpolation_mode`. If the `interpolation_mode` is
    `InterpolationMode.MINIMUM_JERK`, it returns "minimum_jerk". If the `interpolation_mode` is `InterpolationMode.LINEAR`,
    it returns "linear".
    """
    if interpolation_mode not in [InterpolationMode.MINIMUM_JERK, InterpolationMode.LINEAR]:
        raise ValueError(f"Interpolation mode {interpolation_mode} not supported! Should be 'minimum_jerk' or 'linear'")

    if interpolation_mode == InterpolationMode.MINIMUM_JERK:
        mode = "minimum_jerk"
    else:
        mode = "linear"
    return mode


def decompose_matrix(matrix: npt.NDArray[np.float64]) -> Tuple[Quaternion, npt.NDArray[np.float64]]:
    """
    Decomposes a homogeneous 4x4 matrix into rotation (represented as a quaternion) and translation components.

    Args:
      matrix (npt.NDArray[np.float64]): a homogeneous 4x4 matrix represented as a NumPy array of type `np.float64`.

    Returns:
      a tuple containing a Quaternion representing the rotation component and a NumPy array representing
    the translation component of the input homogeneous 4x4 matrix.
    """
    rotation_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]

    # increase tolerance to avoid errors when checking if the matrix is a valid rotation matrix
    # See https://github.com/KieranWynn/pyquaternion/pull/44
    rotation = Quaternion(matrix=rotation_matrix, atol=1e-05, rtol=1e-05)
    return rotation, translation


def recompose_matrix(rotation: npt.NDArray[np.float64], translation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Recomposes a homogeneous 4x4 matrix from rotation (quaternion) and translation components.

    Args:
      rotation (npt.NDArray[np.float64]): a 3x3 rotation matrix represented as a NumPy array of type np.float64.
      translation (npt.NDArray[np.float64]): a vector that represents the displacement of an object in space,
    that contains the x, y, and z components of the translation vector.

    Returns:
      a homogeneous 4x4 matrix composed from the rotation (quaternion) and translation components provided as input.
    """
    matrix = np.eye(4)
    matrix[:3, :3] = rotation  # .as_matrix()
    matrix[:3, 3] = translation
    return matrix


def matrix_from_euler_angles(roll: float, pitch: float, yaw: float, degrees: bool = True) -> npt.NDArray[np.float64]:
    """
    Creates a 4x4 homogeneous rotation matrix from roll, pitch, and yaw angles, with an option to input angles in degrees.

    Args:
      roll (float): rotation angle around the x-axis in the Euler angles representation.
      pitch (float): rotation angle around the y-axis in the Euler angles representation.
      yaw (float): rotation angle around the z-axis in the Euler angles representation.
      degrees (bool): a boolean flag that specifies whether the input angles (`roll`, `pitch`, `yaw`) are in degrees or radians.
    If `degrees` is set to `True`, the input angles are expected to be given in degrees. Defaults to True.

    Returns:
      a 4x4 homogeneous rotation matrix created from the input roll, pitch, and yaw angles.
    """
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
    """
    Creates the 4x4 pose matrix from a position vector and \"roll pitch yaw\" angles (rotation).

    Args :
      position (List[float]): a list of size 3. It is the requested position of the end effector in Reachy coordinate system
      rotation (List[float]): a list of size 3. It it the requested orientation of the end effector in Reachy coordinate system.
    Rotation is given as intrinsic angles, that are executed in roll, pitch, yaw order.
      degrees (bool): a boolean flag that specifies whether the input angles are in degrees or radians. `True` if angles are
    provided in degrees, `False` if they are in radians. Defaults to `True`.

    Returns :
        the constructed 4x4 pose matrix.
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
    Returns a new homogeneous 4x4 pose matrix that is the input matrix rotated in itself.

    Args :
      _frame (npt.NDArray[np.float64]): the input frame, as a 4x4 homogeneous matrix.
      rotation (List[float]): a list of size 3. It is the rotation to be applied given as intrinsic angles, that are executed
    in roll, pitch, yaw order.
      degrees (bool): a boolean flag that specifies whether the input angles are in degrees or radians. `True` if angles are
    provided in degrees, `False` if they are in radians. Defaults to `True`.

    Returns:
      a new 4x4 homogeneous matrix.
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
     Returns a new homogeneous 4x4 pose matrix that is the input frame translated along its own axes.

    Args :
      _frame (npt.NDArray[np.float64]): the input frame
      translation : a list of size 3. It is the trasnlation to be applied given as [x, y, z].

    Returns:
      a new homogeneous 4x4 pose matrix that is the input frame translated in itself.
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
    """
    Inverts a 4x4 homogeneous transformation matrix by computing its transpose and adjusting the translation component,"""
    """with matrix M = [R t] , returns M^-1 = [R.T -R.T * t]"""
    """                [0 1]                  [0          1]"""
    """

    Args:
      matrix (npt.NDArray[np.float64]): a 4x4 NumPy array representing a homogeneous transformation matrix.

    Returns:
      a new 4x4 homogeneous matrix, that is the inverse of the input matrix.
    """
    if matrix.shape != (4, 4):
        raise ValueError("matrix should be 4x4")

    inv_matrix = np.eye(4)
    inv_matrix[:3, :3] = matrix[:3, :3].T
    inv_matrix[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inv_matrix


def get_normal_vector(vector: npt.NDArray[np.float64], arc_direction: str) -> Optional[npt.NDArray[np.float64]]:
    """
    Calculates a normal vector to a given vector based on a specified direction.

    Args:
      vector (npt.NDArray[np.float64]): a vector [x, y, z] in 3D space.
      arc_direction (str): the desired direction for the normal vector. It can be one of the following options: 'above',
    'below', 'front', 'back', 'right', or 'left'.

    Returns:
      the normal vector [x, y, z] to the given vector in the specified direction.
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
