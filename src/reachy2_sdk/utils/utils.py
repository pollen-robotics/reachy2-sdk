"""Reachy utils module.

This module contains various useful functions especially:
- angle conversion from/to degree/radian
- enum conversion to string
"""

from collections import namedtuple
from typing import Any, List, Tuple

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
    """
    This function converts a list of angles from degrees to radians.

    Args:
      my_list (List[float]): A list of angles in degrees that you want to convert to radians.

    Returns:
      The function `convert_to_radians` takes a list of angles in degrees as input, converts them to
    radians using NumPy's `deg2rad` function, and then returns the converted angles as a list.
    """
    a = np.array(my_list)
    a = np.deg2rad(a)

    return a.tolist()


def convert_to_degrees(my_list: List[float]) -> Any:
    """
    This function converts a list of angles from radians to degrees.

    Args:
      my_list (List[float]): The `my_list` parameter is a list of angles in radians that you want to
    convert to degrees using the `convert_to_degrees` function.

    Returns:
      The function `convert_to_degrees` takes a list of angles in radians as input, converts them to
    degrees using NumPy's `rad2deg` function, and returns the converted angles as a list.
    """
    a = np.array(my_list)
    a = np.rad2deg(a)

    return a.tolist()


def list_to_arm_position(positions: List[float], degrees: bool = True) -> ArmPosition:
    """
    The function `list_to_arm_position` converts a list of joint positions to an ArmPosition message,
    considering whether the positions are in degrees or radians.

    Args:
      positions (List[float]): The `positions` parameter is a list of float values representing joint
    positions. The list should contain values for shoulder and elbow joint positions, as well as roll,
    pitch, and yaw values for the wrist position. The length of the list should be 7, with the values in
    the following order:
      degrees (bool): The `degrees` parameter in the `list_to_arm_position` function is a boolean flag
    that indicates whether the input joint positions are in degrees or radians. If `degrees` is set to
    `True`, it means that the input joint positions are in degrees, and they will be converted to
    radians before. Defaults to True

    Returns:
      An ArmPosition message is being returned, which contains the shoulder position, elbow position,
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
    The function `arm_position_to_list` converts an ArmPosition message to a list of joint positions,
    with an option to return the result in degrees.

    Args:
      arm_pos (ArmPosition): ArmPosition message containing shoulder, elbow, and wrist positions.
      degrees (bool): The `degrees` parameter in the `arm_position_to_list` function is a boolean
    parameter that specifies whether the joint positions should be returned in degrees or not. By
    default, it is set to `True`, meaning that the joint positions will be returned in degrees unless
    specified otherwise. Defaults to True

    Returns:
      The function `arm_position_to_list` returns a list of joint positions based on the ArmPosition
    message provided as input. The joint positions are extracted from the shoulder, elbow, and wrist
    positions within the ArmPosition message. If the `degrees` parameter is set to True (which is the
    default), the joint positions are converted to degrees before being returned.
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
    This function converts ExtEulerAngles 3D rotation message to a list of joint positions, with an
    option to return the result in degrees.

    Args:
      euler_angles (ExtEulerAngles): An `ExtEulerAngles` object representing a 3D rotation message. It
    likely contains the following attributes:
      degrees (bool): The `degrees` parameter in the function `ext_euler_angles_to_list` is a boolean
    parameter that specifies whether the output should be in degrees or not. If `degrees` is set to
    `True`, the function will convert the Euler angles to degrees before returning the list of joint
    positions. If. Defaults to True

    Returns:
      A list of joint positions representing the Euler angles in degrees.
    """
    positions = [euler_angles.roll.value, euler_angles.pitch.value, euler_angles.yaw.value]

    if degrees:
        positions = convert_to_degrees(positions)

    return positions


def get_grpc_interpolation_mode(interpolation_mode: str) -> GoToInterpolation:
    """
    The function `get_grpc_interpolation_mode` converts a given interpolation mode string to a
    corresponding GoToInterpolation object.

    Args:
      interpolation_mode (str): The `interpolation_mode` parameter is a string that represents the type
    of interpolation to be used. It can have two possible values: "minimum_jerk" or "linear". The
    function `get_grpc_interpolation_mode` takes this string as input and converts it to the
    corresponding `GoTo

    Returns:
      An instance of the `GoToInterpolation` class with the interpolation type set based on the input
    `interpolation_mode` string.
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
    The function `get_interpolation_mode` converts an interpolation mode enum to a string
    representation.

    Args:
      interpolation_mode (InterpolationMode): `interpolation_mode` is a parameter of type
    `InterpolationMode` that is passed to the `get_interpolation_mode` function. The function converts
    the interpolation mode given as `InterpolationMode` to a string representation. The supported
    interpolation modes are `MINIMUM_JERK` and `

    Returns:
      The function `get_interpolation_mode` returns a string representing the interpolation mode based
    on the input `interpolation_mode`. If the `interpolation_mode` is `InterpolationMode.MINIMUM_JERK`,
    it returns "minimum_jerk". If the `interpolation_mode` is `InterpolationMode.LINEAR`, it returns
    "linear".
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
    This Python function decomposes a homogeneous 4x4 matrix into rotation (represented as a quaternion)
    and translation components.

    Args:
      matrix (npt.NDArray[np.float64]): The `matrix` parameter in the `decompose_matrix` function is
    expected to be a homogeneous 4x4 matrix represented as a NumPy array of type `np.float64`. This
    matrix is typically used in computer graphics and robotics to represent transformations in 3D space,
    including rotation and translation

    Returns:
      A tuple containing a Quaternion representing the rotation component and a NumPy array representing
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
    This function recomposes a homogeneous 4x4 matrix from rotation (quaternion) and translation
    components.

    Args:
      rotation (npt.NDArray[np.float64]): The rotation parameter is expected to be a 3x3 rotation matrix
    represented as a NumPy array of type np.float64.
      translation (npt.NDArray[np.float64]): Translation is a vector that represents the displacement of
    an object in space. In the context of the `recompose_matrix` function, the translation parameter is
    a 1D NumPy array of shape (3,) that contains the x, y, and z components of the translation vector.
    These components specify

    Returns:
      The function `recompose_matrix` returns a homogeneous 4x4 matrix composed from the rotation
    (quaternion) and translation components provided as input.
    """
    matrix = np.eye(4)
    matrix[:3, :3] = rotation  # .as_matrix()
    matrix[:3, 3] = translation
    return matrix


def matrix_from_euler_angles(roll: float, pitch: float, yaw: float, degrees: bool = True) -> npt.NDArray[np.float64]:
    """
    This Python function creates a 4x4 homogeneous rotation matrix from roll, pitch, and yaw angles,
    with an option to input angles in degrees.

    Args:
      roll (float): The `roll` parameter represents the rotation angle around the x-axis in the Euler
    angles representation.
      pitch (float): The `pitch` parameter in the `matrix_from_euler_angles` function represents the
    rotation around the y-axis in the Euler angles representation. It is the angle of rotation about the
    x-axis in a 3D coordinate system.
      yaw (float): The yaw angle represents the rotation around the z-axis in a 3D coordinate system. It
    is also known as the heading angle.
      degrees (bool): The `degrees` parameter in the `matrix_from_euler_angles` function is a boolean
    flag that specifies whether the input angles (`roll`, `pitch`, `yaw`) are in degrees or radians. If
    `degrees` is set to `True`, the function converts the input angles from degrees to radians. Defaults
    to True

    Returns:
      The function `matrix_from_euler_angles` returns a 4x4 homogeneous rotation matrix created from the
    input roll, pitch, and yaw angles.
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


def invert_affine_transformation_matrix(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    This Python function inverts a 4x4 homogeneous transformation matrix by computing its transpose and
    adjusting the translation component,"""
    """with matrix M = [R t] , returns M^-1 = [R.T -R.T * t]"""
    """                [0 1]                  [0          1]"""
    """
    Args:
      matrix (npt.NDArray[np.float64]): The `matrix` parameter in the
    `invert_affine_transformation_matrix` function is expected to be a 4x4 NumPy array representing a
    homogeneous transformation matrix. The matrix should have the following structure:

    Returns:
      The function `invert_affine_transformation_matrix` returns the inverse of a given homogeneous
    transformation matrix. The inverse matrix is calculated as follows:
    """
    if matrix.shape != (4, 4):
        raise ValueError("matrix should be 4x4")

    inv_matrix = np.eye(4)
    inv_matrix[:3, :3] = matrix[:3, :3].T
    inv_matrix[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inv_matrix
