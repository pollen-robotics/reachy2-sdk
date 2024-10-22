"""Reachy utils module.

This module contains various useful functions especially:
- angle conversion from/to degree/radian
- enum conversion to string
- matrix decomposition/recomposition
- pose matrix creation
- various grpc messages conversion
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
    """Convert a list of angles from degrees to radians.

    Args:
        angles_list: A list of angles in degrees to convert to radians.

    Returns:
        A list of angles converted to radians.
    """
    a = np.array(angles_list)
    a = np.deg2rad(a)

    return a.tolist()


def convert_to_degrees(angles_list: List[float]) -> Any:
    """Convert a list of angles from radians to degrees.

    Args:
        angles_list: A list of angles in radians to convert to degrees.

    Returns:
        A list of angles converted to degrees.
    """
    a = np.array(angles_list)
    a = np.rad2deg(a)

    return a.tolist()


def list_to_arm_position(positions: List[float], degrees: bool = True) -> ArmPosition:
    """Convert a list of joint positions to an ArmPosition message, considering whether the positions are in degrees or radians.

    Args:
        positions: A list of float values representing joint positions. The list should contain 7 values
            in the following order: [shoulder.pitch, shoulder.yaw, elbow.pitch, elbow.yaw, wrist.roll, wrist.pitch, wrist.yaw].
        degrees: A flag indicating whether the input joint positions are in degrees. If set to `True`,
            the input positions are in degrees. Defaults to `True`.

    Returns:
        An ArmPosition message containing the shoulder position, elbow position, and wrist position
        based on the input list of joint positions.
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
    """Convert an ArmPosition message to a list of joint positions, with an option to return the result in degrees.

    Args:
        arm_pos: An ArmPosition message containing shoulder, elbow, and wrist positions.
        degrees: Specifies whether the joint positions should be returned in degrees. If set to `True`,
            the positions are converted to degrees. Defaults to `True`.

    Returns:
        A list of joint positions based on the ArmPosition, returned in the following order:
        [shoulder.pitch, shoulder.yaw, elbow.yaw, elbow.pitch, wrist.roll, wrist.pitch, wrist.yaw].
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

    Args:
        euler_angles: An ExtEulerAngles object representing a 3D rotation message.
        degrees: Specifies whether the output should be in degrees. If set to `True`, the function
            converts the angles to degrees before returning the list. Defaults to `True`.

    Returns:
        A list of joint positions representing the Euler angles in the order [roll, pitch, yaw].
    """
    positions = [euler_angles.roll.value, euler_angles.pitch.value, euler_angles.yaw.value]

    if degrees:
        positions = convert_to_degrees(positions)

    return positions


def get_grpc_interpolation_mode(interpolation_mode: str) -> GoToInterpolation:
    """Convert a given interpolation mode string to a corresponding GoToInterpolation object.

    Args:
        interpolation_mode: A string representing the type of interpolation to be used. It can be either
            "minimum_jerk" or "linear".

    Returns:
        An instance of the GoToInterpolation class with the interpolation type set based on the input
        interpolation_mode string.

    Raises:
        ValueError: If the interpolation_mode is not "minimum_jerk" or "linear".
    """
    if interpolation_mode not in ["minimum_jerk", "linear"]:
        raise ValueError(f"Interpolation mode {interpolation_mode} not supported! Should be 'minimum_jerk' or 'linear'")

    if interpolation_mode == "minimum_jerk":
        interpolation_mode = InterpolationMode.MINIMUM_JERK
    else:
        interpolation_mode = InterpolationMode.LINEAR
    return GoToInterpolation(interpolation_type=interpolation_mode)


def get_interpolation_mode(interpolation_mode: InterpolationMode) -> str:
    """Convert an interpolation mode enum to a string representation.

    Args:
        interpolation_mode: The interpolation mode given as InterpolationMode. The supported interpolation
            modes are MINIMUM_JERK and LINEAR.

    Returns:
        A string representing the interpolation mode based on the input interpolation_mode. Returns
        "minimum_jerk" if the mode is InterpolationMode.MINIMUM_JERK, and "linear" if it is
        InterpolationMode.LINEAR.

    Raises:
        ValueError: If the interpolation_mode is not InterpolationMode.MINIMUM_JERK or InterpolationMode.LINEAR.
    """
    if interpolation_mode not in [InterpolationMode.MINIMUM_JERK, InterpolationMode.LINEAR]:
        raise ValueError(f"Interpolation mode {interpolation_mode} not supported! Should be 'minimum_jerk' or 'linear'")

    if interpolation_mode == InterpolationMode.MINIMUM_JERK:
        mode = "minimum_jerk"
    else:
        mode = "linear"
    return mode


def decompose_matrix(matrix: npt.NDArray[np.float64]) -> Tuple[Quaternion, npt.NDArray[np.float64]]:
    """Decompose a homogeneous 4x4 matrix into rotation (represented as a quaternion) and translation components.

    Args:
        matrix: A homogeneous 4x4 matrix represented as a NumPy array of type np.float64.

    Returns:
        A tuple containing a Quaternion representing the rotation component and a NumPy array
        representing the translation component of the input matrix.
    """
    rotation_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]

    # increase tolerance to avoid errors when checking if the matrix is a valid rotation matrix
    # See https://github.com/KieranWynn/pyquaternion/pull/44
    rotation = Quaternion(matrix=rotation_matrix, atol=1e-05, rtol=1e-05)
    return rotation, translation


def recompose_matrix(rotation: npt.NDArray[np.float64], translation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Recompose a homogeneous 4x4 matrix from rotation (quaternion) and translation components.

    Args:
        rotation: A 3x3 rotation matrix represented as a NumPy array of type np.float64.
        translation: A vector representing the displacement in space, containing the x, y, and z
            components of the translation vector.

    Returns:
        A homogeneous 4x4 matrix composed from the provided rotation and translation components.
    """
    matrix = np.eye(4)
    matrix[:3, :3] = rotation  # .as_matrix()
    matrix[:3, 3] = translation
    return matrix


def matrix_from_euler_angles(roll: float, pitch: float, yaw: float, degrees: bool = True) -> npt.NDArray[np.float64]:
    """Create a 4x4 homogeneous rotation matrix from roll, pitch, and yaw angles, with an option to input angles in degrees.

    Args:
        roll: The rotation angle around the x-axis in the Euler angles representation.
        pitch: The rotation angle around the y-axis in the Euler angles representation.
        yaw: The rotation angle around the z-axis in the Euler angles representation.
        degrees: Specifies whether the input angles (roll, pitch, yaw) are in degrees. If set to `True`,
            the input angles are expected to be in degrees. Defaults to `True`.

    Returns:
        A 4x4 homogeneous rotation matrix created from the input roll, pitch, and yaw angles.
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
    """Create a 4x4 pose matrix from a position vector and "roll, pitch, yaw" angles (rotation).

    Args:
        position: A list of size 3 representing the requested position of the end effector in the Reachy coordinate system.
        rotation: A list of size 3 representing the requested orientation of the end effector in the Reachy coordinate system.
            The rotation is given as intrinsic angles, executed in roll, pitch, yaw order.
        degrees: Specifies whether the input angles are in degrees. If set to `True`, the angles are interpreted as degrees.
            If set to `False`, they are interpreted as radians. Defaults to `True`.

    Returns:
        The constructed 4x4 pose matrix.

    Raises:
        TypeError: If `position` is not a list of floats or integers.
        TypeError: If `rotation` is not a list of floats or integers.
        ValueError: If the length of `position` is not 3.
        ValueError: If the length of `rotation` is not 3.
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


def quaternion_from_euler(roll: float, pitch: float, yaw: float, degrees: bool = True) -> Quaternion:
    """Convert Euler angles (intrinsic XYZ order) to a quaternion using the pyquaternion library.

    Args:
        roll (float): Rotation angle around the X-axis (roll), in degrees by default.
        pitch (float): Rotation angle around the Y-axis (pitch), in degrees by default.
        yaw (float): Rotation angle around the Z-axis (yaw), in degrees by default.
        degrees (bool): If True, the input angles are interpreted as degrees. If False, they are
            interpreted as radians. Defaults to True.

    Returns:
        Quaternion: The quaternion representing the combined rotation in 3D space.
    """
    if degrees:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    qx = Quaternion(axis=[1, 0, 0], angle=roll)
    qy = Quaternion(axis=[0, 1, 0], angle=pitch)
    qz = Quaternion(axis=[0, 0, 1], angle=yaw)

    quaternion = qx * qy * qz

    return quaternion


def rotate_in_self(frame: npt.NDArray[np.float64], rotation: List[float], degrees: bool = True) -> npt.NDArray[np.float64]:
    """Return a new homogeneous 4x4 pose matrix that is the input matrix rotated in itself.

    Args:
        frame: The input frame, given as a 4x4 homogeneous matrix.
        rotation: A list of size 3 representing the rotation to be applied. The rotation is given as intrinsic angles,
            executed in roll, pitch, yaw order.
        degrees: Specifies whether the input angles are in degrees. If set to `True`, the angles are interpreted as degrees.
            If set to `False`, they are interpreted as radians. Defaults to `True`.

    Returns:
        A new 4x4 homogeneous matrix after applying the specified rotation.
    """
    new_frame = frame.copy()

    toOrigin = np.eye(4)
    toOrigin[:3, :3] = new_frame[:3, :3]
    toOrigin[:3, 3] = new_frame[:3, 3]
    toOrigin = np.linalg.inv(toOrigin)

    new_frame = toOrigin @ new_frame
    new_frame = get_pose_matrix([0.0, 0.0, 0.0], rotation, degrees=degrees) @ new_frame
    new_frame = np.linalg.inv(toOrigin) @ new_frame

    return new_frame


def translate_in_self(frame: npt.NDArray[np.float64], translation: List[float]) -> npt.NDArray[np.float64]:
    """Return a new homogeneous 4x4 pose matrix that is the input frame translated along its own axes.

    Args:
        frame: The input frame, given as a 4x4 homogeneous matrix.
        translation: A list of size 3 representing the translation to be applied, given as [x, y, z].

    Returns:
        A new homogeneous 4x4 pose matrix after translating the input frame along its own axes.
    """
    new_frame = frame.copy()

    toOrigin = np.eye(4)
    toOrigin[:3, :3] = new_frame[:3, :3]
    toOrigin[:3, 3] = new_frame[:3, 3]
    toOrigin = np.linalg.inv(toOrigin)

    new_frame = toOrigin @ new_frame
    new_frame = get_pose_matrix(translation, [0, 0, 0]) @ new_frame
    new_frame = np.linalg.inv(toOrigin) @ new_frame

    return new_frame


def invert_affine_transformation_matrix(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Invert a 4x4 homogeneous transformation matrix.

    The function computes the inverse by transposing the rotation component and adjusting the translation component.

    Args:
        matrix: A 4x4 NumPy array representing a homogeneous transformation matrix.

    Returns:
        A new 4x4 homogeneous matrix that is the inverse of the input matrix.

    Raises:
        ValueError: If the input matrix is not 4x4.
    """
    if matrix.shape != (4, 4):
        raise ValueError("matrix should be 4x4")

    inv_matrix = np.eye(4)
    inv_matrix[:3, :3] = matrix[:3, :3].T
    inv_matrix[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inv_matrix


def get_normal_vector(vector: npt.NDArray[np.float64], arc_direction: str) -> Optional[npt.NDArray[np.float64]]:
    """Calculate a normal vector to a given vector based on a specified direction.

    Args:
        vector: A vector [x, y, z] in 3D space.
        arc_direction: The desired direction for the normal vector. It can be one of the following options:
            'above', 'below', 'front', 'back', 'right', or 'left'.

    Returns:
        The normal vector [x, y, z] to the given vector in the specified direction. Returns `None` if the
        normal vector cannot be computed or if the vector is in the requested arc_direction.

    Raises:
        ValueError: If the arc_direction is not one of 'above', 'below', 'front', 'back', 'right', or 'left'.
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
