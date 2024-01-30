"""Reachy utils module.

This module contains various useful functions especially:
- angle conversion from/to degree/radian
- enum conversion to string
"""

from typing import Any, List

import numpy as np
from google.protobuf.wrappers_pb2 import FloatValue
from reachy2_sdk_api.arm_pb2 import ArmPosition
from reachy2_sdk_api.goto_pb2 import GoToInterpolation, InterpolationMode
from reachy2_sdk_api.kinematics_pb2 import ExtEulerAngles, Rotation3d
from reachy2_sdk_api.orbita2d_pb2 import Pose2d


def convert_to_radians(my_list: List[float]) -> Any:
    """Convert a list of angles from degrees to radians."""
    a = np.array(my_list)
    a = np.deg2rad(a)

    a = np.round(a, 3)
    return a.tolist()


def convert_to_degrees(my_list: List[float]) -> Any:
    """Convert a list of angles from radians to degrees."""
    a = np.array(my_list)
    a = np.rad2deg(a)

    a = np.round(a, 2)
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
