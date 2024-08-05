import numpy as np
import pytest
from google.protobuf.wrappers_pb2 import FloatValue
from reachy2_sdk_api.kinematics_pb2 import ExtEulerAngles

from reachy2_sdk.utils.utils import (
    arm_position_to_list,
    convert_to_degrees,
    convert_to_radians,
    ext_euler_angles_to_list,
    get_grpc_interpolation_mode,
    get_interpolation_mode,
    list_to_arm_position,
)


@pytest.mark.offline
def test_deg_rad() -> None:
    degs_ref = [0.0, 360.0, 180.0, 10.0, 20.0]
    rads_ref = [0.0, 6.28319, 3.14159, 0.174533, 0.349066]
    rads = convert_to_radians(degs_ref)

    # assert np.array_equal(rads_ref, rads)
    assert np.allclose(rads_ref, rads, atol=1e-03)

    degs = convert_to_degrees(rads_ref)

    assert np.allclose(degs_ref, degs, atol=1e-03)


@pytest.mark.offline
def test_arm_position_to_list() -> None:
    arm_position_float_ref = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    arm_position = list_to_arm_position(arm_position_float_ref, True)

    arm_position_float = arm_position_to_list(arm_position, True)

    assert np.allclose(arm_position_float_ref, arm_position_float, atol=1e-03)

    """
    # Todo: this should be equivalent

    arm_position_float_ref = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    arm_position = list_to_arm_position(arm_position_float_ref)

    arm_position_float = arm_position_to_list(arm_position)

    assert np.array_equal(arm_position_float_ref, arm_position_float)
    """


@pytest.mark.offline
def test_ext_euler_angles_list() -> None:
    rpy_rad = [1, 2, 3]
    ext_euler = ExtEulerAngles(
        roll=FloatValue(value=rpy_rad[0]), pitch=FloatValue(value=rpy_rad[1]), yaw=FloatValue(value=rpy_rad[2])
    )
    euler_list = ext_euler_angles_to_list(ext_euler, True)
    rpy_deg = convert_to_degrees(rpy_rad)

    assert np.array_equal(euler_list, rpy_deg)


@pytest.mark.offline
def test_interpolation_modes() -> None:
    str = "linear"
    mode_grpc = get_grpc_interpolation_mode(str)
    mode_str = get_interpolation_mode(mode_grpc.interpolation_type)

    assert mode_str == str

    str = "minimum_jerk"
    mode_grpc = get_grpc_interpolation_mode(str)
    mode_str = get_interpolation_mode(mode_grpc.interpolation_type)

    assert mode_str == str

    with pytest.raises(ValueError):
        get_grpc_interpolation_mode("dummy")

    with pytest.raises(ValueError):
        get_interpolation_mode("dummy")
