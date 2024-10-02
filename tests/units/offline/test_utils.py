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
    get_pose_matrix,
    list_to_arm_position,
    matrix_from_euler_angles,
)


@pytest.mark.offline
def test_deg_rad() -> None:
    degs_ref = [0.0, 360.0, 180.0, 10.0, 20.0]
    rads_ref = [0.0, 6.28319, 3.14159, 0.174533, 0.349066]
    rads = convert_to_radians(degs_ref)

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


@pytest.mark.offline
def test_matrix_from_euler_angles() -> None:
    A = matrix_from_euler_angles(20, 45, 30)
    scipy_result_A = np.array(
        [[0.61237244, -0.2604026, 0.74645193], [0.35355339, 0.93472006, 0.03603338], [-0.70710678, 0.24184476, 0.66446302]]
    )
    expected_A = np.eye(4)
    expected_A[:3, :3] = scipy_result_A
    np.array_equal(expected_A, A)

    B = matrix_from_euler_angles(40, -50, -15)
    scipy_result_B = np.array(
        [[0.62088515, -0.27735873, -0.73319422], [-0.16636568, 0.86738561, -0.4690039], [0.76604444, 0.41317591, 0.49240388]]
    )
    expected_B = np.eye(4)
    expected_B[:3, :3] = scipy_result_B
    np.array_equal(expected_B, B)

    C = matrix_from_euler_angles(-100, 40, -7)
    scipy_result_C = np.array(
        [[0.76033446, -0.64946616, 0.00923097], [-0.09335733, -0.09520783, 0.99107007], [-0.64278761, -0.75440651, -0.13302222]]
    )
    expected_C = np.eye(4)
    expected_C[:3, :3] = scipy_result_C
    np.array_equal(expected_C, C)


@pytest.mark.offline
def test_get_pose_matrix() -> None:
    A = np.array([[0.0, 0.0, -1.0, 0.5], [0.0, 1.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    pose_A = get_pose_matrix([0.5, 0.1, 0], [0, -90, 0])
    assert np.allclose(A, pose_A, atol=1e-03)

    B = np.array([[0.0, 1.0, -0.0, 0.2], [0.0, 0.0, 1.0, 0.4], [1.0, -0.0, 0.0, -0.2], [0.0, 0.0, 0.0, 1.0]])
    pose_B = get_pose_matrix([0.2, 0.4, -0.2], [-90, -90, 0])
    assert np.allclose(B, pose_B, atol=1e-03)

    C = np.array([[0.262, -0.808, -0.528, 0.5], [0.72, 0.528, -0.451, -0.8], [0.643, -0.262, 0.72, 0.0], [0.0, 0.0, 0.0, 1.0]])
    pose_C = get_pose_matrix([0.5, -0.8, 0], [-20, -40, 70])
    assert np.allclose(C, pose_C, atol=1e-03)

    with pytest.raises(TypeError):
        get_pose_matrix([1, 2, "coucou"], [1, 2, 3])
    with pytest.raises(TypeError):
        get_pose_matrix([1, 2, 3], [1, 2, "coucou"])
    with pytest.raises(TypeError):
        get_pose_matrix([0.1, 0.2, 0.3], -90)
    with pytest.raises(ValueError):
        get_pose_matrix([0.1, 0.2, 0.1, 0.1], [0, -90, 0])
    with pytest.raises(ValueError):
        get_pose_matrix([0.1, 0.2, 0.1], [-20, -90, -50, 10])
