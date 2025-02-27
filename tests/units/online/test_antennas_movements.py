import time

import numpy as np
import pytest

from reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.online
def test_basic_antennas(reachy_sdk_zeroed: ReachySDK) -> None:
    l_goal_position = -20
    r_goal_position = 50
    reachy_sdk_zeroed.head.l_antenna.goal_position = l_goal_position
    reachy_sdk_zeroed.head.r_antenna.goal_position = r_goal_position
    reachy_sdk_zeroed.send_goal_positions()
    time.sleep(1)
    assert np.isclose(reachy_sdk_zeroed.head.l_antenna.present_position, l_goal_position, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.r_antenna.present_position, r_goal_position, 1e-03)


@pytest.mark.online
def test_antennas_goto(reachy_sdk_zeroed: ReachySDK) -> None:
    reachy_sdk_zeroed.head.l_antenna.goto(-20)
    reachy_sdk_zeroed.head.r_antenna.goto(50, wait=True)
    assert np.isclose(reachy_sdk_zeroed.head.l_antenna.present_position, -20, 1e-03)
    assert np.isclose(reachy_sdk_zeroed.head.r_antenna.present_position, 50, 1e-03)

    l_id = reachy_sdk_zeroed.head.l_antenna.goto(40)
    r_id = reachy_sdk_zeroed.head.r_antenna.goto(-30, interpolation_mode="linear", duration=1)
    l_req = reachy_sdk_zeroed.get_goto_request(l_id)
    r_req = reachy_sdk_zeroed.get_goto_request(r_id)

    assert l_req.part == "l_antenna"
    assert np.isclose(l_req.request.target.joints, 40.0, 1e-03)
    assert np.isclose(l_req.request.duration, 2.0, 1e-03)
    assert l_req.request.mode == "minimum_jerk"
    assert l_req.request.interpolation_space == "joint_space"

    assert r_req.part == "r_antenna"
    assert np.isclose(r_req.request.target.joints, -30.0, 1e-03)
    assert np.isclose(r_req.request.duration, 1.0, 1e-03)
    assert r_req.request.mode == "linear"
    assert r_req.request.interpolation_space == "joint_space"

    l_id = reachy_sdk_zeroed.head.l_antenna.goto(10, wait=True)
    assert reachy_sdk_zeroed.is_goto_finished(l_id)
    assert np.isclose(reachy_sdk_zeroed.head.l_antenna.present_position, 10, 1e-03)

    reachy_sdk_zeroed.head.r_antenna.goto(20)
    reachy_sdk_zeroed.head.r_antenna.goto(40)
    r_id = reachy_sdk_zeroed.head.r_antenna.goto(0)
    reachy_sdk_zeroed.head.r_antenna.goto(30)
    time.sleep(0.1)
    assert len(reachy_sdk_zeroed.head.r_antenna.get_goto_queue()) == 3
    reachy_sdk_zeroed.cancel_goto_by_id(r_id)
    time.sleep(0.1)
    assert len(reachy_sdk_zeroed.head.r_antenna.get_goto_queue()) == 2
    reachy_sdk_zeroed.head.r_antenna.cancel_all_goto()
    time.sleep(0.1)
    assert len(reachy_sdk_zeroed.head.r_antenna.get_goto_queue()) == 0
    assert reachy_sdk_zeroed.is_goto_finished(r_id)
    assert reachy_sdk_zeroed.head.r_antenna.get_goto_playing().id == -1
