import time

import numpy as np
import pytest

from reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.online
def test_same_robot(reachy_sdk_zeroed: ReachySDK) -> None:
    assert reachy_sdk_zeroed.is_connected()
    reachy2 = ReachySDK("localhost")
    time.sleep(0.2)

    assert reachy2.is_connected()

    assert reachy2.is_on() == reachy_sdk_zeroed.is_on()
    reachy_sdk_zeroed.turn_off()
    time.sleep(0.1)
    assert reachy2.is_on() == reachy_sdk_zeroed.is_on()

    reachy_sdk_zeroed.turn_on()
    time.sleep(0.1)
    assert reachy2.is_on() == reachy_sdk_zeroed.is_on()

    assert reachy2.r_arm.get_joints_positions() == reachy_sdk_zeroed.r_arm.get_joints_positions()

    for joint in reachy_sdk_zeroed.joints.values():
        joint.goal_position = -10
    reachy_sdk_zeroed.send_goal_positions()

    time.sleep(0.2)
    assert reachy_sdk_zeroed.r_arm.get_joints_positions() == [-10, -10, -10, -10, -10, -10, -10]
    assert reachy2.r_arm.get_joints_positions() == reachy_sdk_zeroed.r_arm.get_joints_positions()

    for joint in reachy2.joints.values():
        joint.goal_position = 0
    reachy2.send_goal_positions()

    time.sleep(0.2)
    assert reachy2.r_arm.get_joints_positions() == [0, 0, 0, 0, 0, 0, 0]
    assert reachy2.r_arm.get_joints_positions() == reachy_sdk_zeroed.r_arm.get_joints_positions()

    move1 = reachy_sdk_zeroed.l_arm.goto_joints([10, 20, 25, -90, 10, 10, 10], duration=5)

    time.sleep(0.1)
    assert reachy_sdk_zeroed.is_move_playing(move1) == reachy2.is_move_playing(move1)

    while not reachy_sdk_zeroed.is_move_finished(move1):
        # update loops are not synchronized, we do not expect the same values
        assert np.allclose(reachy_sdk_zeroed.l_arm.get_joints_positions(), reachy2.l_arm.get_joints_positions(), atol=1)
