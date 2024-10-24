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

    assert (
        np.round(reachy2.r_arm.get_current_positions(), 3).tolist()
        == np.round(reachy_sdk_zeroed.r_arm.get_current_positions()).tolist()
    )

    for joint in reachy_sdk_zeroed.joints.values():
        joint.goal_position = -10
    reachy_sdk_zeroed.send_goal_positions()

    time.sleep(0.2)
    assert np.round(reachy_sdk_zeroed.r_arm.get_current_positions(), 3).tolist() == [-10, -10, -10, -10, -10, -10, -10]
    assert reachy2.r_arm.get_current_positions() == reachy_sdk_zeroed.r_arm.get_current_positions()

    for joint in reachy2.joints.values():
        joint.goal_position = 0
    reachy2.send_goal_positions()

    time.sleep(0.2)
    assert np.round(reachy2.r_arm.get_current_positions(), 3).tolist() == [0, 0, 0, 0, 0, 0, 0]
    assert reachy2.r_arm.get_current_positions() == reachy_sdk_zeroed.r_arm.get_current_positions()

    move1_goal = [10, 20, 25, -90, 10, 10, 10]
    move1_id = reachy_sdk_zeroed.l_arm.goto(move1_goal, duration=5)

    time.sleep(0.1)

    while not reachy_sdk_zeroed.is_goto_finished(move1_id):
        # update loops are not synchronized, we do not expect the same values
        assert np.allclose(reachy_sdk_zeroed.l_arm.get_current_positions(), reachy2.l_arm.get_current_positions(), atol=1)
    assert np.allclose(reachy_sdk_zeroed.l_arm.get_current_positions(), move1_goal, atol=1e-03)
    assert np.allclose(reachy2.l_arm.get_current_positions(), move1_goal, atol=1e-03)
