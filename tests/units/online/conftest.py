"""
Common functions shared by all tests
"""

import time

import pytest

from reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.online
@pytest.fixture(scope="package")
def reachy_sdk() -> ReachySDK:
    """
    Same connection for all online tests
    """

    reachy = ReachySDK(host="localhost")
    assert reachy.is_connected()

    assert reachy.turn_on()

    yield reachy

    assert reachy.turn_off()

    reachy.disconnect()


@pytest.fixture
def reachy_sdk_zeroed(reachy_sdk: ReachySDK) -> ReachySDK:
    reachy_sdk.cancel_all_goto()
    for joint in reachy_sdk.joints.values():
        joint.goal_position = 0
    reachy_sdk.send_goal_positions()
    reachy_sdk.r_arm.gripper.set_opening(100)
    reachy_sdk.l_arm.gripper.set_opening(100)

    time.sleep(1)

    return reachy_sdk
