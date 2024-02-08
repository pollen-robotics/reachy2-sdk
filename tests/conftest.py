"""
Common functions shared by all tests
"""

import time

import pytest

from src.reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.online
@pytest.fixture(scope="package")
def reachy_sdk() -> ReachySDK:
    """
    Same connection for all online tests
    """

    reachy = ReachySDK(host="localhost")
    assert reachy.grpc_status == "connected"

    assert reachy.turn_on()

    yield reachy

    assert reachy.turn_off()

    reachy.disconnect()
    ReachySDK.clear()


@pytest.fixture
def reachy_sdk_zeroed(reachy_sdk: ReachySDK) -> ReachySDK:
    reachy_sdk.cancel_all_goto()
    for joint in reachy_sdk.joints.values():
        joint.goal_position = 0

    time.sleep(1)

    return reachy_sdk
