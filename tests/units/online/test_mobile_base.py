import time

import pytest

from reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.online
def test_modes(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        reachy_sdk_zeroed.turn_on()
        time.sleep(0.2)
        assert reachy_sdk_zeroed.mobile_base.is_on()

        reachy_sdk_zeroed.mobile_base.turn_off()
        time.sleep(0.2)
        assert reachy_sdk_zeroed.mobile_base._drive_mode == "free_wheel"
        assert reachy_sdk_zeroed.mobile_base.is_off()

        reachy_sdk_zeroed.mobile_base._set_drive_mode("brake")
        time.sleep(0.2)
        assert reachy_sdk_zeroed.mobile_base._drive_mode == "brake"
        assert reachy_sdk_zeroed.mobile_base.is_on()

        reachy_sdk_zeroed.mobile_base._set_control_mode("pid")
        time.sleep(0.2)
        assert reachy_sdk_zeroed.mobile_base._control_mode == "pid"


@pytest.mark.online
def test_lidar_safety_distances(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        reachy_sdk_zeroed.mobile_base.lidar.safety_slowdown_distance = 5.0
        reachy_sdk_zeroed.mobile_base.lidar.safety_critical_distance = 1.0

        time.sleep(0.2)
        assert reachy_sdk_zeroed.mobile_base.lidar.safety_slowdown_distance == 5.0
        assert reachy_sdk_zeroed.mobile_base.lidar.safety_critical_distance == 1.0

        reachy_sdk_zeroed.mobile_base.lidar.reset_safety_default_values()

        time.sleep(0.2)
        assert reachy_sdk_zeroed.mobile_base.lidar.safety_slowdown_distance == 0.7
        assert reachy_sdk_zeroed.mobile_base.lidar.safety_critical_distance == 0.55
