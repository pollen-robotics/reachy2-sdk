import asyncio

import grpc
import pytest
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.mobile_base_lidar_pb2 import (
    LidarObstacleDetectionEnum,
    LidarObstacleDetectionStatus,
    LidarSafety,
)
from reachy2_sdk_api.mobile_base_utility_pb2 import (
    BatteryLevel,
    ControlModeCommand,
    ControlModePossiblities,
)
from reachy2_sdk_api.mobile_base_utility_pb2 import MobileBase as MobileBase_proto
from reachy2_sdk_api.mobile_base_utility_pb2 import (
    MobileBaseState,
    ZuuuModeCommand,
    ZuuuModePossiblities,
)
from reachy2_sdk_api.part_pb2 import PartId, PartInfo

from reachy2_sdk.parts.mobile_base import MobileBase


@pytest.mark.offline
def test_class() -> None:
    grpc_channel = grpc.insecure_channel("dummy:5050")

    mb_info = PartInfo(
        serial_number="MB-000",
        version_hard="1.2",
        version_soft="1.2",
    )

    mb_proto = MobileBase_proto(
        part_id=PartId(name="mobile_base", id=100),
        info=mb_info,
    )

    battery = BatteryLevel(level=FloatValue(value=25))
    lidar_detection = LidarObstacleDetectionStatus(status=LidarObstacleDetectionEnum.NO_OBJECT_DETECTED)
    lidar_safety = LidarSafety(
        safety_on=BoolValue(value=True),
        safety_distance=FloatValue(value=3.0),
        critical_distance=FloatValue(value=1.0),
        obstacle_detection_status=lidar_detection,
    )

    drive_mode = ZuuuModeCommand(mode=ZuuuModePossiblities.CMD_GOTO)
    control_mode = ControlModeCommand(mode=ControlModePossiblities.OPEN_LOOP)

    mb_state = MobileBaseState(
        battery_level=battery,
        lidar_safety=lidar_safety,
        zuuu_mode=drive_mode,
        control_mode=control_mode,
    )

    mobile_base = MobileBase(mb_msg=mb_proto, initial_state=mb_state, grpc_channel=grpc_channel)

    assert mobile_base.lidar is not None
    assert mobile_base.battery_voltage == 25
    assert mobile_base._drive_mode == "cmd_goto"
    assert mobile_base._control_mode == "open_loop"

    assert mobile_base.is_on()
    assert not mobile_base.is_off()

    assert mobile_base.__repr__() != ""

    new_battery = BatteryLevel(level=FloatValue(value=20))

    new_drive_mode = ZuuuModeCommand(mode=ZuuuModePossiblities.FREE_WHEEL)
    new_control_mode = ControlModeCommand(mode=ControlModePossiblities.PID)

    mb_new_state = MobileBaseState(
        battery_level=new_battery,
        lidar_safety=lidar_safety,
        zuuu_mode=new_drive_mode,
        control_mode=new_control_mode,
    )

    mobile_base._update_with(mb_new_state)

    assert mobile_base.battery_voltage == 20
    assert mobile_base._drive_mode == "free_wheel"
    assert mobile_base._control_mode == "pid"

    assert not mobile_base.is_on()
    assert mobile_base.is_off()

    with pytest.raises(ValueError):
        mobile_base._set_control_mode("wrong")

    with pytest.raises(ValueError):
        mobile_base._set_drive_mode("wrong")

    with pytest.raises(ValueError):
        mobile_base.set_speed(0.5, 0.5, 200)

    with pytest.raises(ValueError):
        mobile_base.set_speed(1.5, 1.5, 100)

    with pytest.raises(ValueError):
        asyncio.run(mobile_base._goto_async(x=1.5, y=1.5, theta=10, timeout=4))
