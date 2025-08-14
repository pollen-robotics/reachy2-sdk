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
from reachy2_sdk_api.reachy_pb2 import Reachy
from reachy2_sdk_api.reachy_pb2 import ReachyInfo as ReachyInfo_proto

from reachy2_sdk.config.reachy_info import ReachyInfo
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

    mobile_base = MobileBase(mb_msg=mb_proto, initial_state=mb_state, grpc_channel=grpc_channel, goto_stub=None)

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

    serial_number = "Reachy-12345"
    version_hard = "1.1"
    version_soft = "1.2"
    robot_info = ReachyInfo_proto(serial_number=serial_number, version_hard=version_hard, version_soft=version_soft)
    reachy = Reachy(info=robot_info)

    ri = ReachyInfo(reachy)
    ri._set_mobile_base(mobile_base)
    assert ri.battery_voltage == 20.0

    mobile_base.goto(0, 0, 0, 0)

    mobile_base._set_speed_limits(100)

    mobile_base.set_max_xy_goto(2.0)
    assert mobile_base._max_xy_goto == 2.0
