import grpc
import pytest
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.mobile_base_lidar_pb2 import (
    LidarObstacleDetectionEnum,
    LidarObstacleDetectionStatus,
    LidarSafety,
)
from reachy2_sdk_api.mobile_base_utility_pb2 import BatteryLevel
from reachy2_sdk_api.mobile_base_utility_pb2 import MobileBase as MobileBase_proto
from reachy2_sdk_api.mobile_base_utility_pb2 import MobileBaseInfo, MobileBaseState

from reachy2_sdk.parts.mobile_base import MobileBase


@pytest.mark.offline
def test_class() -> None:
    grpc_channel = grpc.insecure_channel("dummy:5050")

    mb_info = MobileBaseInfo(
        serial_number="MB-000",
        version_hard="1.2",
        version_soft="1.2",
    )

    mb_proto = MobileBase_proto(
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

    mb_state = MobileBaseState(battery_level=battery, lidar_safety=lidar_safety)

    mobile_base = MobileBase(mb_msg=mb_proto, initial_state=mb_state, grpc_channel=grpc_channel)

    assert mobile_base.lidar is not None
    assert mobile_base.battery_voltage == 25
