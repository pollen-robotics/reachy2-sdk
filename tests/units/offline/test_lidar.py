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
from reachy2_sdk.sensors.lidar import Lidar


@pytest.mark.offline
def test_class() -> None:
    grpc_channel = grpc.insecure_channel("dummy:5050")

    lidar_detection = LidarObstacleDetectionStatus(status=LidarObstacleDetectionEnum.NO_OBJECT_DETECTED)
    lidar_safety = LidarSafety(
        safety_on=BoolValue(value=True),
        safety_distance=FloatValue(value=3.0),
        critical_distance=FloatValue(value=1.0),
        obstacle_detection_status=lidar_detection,
    )

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

    lidar = Lidar(initial_state=lidar_safety, grpc_channel=grpc_channel, part=mobile_base)

    assert lidar.safety_enabled == True
    assert lidar.safety_slowdown_distance == 3.0
    assert lidar.safety_critical_distance == 1.0
    assert lidar.obstacle_detection_status == "NO_OBJECT_DETECTED"

    assert lidar.__repr__() != ""

    new_lidar_detection = LidarObstacleDetectionStatus(status=LidarObstacleDetectionEnum.OBJECT_DETECTED_SLOWDOWN)
    new_lidar_safety = LidarSafety(
        safety_on=BoolValue(value=False),
        safety_distance=FloatValue(value=2.5),
        critical_distance=FloatValue(value=0.5),
        obstacle_detection_status=new_lidar_detection,
    )

    lidar._update_with(new_lidar_safety)

    assert lidar.safety_enabled == False
    assert lidar.safety_slowdown_distance == 2.5
    assert lidar.safety_critical_distance == 0.5
    assert lidar.obstacle_detection_status == "OBJECT_DETECTED_SLOWDOWN"
