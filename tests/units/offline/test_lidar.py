import grpc
import pytest
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.mobile_base_lidar_pb2 import (
    LidarObstacleDetectionEnum,
    LidarObstacleDetectionStatus,
    LidarSafety,
)

from reachy2_sdk.subparts.lidar import Lidar


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

    lidar = Lidar(initial_state=lidar_safety, grpc_channel=grpc_channel)

    assert lidar.safety_enabled == True
    assert lidar.safety_slowdown_distance == 3.0
    assert lidar.safety_critical_distance == 1.0
    assert lidar.obstacle_detection_status == "NO_OBJECT_DETECTED"

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
