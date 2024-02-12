import pytest
from reachy2_sdk_api.arm_pb2 import Arm
from reachy2_sdk_api.head_pb2 import Head
from reachy2_sdk_api.mobile_base_utility_pb2 import MobileBase
from reachy2_sdk_api.reachy_pb2 import Reachy
from reachy2_sdk_api.reachy_pb2 import ReachyInfo as ReachyInfo_proto

from reachy2_sdk.config.reachy_info import ReachyInfo


@pytest.mark.offline
def test_ReachyInfo() -> None:
    serial_number = "Reachy-12345"
    version_hard = "1.1"
    version_soft = "1.2"
    robot_info = ReachyInfo_proto(serial_number=serial_number, version_hard=version_hard, version_soft=version_soft)
    reachy = Reachy(info=robot_info)

    ri = ReachyInfo(reachy)

    assert ri.robot_serial_number == serial_number
    assert ri.hardware_version == version_hard
    assert ri.core_software_version == version_soft
    assert ri.battery_voltage == 30.0


@pytest.mark.offline
def test_config() -> None:
    serial_number = "Reachy-12345"
    version_hard = "1.1"
    version_soft = "1.2"
    robot_info = ReachyInfo_proto(serial_number=serial_number, version_hard=version_hard, version_soft=version_soft)

    robot = Reachy(info=robot_info)
    ri = ReachyInfo(robot)
    assert ri.config == "custom_config"

    robot = Reachy(head=Head(), l_arm=Arm(), info=robot_info)
    ri = ReachyInfo(robot)
    assert ri.config == "starter_kit (left arm)"

    robot = Reachy(head=Head(), r_arm=Arm(), info=robot_info)
    ri = ReachyInfo(robot)
    assert ri.config == "starter_kit (right arm)"

    robot = Reachy(head=Head(), l_arm=Arm(), r_arm=Arm(), info=robot_info)
    ri = ReachyInfo(robot)
    assert ri.config == "full_kit"

    robot = Reachy(head=Head(), l_arm=Arm(), mobile_base=MobileBase(), info=robot_info)
    ri = ReachyInfo(robot)
    assert ri.config == "starter_kit (left arm) with mobile_base"

    robot = Reachy(head=Head(), r_arm=Arm(), mobile_base=MobileBase(), info=robot_info)
    ri = ReachyInfo(robot)
    assert ri.config == "starter_kit (right arm) with mobile_base"

    robot = Reachy(head=Head(), l_arm=Arm(), r_arm=Arm(), mobile_base=MobileBase(), info=robot_info)
    ri = ReachyInfo(robot)
    assert ri.config == "full_kit with mobile_base"
