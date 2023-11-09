from reachy_sdk_api_v2.reachy_pb2 import Reachy
from reachy_sdk_api_v2.reachy_pb2 import ReachyInfo as ReachyInfo_proto


class ReachyInfo:
    def __init__(self, host: str, info_msg: ReachyInfo_proto) -> None:
        self.robot_serial_number = info_msg.serial_number

        self.hardware_version = info_msg.version_hard
        self.core_software_version = info_msg.version_soft


def get_config(msg: Reachy) -> str:
    mobile_base_presence = ""
    if msg.HasField("mobile_base"):
        mobile_base_presence = " with mobile_base"
    if msg.HasField("head"):
        if msg.HasField("l_arm") and msg.HasField("r_arm"):
            return "full_kit" + mobile_base_presence
        elif msg.HasField("l_arm"):
            return "starter_kit (left arm)" + mobile_base_presence
        else:
            return "starter_kit (right arm)" + mobile_base_presence
    else:
        return "custom_config"
