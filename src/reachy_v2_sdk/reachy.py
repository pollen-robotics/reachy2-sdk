from reachy_sdk_api_v2.reachy_pb2 import ReachyInfo as ReachyInfo_proto, Reachy


class ReachyInfo:
    def __init__(self, host: str, info_msg: ReachyInfo_proto) -> None:
        self.robot_serial_number = info_msg.serial_number

        self.hardware_version = info_msg.version_hard
        self.core_software_version = info_msg.version_soft


def get_config(msg: Reachy) -> str:
    if msg.HasField("l_arm"):
        return "full_kit"
    else:
        return "none"
