from reachy_v2_sdk_api.config_pb2 import ConfigReachy


class BaseInfo:
    def __init__(self, host: str, config_msg: ConfigReachy) -> None:
        self.ip_address = host

        self.config = config_msg.config
        self.with_mobile_base = config_msg.with_mobile_base

        self.enabled_parts = config_msg.enabled_parts
        self.disabled_parts = config_msg.disabled_parts

        self.core_software_version = config_msg.core_software_version

        self.robot_serial_number = config_msg.robot_serial_number
        self.mobile_base_serial_number = config_msg.mobile_base_serial_number
