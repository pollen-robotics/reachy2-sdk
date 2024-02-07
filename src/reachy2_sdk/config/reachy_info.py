"""ReachyInfo module.

This module provides main info of the robot.
"""
from typing import Any, Dict, List

from reachy2_sdk_api.reachy_pb2 import Reachy


class ReachyInfo:
    """The ReachyInfo class saves information of the robot that won't be modified during the session.

    The ReachyInfo class gives access to:
        - the robot's serial_number, that will never change
        - the robot's hardware version, that will not change during a session
        - the robot's core software version, that will not change during a session
    """

    def __init__(self, reachy: Reachy) -> None:
        self.robot_serial_number = reachy.info.serial_number

        self.hardware_version = reachy.info.version_hard
        self.core_software_version = reachy.info.version_soft

        self._enabled_parts: Dict[str, Any] = {}
        self._disabled_parts: List[str] = []

        self.battery_voltage: float = 30.0

        self._set_config(reachy)

    def _set_config(self, msg: Reachy) -> None:
        """Return the current configuration of the robot."""
        self.config: str = ""

        mobile_base_presence = ""
        if msg.HasField("mobile_base"):
            mobile_base_presence = " with mobile_base"
        if msg.HasField("head"):
            if msg.HasField("l_arm") and msg.HasField("r_arm"):
                self.config = "full_kit" + mobile_base_presence
            elif msg.HasField("l_arm"):
                self.config = "starter_kit (left arm)" + mobile_base_presence
            else:
                self.config = "starter_kit (right arm)" + mobile_base_presence
        else:
            self.config = "custom_config"
