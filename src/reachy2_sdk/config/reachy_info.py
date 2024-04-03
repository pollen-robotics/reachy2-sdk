"""ReachyInfo module.

This module provides main info of the robot.
"""
from typing import Any, Dict, List, Optional

from mobile_base_sdk import MobileBaseSDK
from reachy2_sdk_api.reachy_pb2 import Reachy


class ReachyInfo:
    """The ReachyInfo class saves information of the robot that won't be modified during the session.

    The ReachyInfo class gives access to:
        - the robot's serial_number, that will never change
        - the robot's hardware version, that will not change during a session
        - the robot's core software version, that will not change during a session
    """

    def __init__(self, reachy: Reachy) -> None:
        self._robot_serial_number: str = reachy.info.serial_number

        self._hardware_version: str = reachy.info.version_hard
        self._core_software_version: str = reachy.info.version_soft

        self._enabled_parts: Dict[str, Any] = {}
        self._disabled_parts: List[str] = []
        self._mobile_base: Optional[MobileBaseSDK] = None

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

    def _set_mobile_base(self, mobile_base: MobileBaseSDK) -> None:
        self._mobile_base = mobile_base

    def __repr__(self) -> str:
        """Clean representation of an Reachy info."""
        repr_template = (
            '<ReachyInfo robot_serial_number="{serial_number}" \n'
            ' hardware_version="{hardware_version}" \n'
            ' core_software_version="{software_version}" \n'
            " battery_voltage={battery_voltage} >"
        )
        return repr_template.format(
            serial_number=self.robot_serial_number,
            hardware_version=self.hardware_version,
            software_version=self.core_software_version,
            battery_voltage=self.battery_voltage,
        )

    @property
    def battery_voltage(self) -> float:
        """Returns mobile base battery voltage.

        If there is no mobile base, returns full battery value.
        """
        if self._mobile_base is not None:
            # ToDo : https://github.com/pollen-robotics/mobile-base-sdk/issues/18
            # and removing cast
            return (float)(self._mobile_base.battery_voltage)
        return 30.0

    @property
    def robot_serial_number(self) -> str:
        return self._robot_serial_number

    @property
    def hardware_version(self) -> str:
        return self._hardware_version

    @property
    def core_software_version(self) -> str:
        return self._core_software_version
