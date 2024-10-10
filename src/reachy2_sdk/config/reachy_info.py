"""ReachyInfo module.

This module provides main informations about the robot.
"""

from typing import Any, Dict, List, Optional

from reachy2_sdk_api.reachy_pb2 import Reachy

from ..parts.mobile_base import MobileBase


class ReachyInfo:
    """The ReachyInfo class saves information of the global robot.

    The ReachyInfo class gives access to informations that won't be modified during the session:
        - the robot's hardware version
        - the robot's core software version
        - the robot's configuration
        - the robot's serial_number
    But also to the battery voltage.
    """

    def __init__(self, reachy: Reachy) -> None:
        """Initialize the ReachyInfo instance with robot details.

        Args:
            reachy: The Reachy robot object, which provides the robot's info and configuration details.
        """
        self._robot_serial_number: str = reachy.info.serial_number

        self._hardware_version: str = reachy.info.version_hard
        self._core_software_version: str = reachy.info.version_soft

        self._enabled_parts: Dict[str, Any] = {}
        self._disabled_parts: List[str] = []
        self._mobile_base: Optional[MobileBase] = None

        self._set_config(reachy)

    def _set_config(self, msg: Reachy) -> None:
        """Determine the robot's configuration.

        Sets the configuration string to indicate whether the robot is a full kit, starter kit
        (with left or right arm), or a custom configuration. Also accounts for the presence of a mobile base.

        Args:
            msg: The Reachy instance containing the current configuration of the robot.
        """
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

    def _set_mobile_base(self, mobile_base: MobileBase) -> None:
        """Set the mobile base for the robot.

        Args:
            mobile_base: The MobileBase instance to associate with the robot.
        """
        self._mobile_base = mobile_base

    def __repr__(self) -> str:
        """Clean representation of a ReachyInfo."""
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
        """Get the battery voltage of the mobile base.

        If the mobile base is present, returns its battery voltage. Otherwise, returns a default full
        battery value.

        Returns:
            The battery voltage as a float.
        """
        if self._mobile_base is not None:
            # ToDo : https://github.com/pollen-robotics/mobile-base-sdk/issues/18
            # and removing cast
            return (float)(self._mobile_base.battery_voltage)
        return 30.0

    @property
    def robot_serial_number(self) -> str:
        """Returns the robot's serial number."""
        return self._robot_serial_number

    @property
    def hardware_version(self) -> str:
        """Returns the robot's hardware version."""
        return self._hardware_version

    @property
    def core_software_version(self) -> str:
        """Returns the robot's core software version."""
        return self._core_software_version
