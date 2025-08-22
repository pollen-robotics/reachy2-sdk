"""ReachyInfo module.

This module provides main informations about the robot.
"""

import logging
from typing import Any, Dict, List, Optional

import reachy2_sdk_api
from reachy2_sdk_api.reachy_pb2 import Reachy, ReachyCoreMode

from ..parts.mobile_base import MobileBase


class ReachyInfo:
    """The ReachyInfo class saves information of the global robot.

    The ReachyInfo class gives access to informations that won't be modified during the session:
        the robot's hardware version
        the robot's core software version
        the robot's configuration
        the robot's serial_number
    But also to the battery voltage.
    """

    def __init__(self, reachy: Reachy) -> None:
        """Initialize the ReachyInfo instance with robot details.

        Args:
            reachy: The Reachy robot object, which provides the robot's info and configuration details.
        """
        self._logger = logging.getLogger(__name__)
        self._robot_serial_number: str = reachy.info.serial_number

        self._hardware_version: str = reachy.info.version_hard
        self._core_software_version: str = reachy.info.version_soft

        try:
            self._robot_api_version: Optional[str] = reachy.info.api_version if reachy.info.api_version else None
            self._check_api_compatibility()
        except AttributeError:
            self._robot_api_version = None
            self._logger.warning(
                "Your local API version is below the required version for reachy2_sdk."
                "\nPlease update the reachy2_sdk_api package to ensure compatibility."
            )

        self._enabled_parts: Dict[str, Any] = {}
        self._disabled_parts: List[str] = []
        self._mobile_base: Optional[MobileBase] = None

        self._mode: ReachyCoreMode = reachy.info.core_mode

        self._set_config(reachy)

    def _check_api_compatibility(self) -> None:
        """Check the compatibility of the API versions between the robot and the SDK."""
        if self._robot_api_version is None:
            self._logger.warning(
                "The robot's API version is below your local API version."
                "\nSome features may not work properly."
                "\nPlease update the reachy2_core image on the robot to ensure compatibility."
                " or downgrade your local reachy2_sdk package."
            )
        elif reachy2_sdk_api.__version__ != self._robot_api_version:
            local_version = reachy2_sdk_api.__version__.split(".")
            robot_version = self._robot_api_version.split(".")
            for local, remote in zip(local_version, robot_version):
                if int(local) > int(remote):
                    self._logger.warning(
                        f"Local API version ({reachy2_sdk_api.__version__}) is different"
                        f" from the robot's API version ({self._robot_api_version})."
                        f"\nSome features may not work properly."
                        f"\nPlease update the reachy2_core image on the robot to ensure compatibility,"
                        f" or downgrade your local reachy2_sdk package."
                    )
                    break
                elif int(local) < int(remote):
                    self._logger.warning(
                        f"Local API version ({reachy2_sdk_api.__version__}) is different"
                        f" from the robot's API version ({self._robot_api_version})."
                        f"\nSome features may not work properly."
                        f"\nPlease update your local reachy2_sdk package to ensure compatibility."
                    )
                    break

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
            '<ReachyInfo mode="{mode}" \n'
            ' robot_serial_number="{serial_number}" \n'
            ' hardware_version="{hardware_version}" \n'
            ' core_software_version="{software_version}" \n'
            ' robot_api_version="{api_version}" \n'
            " battery_voltage={battery_voltage} >"
        )
        return repr_template.format(
            mode=self.mode,
            serial_number=self.robot_serial_number,
            hardware_version=self.hardware_version,
            software_version=self.core_software_version,
            api_version=self._robot_api_version,
            battery_voltage=self.battery_voltage,
        )

    @property
    def battery_voltage(self) -> float:
        """Get the battery voltage of the mobile base.

        If the mobile base is present, returns its battery voltage. Otherwise, returns a default full
        battery value.
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

    @property
    def mode(self) -> str:
        """Returns the robot's core mode.

        Can be either "FAKE", "REAL", "GAZEBO" or "MUJOCO".
        """
        return str(ReachyCoreMode.keys()[self._mode])
