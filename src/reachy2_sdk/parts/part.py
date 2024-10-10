import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

import grpc
from reachy2_sdk_api.arm_pb2 import Arm as Arm_proto
from reachy2_sdk_api.arm_pb2 import ArmState, ArmStatus
from reachy2_sdk_api.arm_pb2_grpc import ArmServiceStub
from reachy2_sdk_api.hand_pb2 import Hand as Hand_proto
from reachy2_sdk_api.hand_pb2 import HandState, HandStatus
from reachy2_sdk_api.hand_pb2_grpc import HandServiceStub
from reachy2_sdk_api.head_pb2 import Head as Head_proto
from reachy2_sdk_api.head_pb2 import HeadState, HeadStatus
from reachy2_sdk_api.head_pb2_grpc import HeadServiceStub
from reachy2_sdk_api.mobile_base_utility_pb2 import MobileBase as MobileBase_proto
from reachy2_sdk_api.mobile_base_utility_pb2 import MobileBaseState, MobileBaseStatus
from reachy2_sdk_api.mobile_base_utility_pb2_grpc import MobileBaseUtilityServiceStub
from reachy2_sdk_api.part_pb2 import PartId


class Part(ABC):
    """The Part class is an abstract class to represent any part of the robot.

    This class is meant to be derived by any part of the robot : Arm, Hand, Head, MobileBase
    """

    def __init__(
        self,
        proto_msg: Arm_proto | Head_proto | Hand_proto | MobileBase_proto,
        grpc_channel: grpc.Channel,
        stub: ArmServiceStub | HeadServiceStub | HandServiceStub | MobileBaseUtilityServiceStub,
    ) -> None:
        """Initialize the common attributes."""
        self._grpc_channel = grpc_channel
        self._stub = stub
        self._part_id = PartId(id=proto_msg.part_id.id, name=proto_msg.part_id.name)
        self._logger = logging.getLogger(__name__)

        self._actuators: Dict[str, Any] = {}

    def turn_on(self) -> None:
        """Turn on the part.

        This method sets the speed limits to a low value, turns on the part, and then restores the speed limits to maximum.
        It waits for a brief period to ensure the operation is complete.
        """
        self._set_speed_limits(1)
        time.sleep(0.05)
        self._turn_on()
        time.sleep(0.05)
        self._set_speed_limits(100)
        time.sleep(0.4)

    def turn_off(self) -> None:
        """Turn off the part.

        This method shuts down the part and waits for a brief period to ensure the operation is complete.
        """
        self._turn_off()
        time.sleep(0.5)

    def _turn_on(self) -> None:
        """Send a command to turn on immediately the part."""
        self._stub.TurnOn(self._part_id)

    def _turn_off(self) -> None:
        """Send a command to turn off immediately the part."""
        self._stub.TurnOff(self._part_id)

    def is_on(self) -> bool:
        """Check if all actuators of the part are currently on.

        Returns:
            True if all actuators are on, otherwise False.
        """
        for actuator in self._actuators.values():
            if not actuator.is_on():
                return False
        return True

    def is_off(self) -> bool:
        """Check if all actuators of the part are currently off.

        Returns:
            True if all actuators are off, otherwise False.
        """
        for actuator in self._actuators.values():
            if actuator.is_on():
                return False
        return True

    @abstractmethod
    def _update_with(self, state: ArmState | HeadState | HandState | MobileBaseState) -> None:
        """Update the part's state with newly received data.

        This method must be implemented by subclasses to update the state of the part based on
        specific state data types such as ArmState, HeadState, HandState, or MobileBaseState.

        Args:
            state: The state data used to update the part, which can be an ArmState, HeadState,
                HandState, or MobileBaseState.
        """
        pass

    @abstractmethod
    def _update_audit_status(self, state: ArmStatus | HeadStatus | HandStatus | MobileBaseStatus) -> None:
        """Update the audit status of the part.

        This method must be implemented by subclasses to update the audit status of the part based on
        specific status data types such as ArmStatus, HeadStatus, HandStatus, or MobileBaseStatus.

        Args:
            state: The status data used to update the audit status, which can be an ArmStatus,
                HeadStatus, HandStatus, or MobileBaseStatus.
        """
        pass

    @abstractmethod
    def _set_speed_limits(self, value: int) -> None:
        """Set the speed limits for the part.

        This method must be implemented by subclasses to set speed limits.

        Args:
            value: The speed limit value to be set, as a percentage of the maximum speed allowed (0-100).
        """
        pass

    @property
    def audit(self) -> Dict[str, str]:
        """Get the audit status of all actuators of the part.

        Returns:
            A dictionary where each key is the name of an actuator and the value is its audit status.
            If an error is detected in any actuator, a warning is logged. Otherwise, an informational
            message indicating no errors is logged.
        """
        error_dict: Dict[str, str] = {}
        error_detected = False
        for act_name, actuator in self._actuators.items():
            error_dict[act_name] = actuator.audit
            if actuator.audit != "Ok":
                self._logger.warning(f'Error detected on {self._part_id.name}_{act_name}: "{actuator.audit}"')
                error_detected = True
        if not error_detected:
            self._logger.info(f"No error detected on {self._part_id.name}")
        return error_dict
