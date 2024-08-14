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
        self._turn_on()
        time.sleep(0.5)

    def turn_off(self) -> None:
        self._turn_off()
        time.sleep(0.5)

    def _turn_on(self) -> None:
        self._stub.TurnOn(self._part_id)

    def _turn_off(self) -> None:
        self._stub.TurnOff(self._part_id)

    def is_on(self) -> bool:
        for actuator in self._actuators.values():
            if not actuator.is_on():
                return False
        return True

    def is_off(self) -> bool:
        for actuator in self._actuators.values():
            if actuator.is_on():
                return False
        return True

    @abstractmethod
    def _update_with(self, state: ArmState | HeadState | HandState | MobileBaseState) -> None:
        pass

    @abstractmethod
    def _update_audit_status(self, state: ArmStatus | HeadStatus | HandStatus | MobileBaseStatus) -> None:
        pass

    @property
    def audit(self) -> Dict[str, str]:
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
