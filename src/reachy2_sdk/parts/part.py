from abc import ABC, abstractmethod
from typing import Any, Dict

import grpc
from reachy2_sdk_api.arm_pb2 import Arm as Arm_proto
from reachy2_sdk_api.arm_pb2 import ArmState
from reachy2_sdk_api.arm_pb2_grpc import ArmServiceStub
from reachy2_sdk_api.hand_pb2 import Hand as Hand_proto
from reachy2_sdk_api.hand_pb2 import HandState
from reachy2_sdk_api.hand_pb2_grpc import HandServiceStub
from reachy2_sdk_api.head_pb2 import Head as Head_proto
from reachy2_sdk_api.head_pb2 import HeadState
from reachy2_sdk_api.head_pb2_grpc import HeadServiceStub
from reachy2_sdk_api.mobile_base_utility_pb2 import MobileBase as MobileBase_proto
from reachy2_sdk_api.mobile_base_utility_pb2 import MobileBaseState
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

        self._actuators: Dict[str, Any] = {}

    def turn_on(self) -> None:
        self._stub.TurnOn(self._part_id)

    def turn_off(self) -> None:
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
