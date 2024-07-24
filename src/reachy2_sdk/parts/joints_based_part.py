from abc import abstractmethod
from typing import List

import grpc
from reachy2_sdk_api.arm_pb2 import Arm as Arm_proto
from reachy2_sdk_api.arm_pb2 import SpeedLimitRequest, TorqueLimitRequest
from reachy2_sdk_api.arm_pb2_grpc import ArmServiceStub
from reachy2_sdk_api.head_pb2 import Head as Head_proto
from reachy2_sdk_api.head_pb2_grpc import HeadServiceStub
from reachy2_sdk_api.mobile_base_utility_pb2 import MobileBase as MobileBase_proto
from reachy2_sdk_api.mobile_base_utility_pb2_grpc import MobileBaseUtilityServiceStub

from ..orbita.orbita_joint import OrbitaJoint
from ..utils.custom_dict import CustomDict
from .part import Part


class JointsBasedPart(Part):
    """The JointsBasedPart class is an class to represent any part of the robot composed of controllable joints.

    This class is meant to be derived by relevant parts of the robot : Arm, Head
    """

    def __init__(
        self,
        proto_msg: Arm_proto | Head_proto | MobileBase_proto,
        grpc_channel: grpc.Channel,
        stub: ArmServiceStub | HeadServiceStub | MobileBaseUtilityServiceStub,
    ) -> None:
        """Initialize the common attributes."""
        super().__init__(proto_msg, grpc_channel, stub)

    @property
    def joints(self) -> CustomDict[str, OrbitaJoint]:
        """Get all the arm's joints."""
        _joints: CustomDict[str, OrbitaJoint] = CustomDict({})
        for actuator_name, actuator in self._actuators.items():
            for joint in actuator._joints.values():
                _joints[actuator_name + "." + joint._axis_type] = joint
        return _joints

    @abstractmethod
    def get_joints_positions(self) -> List[float]:
        pass

    def set_torque_limit(self, value: int) -> None:
        """Choose percentage of torque max value applied as limit to the arms."""
        req = TorqueLimitRequest(
            id=self._part_id,
            limit=value,
        )
        self._stub.SetTorqueLimit(req)

    def set_speed_limit(self, value: int) -> None:
        """Choose percentage of speed max value applied as limit to the arms."""
        req = SpeedLimitRequest(
            id=self._part_id,
            limit=value,
        )
        self._stub.SetSpeedLimit(req)
