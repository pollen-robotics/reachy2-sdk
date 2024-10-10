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
    """Base class for parts of the robot composed of controllable joints.

    The `JointsBasedPart` class serves as a base for parts of the robot that consist of multiple joints,
    such as arms and heads. This class provides common functionality for controlling joints, setting speed
    and torque limits, and managing joint positions.

    Attributes:
        _joints: A dictionary containing all the joints of the part, with joint names as keys and joint objects as values.

    Methods:
        get_joints_positions: Abstract method to retrieve the current positions of all joints.
        send_goal_positions: Abstract method to send goal positions to the part's joints.
        set_torque_limits: Set the torque limit as a percentage of the maximum torque for all motors.
        set_speed_limits: Set the speed limit as a percentage of the maximum speed for all motors.
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
        """Get all the arm's joints.

        Returns:
            A dictionary of all the arm's joints, with joint names as keys and joint objects as values.
        """
        _joints: CustomDict[str, OrbitaJoint] = CustomDict({})
        for actuator_name, actuator in self._actuators.items():
            for joint in actuator._joints.values():
                _joints[actuator_name + "." + joint._axis_type] = joint
        return _joints

    @abstractmethod
    def get_joints_positions(self) -> List[float]:
        """Get the current positions of all joints.

        Returns:
            A list of float values representing the present positions in degrees of the arm's joints.
        """
        pass

    @abstractmethod
    def send_goal_positions(self) -> None:
        """Send goal positions to the part's joints.

        If goal positions have been specified for any joint of the part, sends them to the robot.
        """
        pass

    def set_torque_limits(self, value: int) -> None:
        """Set the torque limit as a percentage of the maximum torque for all motors of the part.

        Args:
            torque_limit: The desired torque limit as a percentage (0-100) of the maximum torque. Can be
                specified as a float or int.
        """
        if not isinstance(value, float | int):
            raise ValueError(f"Expected one of: float, int for torque_limit, got {type(value).__name__}")
        if not (0 <= value <= 100):
            raise ValueError(f"torque_limit must be in [0, 100], got {value}.")
        req = TorqueLimitRequest(
            id=self._part_id,
            limit=value,
        )
        self._stub.SetTorqueLimit(req)

    def set_speed_limits(self, value: int) -> None:
        """Set the speed limit as a percentage of the maximum speed for all motors of the part.

        Args:
            speed_limit: The desired speed limit as a percentage (0-100) of the maximum speed. Can be
                specified as a float or int.
        """
        if not isinstance(value, float | int):
            raise ValueError(f"Expected one of: float, int for speed_limit, got {type(value).__name__}")
        if not (0 <= value <= 100):
            raise ValueError(f"speed_limit must be in [0, 100], got {value}.")
        req = SpeedLimitRequest(
            id=self._part_id,
            limit=value,
        )
        self._stub.SetSpeedLimit(req)

    def _set_speed_limits(self, value: int) -> None:
        """Set the speed limit as a percentage of the maximum speed for all motors of the part.

        Args:
            speed_limit: The desired speed limit as a percentage (0-100) of the maximum speed. Can be
                specified as a float or int.
        """
        return self.set_speed_limits(value)
