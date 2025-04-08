"""Reachy DynamixelMotor module.

Handles all specific methods to DynamixelMotor.
"""

import logging
from typing import Optional

from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from grpc import Channel
from reachy2_sdk_api.component_pb2 import ComponentId
from reachy2_sdk_api.dynamixel_motor_pb2 import (
    DynamixelMotorCommand,
    DynamixelMotorsCommand,
    DynamixelMotorState,
)
from reachy2_sdk_api.dynamixel_motor_pb2_grpc import DynamixelMotorServiceStub

from ..orbita.utils import to_internal_position, to_position


class DynamixelMotor:
    """The DynamixelMotor class represents any Dynamixel motor.

    The DynamixelMotor class is used to store the up-to-date state of the motor, especially:
    - its present_position (RO)
    - its goal_position (RW)
    """

    def __init__(
        self,
        uid: int,
        name: str,
        initial_state: DynamixelMotorState,
        grpc_channel: Channel,
    ):
        """Initialize the DynamixelMotor with its initial state and configuration.

        This sets up the motor by assigning its state based on the provided initial values.

        Args:
            uid: The unique identifier of the component.
            name: The name of the component.
            initial_state: A dictionary containing the initial state of the joint, with
                each entry representing a specific parameter of the joint (e.g., present position).
            grpc_channel: The gRPC channel used to communicate with the DynamixelMotor service.
        """
        self._logger = logging.getLogger(__name__)
        self._name = name
        self._id = uid
        self._stub = DynamixelMotorServiceStub(grpc_channel)
        self._update_with(initial_state)
        self._outgoing_goal_position: Optional[float] = None

    def __repr__(self) -> str:
        """Clean representation of the DynamixelMotor."""
        repr_template = "<DynamixelMotor on={dxl_on} present_position={present_position} goal_position={goal_position} >"
        return repr_template.format(
            dxl_on=self.is_on(),
            present_position=round(self.present_position, 2),
            goal_position=round(self.goal_position, 2),
        )

    def turn_on(self) -> bool:
        """Turn on the motor. Returns 'True' if it succeeded.

        Returns:
            `True` if the motor is stiff (not compliant), `False` otherwise.
        """
        self._set_compliant(False)
        return self.is_on()

    def turn_off(self) -> bool:
        """Turn off the motor.

        Returns:
            `True` if the motor is compliant (not stiff), `False` otherwise.
        """
        self._set_compliant(True)
        return not self.is_on()

    def is_on(self) -> bool:
        """Check if the dynamixel motor is currently stiff.

        Returns:
            `True` if the motor is stiff (not compliant), `False` otherwise.
        """
        return not self._compliant

    @property
    def present_position(self) -> float:
        """Get the present position of the joint in degrees."""
        return to_position(self._present_position)

    @property
    def goal_position(self) -> float:
        """Get the goal position of the joint in degrees."""
        return to_position(self._goal_position)

    @goal_position.setter
    def goal_position(self, value: float | int) -> None:
        """Set the goal position of the joint in degrees.

        The goal position is not send to the joint immediately, it is stored locally until the `send_goal_positions` method
        is called.

        Args:
            value: The goal position to set, specified as a float or int.

        Raises:
            TypeError: If the provided value is not a float or int.
        """
        if isinstance(value, float) or isinstance(value, int):
            self._outgoing_goal_position = to_internal_position(value)
        else:
            raise TypeError("goal_position must be a float or int")

    def _set_compliant(self, compliant: bool) -> None:
        """Set the compliance mode of the motor.

        Compliance mode determines whether the motor is stiff or compliant.

        Args:
            compliant: A boolean value indicating whether to set the motor to
                compliant (`True`) or stiff (`False`).
        """
        command = DynamixelMotorsCommand(
            cmd=[
                DynamixelMotorCommand(
                    id=ComponentId(id=self._id),
                    compliant=BoolValue(value=compliant),
                )
            ]
        )
        self._stub.SendCommand(command)

    def _get_goal_positions_message(self, check_positions: bool = True) -> Optional[DynamixelMotorsCommand]:
        """Get the DynamixelMotorsCommand message to send the goal positions to the actuator."""
        if self._outgoing_goal_position is not None:
            if not self.is_on():
                self._logger.warning(f"{self._name} is off. Command not sent.")
                return None
            command = DynamixelMotorsCommand(
                cmd=[
                    DynamixelMotorCommand(
                        id=ComponentId(id=self._id),
                        goal_position=FloatValue(value=self._outgoing_goal_position),
                    )
                ]
            )
            return command
        return None

    def _clean_outgoing_goal_positions(self) -> None:
        """Clean the outgoing goal positions."""
        self._outgoing_goal_position = None

    def send_goal_positions(self, check_positions: bool = False) -> None:
        """Send goal positions to the motor.

        If goal positions have been specified, sends them to the motor.
        Args :
            check_positions: A boolean indicating whether to check the positions after sending the command.
                Defaults to True.
        """
        command = self._get_goal_positions_message()
        if command is not None:
            self._clean_outgoing_goal_positions()
            self._stub.SendCommand(command)
            if check_positions:
                pass

    def set_speed_limits(self, speed_limit: float | int) -> None:
        """Set the speed limit as a percentage of the maximum speed the motor.

        Args:
            speed_limit: The desired speed limit as a percentage (0-100) of the maximum speed. Can be
                specified as a float or int.
        """
        if not isinstance(speed_limit, float | int):
            raise TypeError(f"Expected one of: float, int for speed_limit, got {type(speed_limit).__name__}")
        if not (0 <= speed_limit <= 100):
            raise ValueError(f"speed_limit must be in [0, 100], got {speed_limit}.")
        speed_limit = speed_limit / 100.0
        command = DynamixelMotorsCommand(
            cmd=[
                DynamixelMotorCommand(
                    id=ComponentId(id=self._id),
                    speed_limit=FloatValue(value=speed_limit),
                )
            ]
        )
        self._stub.SendCommand(command)

    def _update_with(self, new_state: DynamixelMotorState) -> None:
        """Update the present and goal positions of the joint with new state values.

        Args:
            new_state: A dictionary containing the new state values for the joint. The keys should include
                "present_position" and "goal_position", with corresponding FloatValue objects as values.
        """
        self._present_position = new_state.present_position.value
        self._goal_position = new_state.goal_position.value
        self._compliant = new_state.compliant.value
