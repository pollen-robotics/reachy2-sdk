"""Reachy Orbita module.

Handles all specific methods commmon to all Orbita2d and Orbita3d.
"""

import logging
import time
from abc import ABC, abstractmethod
from threading import Thread
from typing import Any, Dict, Optional, Tuple

import numpy as np
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.component_pb2 import ComponentId
from reachy2_sdk_api.orbita2d_pb2 import (
    Orbita2dCommand,
    Orbita2dsCommand,
    Orbita2dState,
    Orbita2dStatus,
)
from reachy2_sdk_api.orbita2d_pb2_grpc import Orbita2dServiceStub
from reachy2_sdk_api.orbita3d_pb2 import Orbita3dState, Orbita3dStatus
from reachy2_sdk_api.orbita3d_pb2_grpc import Orbita3dServiceStub

from ..parts.part import Part
from .orbita_axis import OrbitaAxis
from .orbita_motor import OrbitaMotor


class Orbita(ABC):
    """The Orbita class is an abstract class to represent any Orbita actuator.

    The Orbita class is used to store the up-to-date state of the actuator, especially:
    - its compliancy
    - its joints state
    - its motors state
    - its axis state

    And apply speed, torque, pid and compliancy to all motors of the actuator.

    This class is meant to be derived by Orbita2d and Orbita3d
    """

    def __init__(self, uid: int, name: str, orbita_type: str, stub: Orbita2dServiceStub | Orbita3dServiceStub, part: Part):
        """Initialize the Orbita actuator with its common attributes.

        Args:
            uid: The unique identifier for the actuator.
            name: The name of the actuator.
            orbita_type: Specifies the type of Orbita, either "2d" or "3d".
            stub: The gRPC stub used for communicating with the actuator, which can be an
                instance of either `Orbita2dServiceStub` or `Orbita3dServiceStub`.
            part: The parent part to which the Orbita belongs, used for referencing the
                part's attributes.
        """
        self._logger = logging.getLogger(__name__)
        self._name = name
        self._id = uid
        self._orbita_type = orbita_type
        self._stub = stub
        self._part = part

        self._compliant: bool

        self._joints: Dict[str, Any] = {}
        self._axis_name_by_joint: Dict[Any, str] = {}
        self._motors: Dict[str, OrbitaMotor] = {}
        self._outgoing_goal_positions: Dict[str, float] = {}
        self._axis: Dict[str, OrbitaAxis] = {}

        self._error_status: Optional[str] = None

        self._thread_check_position: Optional[Thread] = None
        self._cancel_check = False

    @abstractmethod
    def _create_dict_state(self, initial_state: Orbita2dState | Orbita3dState) -> Dict[str, Dict[str, FloatValue]]:
        """Create a dictionary representation of the joint states.

        This method must be implemented by subclasses to create the dict state of the Orbita.

        Args:
            initial_state: The initial state of the Orbita, provided as an instance of
                `Orbita2dState` or `Orbita3dState`.

        Returns:
            A dictionary where each key corresponds to a joint attribute, and each value
            is another dictionary of state information with string keys and `FloatValue` values.
        """
        pass

    def __repr__(self) -> str:
        """Clean representation of an Orbita."""
        s = "\n\t".join([str(joint) for joint in self._joints.values()])
        return f"""<Orbita{self._orbita_type} on={self.is_on()} joints=\n\t{
            s
        }\n>"""

    @abstractmethod
    def set_speed_limits(self, speed_limit: float | int) -> None:
        """Set the speed limits for the Orbita actuator.

        This method defines the maximum speed for the joints, specified as a percentage
        of the maximum speed capability.

        Args:
            speed_limit: The desired speed limit as a percentage (0-100).

        Raises:
            TypeError: If the provided speed_limit is not a float or int.
            ValueError: If the provided speed_limit is outside the range [0, 100].
        """
        if not isinstance(speed_limit, float | int):
            raise TypeError(f"Expected one of: float, int for speed_limit, got {type(speed_limit).__name__}")
        if not (0 <= speed_limit <= 100):
            raise ValueError(f"speed_limit must be in [0, 100], got {speed_limit}.")

    @abstractmethod
    def set_torque_limits(self, torque_limit: float | int) -> None:
        """Set the torque limits for the Orbita actuator.

        This method defines the maximum torque for the joints, specified as a percentage
        of the maximum torque capability.

        Args:
            torque_limit: The desired torque limit as a percentage (0-100).

        Raises:
            TypeError: If the provided torque_limit is not a float or int.
            ValueError: If the provided torque_limit is outside the range [0, 100].
        """
        if not isinstance(torque_limit, float | int):
            raise TypeError(f"Expected one of: float, int for torque_limit, got {type(torque_limit).__name__}")
        if not (0 <= torque_limit <= 100):
            raise ValueError(f"torque_limit must be in [0, 100], got {torque_limit}.")

    # def set_pid(self, pid: Tuple[float, float, float]) -> None:
    #     """Set a pid value on all motors of the actuator"""
    #     if isinstance(pid, tuple) and len(pid) == 3 and all(isinstance(n, float | int) for n in pid):
    #         for m in self._motors.values():
    #             m._tmp_pid = pid
    #         self._update_loop("pid")
    #     else:
    #         raise ValueError("pid should be of type Tuple[float, float, float]")

    def get_speed_limits(self) -> Dict[str, float]:
        """Get the speed limits for all motors of the actuator.

        The speed limits are expressed as percentages of the maximum speed for each motor.

        Returns:
            A dictionary where each key is the motor name and the value is the speed limit
            percentage (0-100) for that motor. Motor names are of format "motor_{n}".
        """
        return {motor_name: m.speed_limit for motor_name, m in self._motors.items()}

    def get_torque_limits(self) -> Dict[str, float]:
        """Get the torque limits for all motors of the actuator.

        The torque limits are expressed as percentages of the maximum torque for each motor.

        Returns:
            A dictionary where each key is the motor name and the value is the torque limit
            percentage (0-100) for that motor. Motor names are of format "motor_{n}".
        """
        return {motor_name: m.torque_limit for motor_name, m in self._motors.items()}

    def get_pids(self) -> Dict[str, Tuple[float, float, float]]:
        """Get the PID values for all motors of the actuator.

        Each motor's PID controller parameters (Proportional, Integral, Derivative) are returned.

        Returns:
            A dictionary where each key is the motor name and the value is a tuple containing
            the PID values (P, I, D) for that motor. Motor names are of format "motor_{n}".
        """
        return {motor_name: m.pid for motor_name, m in self._motors.items()}

    def turn_on(self) -> None:
        """Turn on all motors of the actuator."""
        self._set_compliant(False)

    def turn_off(self) -> None:
        """Turn off all motors of the actuator."""
        self._set_compliant(True)

    def is_on(self) -> bool:
        """Check if the actuator is currently stiff.

        Returns:
            `True` if the actuator is stiff (not compliant), `False` otherwise.
        """
        return not self._compliant

    @property
    def temperatures(self) -> Dict[str, float]:
        """Get the current temperatures of all the motors in the actuator.

        Returns:
            A dictionary where each key is the motor name and the value is the
            current temperature of the motor in degrees Celsius. Motor names are of format "motor_{n}".
        """
        return {motor_name: m.temperature for motor_name, m in self._motors.items()}

    def _set_compliant(self, compliant: bool) -> None:
        """Set the compliance mode of the actuator's motors.

        Compliance mode determines whether the motors are stiff or compliant.

        Args:
            compliant: A boolean value indicating whether to set the motors to
                compliant (`True`) or stiff (`False`).
        """
        command = Orbita2dsCommand(
            cmd=[
                Orbita2dCommand(
                    id=ComponentId(id=self._id),
                    compliant=BoolValue(value=compliant),
                )
            ]
        )
        self._stub.SendCommand(command)

    def _set_outgoing_goal_position(self, axis_name: str, goal_position: float) -> None:
        """Set the goal position for a specified axis.

        This method sets the target position for an axis, preparing it to be sent as a goal position.

        Args:
            axis_name: The name of the axis for which to set the goal position. Could be "roll", "pitch", "yaw" for Orbita3d
                or "axis_1", "axis_2" for Orbita2d.
            goal_position: The desired goal position converted in radians for the specified axis.
        """
        joint = getattr(self, axis_name)
        axis = self._axis_name_by_joint[joint]
        self._outgoing_goal_positions[axis] = goal_position

    @abstractmethod
    def send_goal_positions(self) -> None:
        """Send the goal positions to the actuator.

        This method is abstract and should be implemented in derived classes to
        send the specified goal positions to the actuator's joints.
        """
        pass

    def _post_send_goal_positions(self) -> None:
        """Start a background thread to check the goal positions after sending them.

        This method stops any ongoing position check thread and starts a new thread
        to monitor the current positions of the joints relative to their last goal positions.
        """
        self._cancel_check = True
        if self._thread_check_position is not None and self._thread_check_position.is_alive():
            self._thread_check_position.join()
        self._thread_check_position = Thread(target=self._check_goal_positions, daemon=True)
        self._thread_check_position.start()

    def _check_goal_positions(self) -> None:
        """Monitor the joint positions to check if they reach the specified goals.

        This method checks the current positions of the joints and compares them to
        the goal positions. If a position is significantly different from the goal after 1 second,
        a warning is logged indicating that the position may be unreachable.
        """
        self._cancel_check = False
        t1 = time.time()
        while time.time() - t1 < 1:
            time.sleep(0.05)
            if self._cancel_check:
                # in case of multiple send_goal_positions we'll check the next call
                return

        for joint, orbitajoint in self._joints.items():
            # precision is low we are looking for unreachable positions
            if not np.isclose(orbitajoint.present_position, orbitajoint.goal_position, atol=1):
                self._logger.warning(
                    f"required goal position for {self._name}.{joint} is unreachable."
                    f" current position is ({orbitajoint.present_position}"
                )

    def _update_with(self, new_state: Orbita2dState | Orbita3dState) -> None:
        """Update the actuator's state with new data.

        This method updates the internal state of the motors, axes, and joints based on
        the new state data received.

        Args:
            new_state: The new state of the actuator, either as an Orbita2dState or
                Orbita3dState object.
        """
        state: Dict[str, Dict[str, FloatValue]] = self._create_dict_state(new_state)

        for name, motor in self._motors.items():
            motor._update_with(state[name])

        for name, axis in self._axis.items():
            axis._update_with(state[name])

        for name, joints in self._joints.items():
            joints._update_with(state[name])

    @property
    def audit(self) -> Optional[str]:
        """Get the current audit status of the actuator.

        Returns:
            The audit status as a string, representing the latest error or status
            message, or `None` if there is no error.
        """
        return self._error_status

    def _update_audit_status(self, new_status: Orbita2dStatus | Orbita3dStatus) -> None:
        """Update the audit status based on the new status data.

        Args:
            new_status: The new status data, either as an Orbita2dStatus or
                Orbita3dStatus object, containing error details.
        """
        self._error_status = new_status.errors[0].details
