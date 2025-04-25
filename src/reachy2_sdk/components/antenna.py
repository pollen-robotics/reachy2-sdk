"""Reachy Antenna module.

Handles all specific methods to Antennas.
"""

import logging
import time
from threading import Thread
from typing import Any, Dict, List, Optional

import numpy as np
from google.protobuf.wrappers_pb2 import FloatValue
from grpc import Channel
from reachy2_sdk_api.component_pb2 import ComponentId
from reachy2_sdk_api.dynamixel_motor_pb2 import DynamixelMotor as DynamixelMotor_proto
from reachy2_sdk_api.dynamixel_motor_pb2 import (
    DynamixelMotorsCommand,
    DynamixelMotorState,
    DynamixelMotorStatus,
)
from reachy2_sdk_api.goto_pb2 import GoToId, GoToRequest, JointsGoal
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.head_pb2 import AntennaJointGoal

from ..dynamixel.dynamixel_motor import DynamixelMotor
from ..parts.part import Part
from ..utils.utils import get_grpc_interpolation_mode
from .goto_based_component import IGoToBasedComponent


class Antenna(IGoToBasedComponent):
    """The Antenna class represents any antenna of the robot's head."""

    def __init__(
        self,
        uid: int,
        name: str,
        initial_state: DynamixelMotorState,
        grpc_channel: Channel,
        goto_stub: GoToServiceStub,
        part: Part,
    ):
        """Initialize the Antenna with its initial state and configuration.

        Args:
            uid: The unique identifier of the component.
            name: The name of the joint.
            initial_state: A dictionary containing the initial state of the joint, with
                each entry representing a specific parameter of the joint (e.g., present position).
            grpc_channel: The gRPC channel used to communicate with the DynamixelMotor service.
            goto_stub: The gRPC stub for controlling goto movements.
            part: The part to which this joint belongs.
        """
        self._logger = logging.getLogger(__name__)
        IGoToBasedComponent.__init__(self, ComponentId(id=uid, name=name), goto_stub)
        self._part = part
        self._error_status: Optional[str] = None
        self._joints: Dict[str, DynamixelMotor] = {}
        if name == "antenna_left":
            self._name = "l_antenna"
        else:
            self._name = "r_antenna"
        self._joints[self._name] = DynamixelMotor(uid, name, initial_state, grpc_channel)

        self._thread_check_position: Optional[Thread] = None
        self._cancel_check = False

    def _check_goto_parameters(self, target: Any, duration: Optional[float], q0: Optional[List[float]] = None) -> None:
        """Check the validity of the parameters for the `goto` method.

        Args:
            duration: The time in seconds for the movement to be completed.
            target: The target position, either a float or int.

        Raises:
            TypeError: If the target is not a list or a quaternion.
            ValueError: If the target list has a length other than 3.
            ValueError: If the duration is set to 0.
        """
        if not (isinstance(target, float) or isinstance(target, int)):
            raise TypeError(f"Antenna's goto target must be either a float or int, got {type(target)}.")

        elif duration == 0:
            raise ValueError("duration cannot be set to 0.")
        elif duration is not None and duration < 0:
            raise ValueError("duration cannot be negative.")

    def goto_posture(
        self,
        common_posture: str = "default",
        duration: float = 2,
        wait: bool = False,
        wait_for_goto_end: bool = True,
        interpolation_mode: str = "minimum_jerk",
    ) -> GoToId:
        """Send the antenna to standard positions within the specified duration.

        The default posture sets the antenna is 0.0.

        Args:
            common_posture: The standard positions to which all joints will be sent.
                It can be 'default' or 'elbow_90'. Defaults to 'default'.
            duration: The time duration in seconds for the robot to move to the specified posture.
                Defaults to 2.
            wait: Determines whether the program should wait for the movement to finish before
                returning. If set to `True`, the program waits for the movement to complete before continuing
                execution. Defaults to `False`.
            wait_for_goto_end: Specifies whether commands will be sent to a part immediately or
                only after all previous commands in the queue have been executed. If set to `False`, the program
                will cancel all executing moves and queues. Defaults to `True`.
            interpolation_mode: The type of interpolation used when moving the arm's joints.
                Can be 'minimum_jerk' or 'linear'. Defaults to 'minimum_jerk'.

        Returns:
            The unique GoToId associated with the movement command.
        """
        if not wait_for_goto_end:
            self.cancel_all_goto()
        if self.is_on():
            return self.goto(0, duration, wait, interpolation_mode)
        else:
            self._logger.warning(f"{self._name} is off. No command sent.")
        return GoToId(id=-1)

    def goto(
        self,
        target: float,
        duration: float = 2.0,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
    ) -> GoToId:
        """Send the antenna to a specified goal position.

        Args:
            target: The desired goal position for the antenna.
            duration: The time in seconds for the movement to be completed. Defaults to 2.
            wait: If True, the function waits until the movement is completed before returning.
                    Defaults to False.
            interpolation_mode: The interpolation method to be used. It can be either "minimum_jerk"
                    or "linear". Defaults to "minimum_jerk".
            degrees: If True, the joint value in the `target` argument is treated as degrees.
                    Defaults to True.

        Raises:
            TypeError : If the input type for `target` is invalid
            ValueError: If the `duration` is set to 0.

        Returns:
            GoToId: The unique identifier for the movement command.
        """
        if not self.is_on():
            self._logger.warning(f"head.{self._name} is off. No command sent.")
            return GoToId(id=-1)

        self._check_goto_parameters(target, duration)

        if degrees:
            target = np.deg2rad(target)

        request = GoToRequest(
            joints_goal=JointsGoal(
                antenna_joint_goal=AntennaJointGoal(
                    id=self._part._part_id,
                    antenna=DynamixelMotor_proto(
                        id=ComponentId(id=self._joints[self._name]._id, name=self._joints[self._name]._name),
                    ),
                    joint_goal=FloatValue(value=target),
                    duration=FloatValue(value=duration),
                )
            ),
            interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
        )

        response = self._goto_stub.GoToJoints(request)

        if response.id == -1:
            self._logger.error(f"Position {target} was not reachable. No command sent.")
        elif wait:
            self._wait_goto(response, duration)
        return response

    def __repr__(self) -> str:
        """Clean representation of the Antenna only joint (DynamixelMotor)."""
        s = "\n\t".join([str(joint) for joint in self._joints.values()])
        return f"""<Antenna on={self.is_on()} joints=\n\t{
            s
        }\n>"""

    def turn_on(self) -> bool:
        """Turn on the antenna's motor.

        Returns:
            `True` if it succeeded. 'False' otherwise.
        """
        self._joints[self._name].turn_on()
        time.sleep(0.12)
        return self.is_on()

    def turn_off(self) -> bool:
        """Turn off the antenna's motor.

        Returns:
            `True` if it succeeded. 'False' otherwise.
        """
        self._joints[self._name].turn_off()
        time.sleep(0.12)
        return self.is_off()

    def is_on(self) -> bool:
        """Check if the antenna is currently stiff.

        Returns:
            `True` if the antenna's motor is stiff (not compliant), `False` otherwise.
        """
        return bool(self._joints[self._name].is_on())

    def is_off(self) -> bool:
        """Check if the antenna is currently stiff.

        Returns:
            `True` if the antenna's motor is stiff (not compliant), `False` otherwise.
        """
        return not bool(self._joints[self._name].is_on())

    @property
    def present_position(self) -> float:
        """Get the present position of the joint in degrees."""
        return float(self._joints[self._name].present_position)

    @property
    def goal_position(self) -> float:
        """Get the goal position of the joint in degrees."""
        return float(self._joints[self._name].goal_position)

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
        self._joints[self._name].goal_position = value

    def _get_goal_positions_message(self) -> Optional[DynamixelMotorsCommand]:
        """Get the Orbita2dsCommand message to send the goal positions to the actuator."""
        return self._joints[self._name]._get_goal_positions_message()

    def _clean_outgoing_goal_positions(self) -> None:
        """Clean the outgoing goal positions."""
        self._joints[self._name]._clean_outgoing_goal_positions()

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
            time.sleep(0.0001)
            if self._cancel_check:
                # in case of multiple send_goal_positions we'll check the next call
                return

        # precision is low we are looking for unreachable positions
        if not np.isclose(self._joints[self._name].present_position, self._joints[self._name].goal_position, atol=1):
            self._logger.warning(
                f"Required goal position ({round(self._joints[self._name].goal_position, 2)}) for {self._name} is unreachable."
                f"\nCurrent position is ({round(self._joints[self._name].present_position, 2)})."
            )

    def send_goal_positions(self, check_positions: bool = False) -> None:
        """Send goal positions to the motor.

        If goal positions have been specified, sends them to the motor.
        Args :
            check_positions: A boolean indicating whether to check the positions after sending the command.
                Defaults to True.
        """
        if self._joints[self._name]._outgoing_goal_position is not None:
            if self.is_off():
                self._logger.warning(f"{self._name} is off. Command not sent.")
                return
            self._joints[self._name].send_goal_positions(check_positions)

    def set_speed_limits(self, speed_limit: float | int) -> None:
        """Set the speed limit as a percentage of the maximum speed the motor.

        Args:
            speed_limit: The desired speed limit as a percentage (0-100) of the maximum speed. Can be
                specified as a float or int.
        """
        self._joints[self._name].set_speed_limits(speed_limit)

    def _update_with(self, new_state: DynamixelMotorState) -> None:
        """Update the present and goal positions of the joint with new state values.

        Args:
            new_state: A dictionary containing the new state values for the joint. The keys should include
                "present_position" and "goal_position", with corresponding FloatValue objects as values.
        """
        self._joints[self._name]._update_with(new_state)

    @property
    def status(self) -> Optional[str]:
        """Get the current audit status of the actuator.

        Returns:
            The audit status as a string, representing the latest error or status
            message, or `None` if there is no error.
        """
        pass

    def _update_audit_status(self, new_status: DynamixelMotorStatus) -> None:
        """Update the audit status based on the new status data.

        Args:
            new_status: The new status data, as a DynamixelMotorStatus object, containing error details.
        """
        pass
