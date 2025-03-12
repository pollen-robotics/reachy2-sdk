"""Reachy Hand module.

Handles all specific method to a Hand.
"""

from abc import abstractmethod
from typing import Dict, Optional

import grpc
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.hand_pb2 import Hand as Hand_proto
from reachy2_sdk_api.hand_pb2 import HandStatus
from reachy2_sdk_api.hand_pb2_grpc import HandServiceStub

from ..grippers.gripper_joint import GripperJoint
from .goto_based_part import IGoToBasedPart
from .part import Part


class Hand(Part, IGoToBasedPart):
    """Class for controlling the Reachy's hand.

    The `Hand` class provides methods to control the gripper of Reachy, including opening and closing
    the hand, setting the goal position, and checking the hand's state. It also manages the hand's
    compliance status (whether it is stiff or free).

    Attributes:
        opening: The opening of the hand as a percentage (0-100), rounded to two decimal places.
        present_position: The current position of the hand in degrees.
        goal_position: The target goal position of the hand in degrees.
    """

    def __init__(
        self,
        hand_msg: Hand_proto,
        grpc_channel: grpc.Channel,
        goto_stub: GoToServiceStub,
    ) -> None:
        """Initialize the Hand component.

        Sets up the necessary attributes and configuration for the hand, including the gRPC
        stub and initial state.

        Args:
            hand_msg: The Hand_proto object containing the configuration details for the hand.
            initial_state: The initial state of the hand, represented as a HandState object.
            grpc_channel: The gRPC channel used to communicate with the hand's gRPC service.
            goto_stub: The gRPC stub for controlling goto movements.
        """
        super().__init__(hand_msg, grpc_channel, HandServiceStub(grpc_channel))
        IGoToBasedPart.__init__(self, self._part_id, goto_stub)
        self._stub = HandServiceStub(grpc_channel)

        self._last_goto_checked: Optional[int] = None
        self._joints: Dict[str, GripperJoint] = {}

    def _set_speed_limits(self, value: int) -> None:
        """Set the speed limits for the hand.

        Args:
            value: The speed limit value to be set, as a percentage (0-100) of the maximum allowed speed,
                represented as an integer.
        """
        return super()._set_speed_limits(value)

    def is_on(self) -> bool:
        """Check if the hand is stiff.

        Returns:
            `True` if the hand is on (not compliant), `False` otherwise.
        """
        for j in self._joints.values():
            if not j.is_on():
                return False
        return True

    def is_off(self) -> bool:
        """Check if the hand is compliant.

        Returns:
            `True` if the hand is off (compliant), `False` otherwise.
        """
        for j in self._joints.values():
            if j.is_on():
                return False
        return True

    def is_moving(self) -> bool:
        """Check if the hand is currently moving.

        Returns:
            `True` if the gripper is moving, `False` otherwise.
        """
        goto_playing = self.get_goto_playing()
        if goto_playing.id != -1 and goto_playing.id != self._last_goto_checked:
            self._last_goto_checked = goto_playing.id
            for j in self._joints.values():
                j._is_moving = True
                j._check_joint_movement()
        for j in self._joints.values():
            if j.is_moving():
                return True
        return False

    def open(self) -> None:
        """Open the hand.

        Raises:
            RuntimeError: If the gripper is off and the open request cannot be sent.
        """
        if not self.is_on():
            raise RuntimeError("Gripper is off. Open request not sent.")
        self._stub.OpenHand(self._part_id)
        for j in self._joints.values():
            j._is_moving = True

    def close(self) -> None:
        """Close the hand.

        Raises:
            RuntimeError: If the gripper is off and the close request cannot be sent.
        """
        if not self.is_on():
            raise RuntimeError("Gripper is off. Close request not sent.")
        self._stub.CloseHand(self._part_id)
        for j in self._joints.values():
            j._is_moving = True

    @abstractmethod
    def send_goal_positions(self, check_positions: bool = True) -> None:
        """Send the goal position to the hand actuator.

        If any goal position has been specified to the gripper, sends them to the robot.
        If the hand is off, the command is not sent.

        Args :
            check_positions: A boolean indicating whether to check the positions after sending the command.
                Defaults to True.
        """

    def _update_audit_status(self, new_status: HandStatus) -> None:
        """Update the audit status with the new status received from the gRPC server.

        Args:
            new_status: A HandStatus object representing the new status of the hand.
        """
        pass  # pragma: no cover
