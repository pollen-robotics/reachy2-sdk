"""Reachy GripperJoint module.

Handles all specific methods to a GripperJoint.
"""

import logging
from collections import deque
from typing import Deque, Optional

import numpy as np
from reachy2_sdk_api.hand_pb2 import HandState

from ..orbita.utils import to_internal_position, to_position


class GripperJoint:
    """Class to represent any gripper's joint of the robot.

    The `GripperJoint` class provides methods to get output values of the joint, such as the opening,
    present position, and goal position. It also allows setting the goal position and checking the
    joint's movement status.

    Attributes:
        opening: The opening of the joint as a percentage (0-100), rounded to two decimal places.
        present_position: The current position of the joint in degrees.
        goal_position: The target goal position of the joint in degrees.
    """

    def __init__(
        self,
        initial_state: HandState,
    ):
        """Initialize the GripperJoint with its initial state.

        This sets up the joint by assigning its state based on the provided initial values.

        Args:
            initial_state: A HandState containing the initial state of the joint.
        """
        self._logger = logging.getLogger(__name__)

        self._is_moving = False
        self._nb_steps_to_ignore = 0
        self._steps_ignored = 0
        self._last_present_positions_queue_size = 10
        self._last_present_positions: Deque[float] = deque(maxlen=self._last_present_positions_queue_size)

        self._update_with(initial_state)
        self._last_present_positions.append(self._present_position)
        self._outgoing_goal_positions: Optional[float] = None

    def __repr__(self) -> str:
        """Clean representation of a GripperJoint."""
        repr_template = "<GripperJoint on={is_on} present_position={present_position} goal_position={goal_position} >"
        return repr_template.format(
            is_on=self.is_on(),
            present_position=round(self.present_position, 2),
            goal_position=round(self.goal_position, 2),
        )

    @property
    def opening(self) -> float:
        """Get the opening of the joint as a percentage.

        Returns:
            The joint opening as a percentage (0-100), rounded to two decimal places.
        """
        return float(round(self._opening * 100, 2))

    @property
    def present_position(self) -> float:
        """Get the current position of the joint.

        Returns:
            The present position of the joint in degrees.
        """
        return to_position(self._present_position)

    @property
    def goal_position(self) -> float:
        """Get the goal position of the joint.

        Returns:
            The goal position of the joint in degrees.
        """
        return to_position(self._goal_position)

    @goal_position.setter
    def goal_position(self, value: float | int) -> None:
        """Set the goal position for the joint.

        Args:
            value: The goal position to set, specified as a float or int.

        Raises:
            TypeError: If the provided value is not a float or int.
        """
        if isinstance(value, float) | isinstance(value, int):
            self._outgoing_goal_positions = to_internal_position(value)
        else:
            raise TypeError("goal_position must be a float or int")

    def is_on(self) -> bool:
        """Check if the joint is stiff.

        Returns:
            `True` if the joint is on (not compliant), `False` otherwise.
        """
        return not self._compliant

    def is_off(self) -> bool:
        """Check if the joint is compliant.

        Returns:
            `True` if the joint is off (compliant), `False` otherwise.
        """
        return bool(self._compliant)

    def is_moving(self) -> bool:
        """Check if the joint is currently moving.

        Returns:
            `True` if the joint is moving, `False` otherwise.
        """
        return self._is_moving

    def _check_joint_movement(self) -> None:
        """Check if the joint is still moving based on the present position.

        This method updates the movement status by comparing the current position to the last few positions.
        If the position has not changed significantly, the joint is considered to have stopped moving.
        """
        present_position = self._present_position
        if (
            len(self._last_present_positions) >= self._last_present_positions_queue_size
            and np.isclose(present_position, self._last_present_positions[-1], np.deg2rad(0.1))
            and np.isclose(present_position, self._last_present_positions[-2], np.deg2rad(0.1))
        ):
            self._is_moving = False
            self._last_present_positions.clear()
        self._last_present_positions.append(present_position)

    def _update_with(self, new_state: HandState) -> None:
        """Update the joint with a newly received (partial) state from the gRPC server.

        This method updates the present position, goal position, opening, and compliance status.
        It also checks if the joint is still moving based on the new state.

        Args:
            new_state: A HandState object representing the new state of the hand.
        """
        self._present_position = new_state.present_position.parallel_gripper.position.value
        self._goal_position = new_state.goal_position.parallel_gripper.position.value
        self._opening = new_state.opening.value
        self._compliant = new_state.compliant.value
        if self._is_moving:
            self._check_joint_movement()
