"""Reachy OrbitaJoint module.

Handles all specific methods to OrbitaJoint.
"""

from typing import Any, Dict

from google.protobuf.wrappers_pb2 import FloatValue
from reachy2_sdk_api.goto_pb2 import GoToId

from .utils import to_internal_position, to_position


class OrbitaJoint:
    """The OrbitaJoint class represents any Orbita2d or Orbita3d joint.

    The OrbitaJoint class is used to store the up-to-date state of the joint, especially:
        - its present_position (RO)
        - its goal_position (RW)
    """

    def __init__(
        self,
        initial_state: Dict[str, FloatValue],
        axis_type: str,
        actuator: Any,
        position_order_in_part: int,
    ) -> None:
        """Initialize the OrbitaJoint with its initial state and configuration.

        This sets up the joint by assigning its actuator, axis type, and position order within
        the part, and updates its state based on the provided initial values.

        Args:
            initial_state: A dictionary containing the initial state of the joint, with
                each entry representing a specific parameter of the joint (e.g., present position).
            axis_type: The type of axis for the joint (e.g., roll, pitch, yaw).
            actuator: The actuator to which this joint belongs.
            position_order_in_part: The position order of this joint in the overall part's
                list of joints.
        """
        self._actuator = actuator
        self._axis_type = axis_type
        self._position_order_in_part = position_order_in_part

        self._update_with(initial_state)

    def __repr__(self) -> str:
        """Clean representation of the OrbitaJoint."""
        repr_template = (
            '<OrbitaJoint axis_type="{axis_type}" present_position={present_position} goal_position={goal_position} >'
        )
        return repr_template.format(
            axis_type=self._axis_type,
            present_position=round(self.present_position, 2),
            goal_position=round(self.goal_position, 2),
        )

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
        if isinstance(value, float) | isinstance(value, int):
            self._actuator._set_outgoing_goal_position(self._axis_type, to_internal_position(value))
        else:
            raise TypeError("goal_position must be a float or int")

    def goto(
        self,
        goal_position: float,
        duration: float = 2,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
    ) -> GoToId:
        """Send the joint to the specified goal position within a given duration.

        Acts like a "goto" movement on the part, where "goto" movements for joints are queued on the part they belong to.

        Args:
            goal_position: The target position to move the joint to.
            duration: The time in seconds for the joint to reach the goal position. Defaults to 2.
            wait: Whether to wait for the movement to finish before continuing. Defaults to False.
            interpolation_mode: The type of interpolation to use for the movement, either "minimum_jerk" or "linear".
                Defaults to "minimum_jerk".
            degrees: Whether the goal position is specified in degrees. If True, the position is interpreted as degrees.
                Defaults to True.

        Returns:
            The GoToId associated with the movement command.
        """
        return self._actuator._part._goto_single_joint(
            self._position_order_in_part,
            goal_position=goal_position,
            duration=duration,
            wait=wait,
            interpolation_mode=interpolation_mode,
            degrees=degrees,
        )

    def _update_with(self, new_state: Dict[str, FloatValue]) -> None:
        """Update the present and goal positions of the joint with new state values.

        Args:
            new_state: A dictionary containing the new state values for the joint. The keys should include
                "present_position" and "goal_position", with corresponding FloatValue objects as values.
        """
        self._present_position = new_state["present_position"].value
        self._goal_position = new_state["goal_position"].value
