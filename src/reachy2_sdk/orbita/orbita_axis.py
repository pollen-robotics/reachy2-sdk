"""Reachy OrbitaAxis module.

Handles all specific methods to OrbitaAxis.
"""

from typing import Dict

from google.protobuf.wrappers_pb2 import FloatValue

from .utils import to_position


class OrbitaAxis:
    """The OrbitaAxis class represents any Orbita3d or Orbita2d axis.

    The OrbitaAxis class is used to store the up-to-date state of the axis, especially:
    - its present speed (RO)
    - its present load (RO)
    """

    def __init__(self, initial_state: Dict[str, FloatValue]) -> None:
        """Initialize the axis with its initial state.

        Args:
            initial_state: A dictionary containing the initial state values for the axis. The keys should include
                "present_speed" and "present_load", with corresponding FloatValue objects as values.
        """
        self._update_with(initial_state)

    @property
    def present_speed(self) -> float:
        """Get the present speed of the axis in radians per second."""
        return to_position(self._present_speed)

    @property
    def present_load(self) -> float:
        """Get the present load of the axis in Newtons."""
        return float(self._present_load)

    def _update_with(self, new_state: Dict[str, FloatValue]) -> None:
        """Update the present speed and load of the axis with new state values.

        Args:
            new_state: A dictionary containing the new state values for the axis. The keys should include
                "present_speed" and "present_load", with corresponding FloatValue objects as values.
        """
        self._present_speed = new_state["present_speed"].value
        self._present_load = new_state["present_load"].value
