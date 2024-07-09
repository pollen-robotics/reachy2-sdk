"""This module describes Orbita axes."""
from typing import Dict

from google.protobuf.wrappers_pb2 import FloatValue

from .utils import to_internal_position, to_position, unwrapped_proto_value


class OrbitaAxis:
    """The OrbitaAxis class represents any Orbita3d or Orbita2d axis.

    The OrbitaAxis class is used to store the up-to-date state of the axis, especially:
        - its present speed (RO)
        - its present load (RO)
    """

    def __init__(self, initial_state: Dict[str, FloatValue]) -> None:
        """Initialize the axis with its initial state."""

        self._present_speed = unwrapped_proto_value(initial_state['present_speed'])
        self._present_load = unwrapped_proto_value(initial_state['present_load'])

        self._state = initial_state

    @property
    def present_speed(self) -> float:
        return to_position(self._present_speed)

    @property
    def present_load(self) -> float:
        return self._present_load

    def _update_with(self, new_state: Dict[str, FloatValue]) -> None:
        self._present_speed = unwrapped_proto_value(new_state['present_speed'])
        self._present_load = unwrapped_proto_value(new_state['present_load'])
