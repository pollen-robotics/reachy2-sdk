"""This module describes Orbita axes."""
from typing import Dict

from google.protobuf.wrappers_pb2 import FloatValue

from ..register import Register
from .utils import to_internal_position, to_position


class OrbitaAxis:
    """The OrbitaAxis class represents any Orbita3d or Orbita2d axis.

    The OrbitaAxis class is used to store the up-to-date state of the axis, especially:
        - its present speed (RO)
        - its present load (RO)
    """

    present_speed = Register(
        readonly=True,
        type=FloatValue,
        label="present_speed",
        conversion=(to_internal_position, to_position),
    )
    present_load = Register(readonly=True, type=FloatValue, label="present_load")

    def __init__(self, initial_state: Dict[str, FloatValue]) -> None:
        """Initialize the axis with its initial state."""
        self._state = initial_state

        for field in dir(self):
            value = getattr(self, field)
            if isinstance(value, Register):
                value.label = field
