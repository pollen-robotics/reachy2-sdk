"""This module describes Orbita2d and Orbita3d joints."""
from typing import Any, Dict, List

from google.protobuf.wrappers_pb2 import FloatValue

from ..motors.register import Register
from .utils import to_internal_position, to_position


class OrbitaJoint:
    """The OrbitaJoint class represents any Orbita2d or Orbita 3d joint.

    The OrbitaJoint class is used to store the up-to-date state of the joint, especially:
        - its present_position (RO)
        - its goal_position (RW)
    """

    present_position = Register(
        readonly=True,
        type=FloatValue,
        label="present_position",
        conversion=(to_internal_position, to_position),
    )
    goal_position = Register(
        readonly=False,
        type=FloatValue,
        label="goal_position",
        conversion=(to_internal_position, to_position),
    )

    def __init__(self, initial_state: Dict[str, FloatValue], axis_type: str, actuator: Any) -> None:
        self._actuator = actuator
        self._axis_type = axis_type
        self._state = initial_state
        self._tmp_state = initial_state.copy()

        for field in dir(self):
            value = getattr(self, field)
            if isinstance(value, Register):
                value.label = field

        self._register_needing_sync: List[str] = []

    def __repr__(self) -> str:
        return f'<OrbitaJoint axis_type="{self._axis_type}" present_position={self.present_position} goal_position={self.goal_position} >'  # noqa: E501
