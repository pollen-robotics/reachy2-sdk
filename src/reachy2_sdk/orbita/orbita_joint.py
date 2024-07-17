"""This module describes Orbita2d and Orbita3d joints."""
from typing import Any, Dict, List

from google.protobuf.wrappers_pb2 import FloatValue

from .utils import to_internal_position, to_position, unwrapped_proto_value


class OrbitaJoint:
    """The OrbitaJoint class represents any Orbita2d or Orbita 3d joint.

    The OrbitaJoint class is used to store the up-to-date state of the joint, especially:
        - its present_position (RO)
        - its goal_position (RW)
    """

    def __init__(self, initial_state: Dict[str, FloatValue], axis_type: str, actuator: Any) -> None:
        self._actuator = actuator
        self._axis_type = axis_type

        self._present_position = unwrapped_proto_value(initial_state["present_position"])
        self._goal_position = unwrapped_proto_value(initial_state["goal_position"])

        self._register_needing_sync: List[str] = []

    def __repr__(self) -> str:
        repr_template = (
            '<OrbitaJoint axis_type="{axis_type}" present_position={present_position} goal_position={goal_position} >'
        )
        return repr_template.format(
            axis_type=self._axis_type,
            present_position=self.present_position,
            goal_position=self.goal_position,
        )

    @property
    def present_position(self) -> float:
        return to_position(self._present_position)

    @property
    def goal_position(self) -> float:
        return to_position(self._goal_position)

    @goal_position.setter
    def goal_position(self, value: float | int) -> None:
        if isinstance(value, float) | isinstance(value, int):
            self._actuator._set_outgoing_goal_position(self._axis_type, to_internal_position(value))
        else:
            raise TypeError("goal_position must be a float or int")

    def _update_with(self, new_state: Dict[str, FloatValue]) -> None:
        self._present_position = unwrapped_proto_value(new_state["present_position"])
        self._goal_position = unwrapped_proto_value(new_state["goal_position"])
