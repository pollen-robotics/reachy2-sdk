"""This module describes Orbita2d and Orbita3d joints."""
from typing import Any, Dict

import numpy as np
from google.protobuf.wrappers_pb2 import FloatValue
from reachy2_sdk_api.arm_pb2 import ArmJointOrder
from reachy2_sdk_api.goto_pb2 import GoToId
from reachy2_sdk_api.head_pb2 import NeckJointOrder

from .utils import to_internal_position, to_position


class OrbitaJoint:
    """The OrbitaJoint class represents any Orbita2d or Orbita 3d joint.

    The OrbitaJoint class is used to store the up-to-date state of the joint, especially:
        - its present_position (RO)
        - its goal_position (RW)
    """

    def __init__(
        self,
        initial_state: Dict[str, FloatValue],
        axis_type: str,
        actuator: Any,
        position_order_in_part: ArmJointOrder | NeckJointOrder,
    ) -> None:
        self._actuator = actuator
        self._axis_type = axis_type
        self._position_order_in_part = position_order_in_part

        self._update_with(initial_state)

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

    def goto(
        self, goal_position: float, duration: float = 2, interpolation_mode: str = "minimum_jerk", degrees: bool = True
    ) -> GoToId:
        if degrees:
            goal_position = np.deg2rad(goal_position)
        return self._actuator._part._goto_single_joint(
            self._position_order_in_part, goal_position, duration, interpolation_mode, degrees
        )

    def _update_with(self, new_state: Dict[str, FloatValue]) -> None:
        self._present_position = new_state["present_position"].value
        self._goal_position = new_state["goal_position"].value
