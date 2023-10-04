from grpc import Channel

from .register import Register

from typing import Dict

from reachy_sdk_api_v2.orbita2d_pb2 import (
    Axis,
    Orbita2DState,
    Float2D,
)

from reachy_sdk_api_v2.orbita2d_pb2_grpc import Orbita2DServiceStub

from .orbita_utils import OrbitaJoint


class Orbita2d:
    compliant = Register(readonly=False, label="compliant")

    def __init__(self, name: str, axis1: Axis, axis2: Axis, initial_state: Orbita2DState, grpc_channel: Channel):
        self.name = name
        self._stub = Orbita2DServiceStub(grpc_channel)

        axis1_name = Axis.DESCRIPTOR.values_by_number[axis1].name.lower()
        axis2_name = Axis.DESCRIPTOR.values_by_number[axis2].name.lower()

        self._axis1 = axis1_name
        self._axis2 = axis2_name

        self._axis_to_name: Dict[str, str] = {"axis_1": self._axis1, "axis_2": self._axis2}

        init_state = {
            "present_position": 20.0,
            "present_speed": 0.0,
            "present_load": 0.0,
            "temperature": 0.0,
            "goal_position": 100.0,
            "speed_limit": 0.0,
            "torque_limit": 0.0,
        }

        self._state: Dict[str, bool] = {}

        # TODO get initial state from grpc server
        setattr(self, axis1_name, OrbitaJoint(initial_state=init_state.copy(), axis_type=axis1))
        setattr(self, axis2_name, OrbitaJoint(initial_state=init_state.copy(), axis_type=axis2))

        self.compliant = False

    def _update_with(self, new_state: Orbita2DState) -> None:
        """Update the orbita with a newly received (partial) state received from the gRPC server."""
        for field, value in new_state.ListFields():
            if field.name == "compliant":
                self._state[field.name] = value.value
            else:
                if isinstance(value, Float2D):
                    for axis, val in value.ListFields():
                        joint = getattr(self, self._axis_to_name[axis.name])
                        joint._state[field.name] = val
