from grpc import Channel

from google.protobuf.wrappers_pb2 import BoolValue

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
    compliant = Register(readonly=False, type=BoolValue, label="compliant")

    def __init__(self, uid: int, name: str, axis1: Axis, axis2: Axis, initial_state: Orbita2DState, grpc_channel: Channel):
        self.name = name
        self.id = uid
        self._stub = Orbita2DServiceStub(grpc_channel)

        axis1_name = Axis.DESCRIPTOR.values_by_number[axis1].name.lower()
        axis2_name = Axis.DESCRIPTOR.values_by_number[axis2].name.lower()

        self._axis1 = axis1_name
        self._axis2 = axis2_name

        self._axis_to_name: Dict[str, str] = {"axis_1": self._axis1, "axis_2": self._axis2}

        self._state: Dict[str, bool] = {}
        init_state: Dict[str, Dict[str, float]] = {}

        for field, value in initial_state.ListFields():
            if field.name == "compliant":
                self._state[field.name] = value.value
            else:
                if isinstance(value, Float2D):
                    for axis, val in value.ListFields():
                        if axis.name not in init_state:
                            init_state[axis.name] = {}
                        init_state[axis.name][field.name] = val

        setattr(self, axis1_name, OrbitaJoint(initial_state=init_state["axis_1"], axis_type=axis1))
        setattr(self, axis2_name, OrbitaJoint(initial_state=init_state["axis_2"], axis_type=axis2))

    def _update_with(self, new_state: Orbita2DState) -> None:
        """Update the orbita with a newly received (partial) state received from the gRPC server."""
        for field, value in new_state.ListFields():
            if field.name == "compliant":
                self._state[field.name] = value
            else:
                if isinstance(value, Float2D):
                    for axis, val in value.ListFields():
                        joint = getattr(self, self._axis_to_name[axis.name])
                        joint._state[field.name] = val
