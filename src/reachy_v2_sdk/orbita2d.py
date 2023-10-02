from grpc import Channel

from reachy_sdk_api_v2.orbita2d_pb2 import (
    Axis,
    Orbita2DField,
    Orbita2DStateRequest,
    Orbita2DState,
)

from reachy_sdk_api_v2.component_pb2 import ComponentId
from reachy_sdk_api_v2.orbita2d_pb2_grpc import Orbita2DServiceStub

from .orbita_utils import OrbitaAxis


class Orbita2d:
    def __init__(self, name: str, axis1: Axis, axis2: Axis, grpc_channel: Channel):
        self.name = name
        self._stub = Orbita2DServiceStub(grpc_channel)

        axis1_name = Axis.DESCRIPTOR.values_by_number[axis1].name.lower()
        axis2_name = Axis.DESCRIPTOR.values_by_number[axis2].name.lower()

        self._axis1 = axis1_name
        self._axis2 = axis2_name
        setattr(self, axis1_name, OrbitaAxis(axis1))
        setattr(self, axis2_name, OrbitaAxis(axis2))

        self.compliant = False

    # TODO: perform the update in a thread
    # TODO: find a smarter way to do this
    def update_2dstate(self) -> None:
        resp = self._stub.GetState(
            Orbita2DStateRequest(
                id=ComponentId(id=self.name),
                fields=[
                    Orbita2DField.PRESENT_POSITION,
                    Orbita2DField.PRESENT_SPEED,
                    Orbita2DField.PRESENT_LOAD,
                    Orbita2DField.TEMPERATURE,
                    Orbita2DField.GOAL_POSITION,
                    Orbita2DField.SPEED_LIMIT,
                    Orbita2DField.TORQUE_LIMIT,
                ],
            )
        )
        axis1_attr = getattr(self, self._axis1)
        axis2_attr = getattr(self, self._axis2)

        axis1_attr._present_position = resp.present_position.axis_1
        axis2_attr._present_position = resp.present_position.axis_2

        axis1_attr._present_speed = resp.present_speed.axis_1
        axis2_attr._present_speed = resp.present_speed.axis_2

        axis1_attr._present_load = resp.present_load.axis_1
        axis2_attr._present_load = resp.present_load.axis_2

        axis1_attr._goal_position = resp.goal_position.axis_1
        axis2_attr._goal_position = resp.goal_position.axis_2

        axis1_attr._speed_limit = resp.speed_limit.axis_1
        axis2_attr._speed_limit = resp.speed_limit.axis_2

        axis1_attr._torque_limit = resp.torque_limit.axis_1
        axis2_attr._torque_limit = resp.torque_limit.axis_2

    def _update_with(self, new_state: Orbita2DState) -> None:
        """Update the orbita with a newly received (partial) state received from the gRPC server."""
        axis1_attr = getattr(self, self._axis1)
        axis2_attr = getattr(self, self._axis2)
        axis1_attr._temperature = new_state.temperature.axis_1
        axis2_attr._temperature = new_state.temperature.axis_2
