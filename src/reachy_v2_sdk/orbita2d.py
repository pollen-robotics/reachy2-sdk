import asyncio
from grpc import Channel

from google.protobuf.wrappers_pb2 import BoolValue

from reachy_sdk_api_v2.orbita2d_pb2 import (
    Axis,
    Float2D,
    Orbita2DCommand,
    Orbita2DsCommand,
    Orbita2DField,
    Orbita2DStateRequest,
)

from reachy_sdk_api_v2.component_pb2 import ComponentId
from reachy_sdk_api_v2.orbita2d_pb2_grpc import Orbita2DServiceStub

from .orbita_utils import OrbitaJoint


class Orbita2d:
    def __init__(self, name: str, axis1: Axis, axis2: Axis, grpc_channel: Channel):
        self.name = name
        self._stub = Orbita2DServiceStub(grpc_channel)

        axis1_name = Axis.DESCRIPTOR.values_by_number[axis1].name.lower()
        axis2_name = Axis.DESCRIPTOR.values_by_number[axis2].name.lower()

        self._axis1 = axis1_name
        self._axis2 = axis2_name

        init_state = {
            "present_position": 20.0,
            "present_speed": 0.0,
            "present_load": 0.0,
            "temperature": 0.0,
            "goal_position": 100.0,
            "speed_limit": 0.0,
            "torque_limit": 0.0,
        }

        # TODO get initial state from grpc server
        # Should set this as @property?
        setattr(self, axis1_name, OrbitaJoint(initial_state=init_state.copy(), axis_type=axis1, actuator=self))
        setattr(self, axis2_name, OrbitaJoint(initial_state=init_state.copy(), axis_type=axis2, actuator=self))

        self.compliant = False

    def _build_2d_float_msg(self, field: str) -> Float2D:
        axis1_attr = getattr(self, self._axis1)
        axis2_attr = getattr(self, self._axis2)

        return Float2D(
            motor_1=getattr(axis1_attr, field),
            motor_2=getattr(axis2_attr, field),
        )

    def _setup_sync_loop(self) -> None:
        """Set up the async synchronisation loop.

        The setup is done separately, as the async Event should be created in the same EventLoop than it will be used.

        The _need_sync Event is used to inform the robot that some data need to be pushed to the real robot.
        The _register_needing_sync stores a list of the register that need to be synced.
        """
        self._need_sync = asyncio.Event()
        self._loop = asyncio.get_running_loop()

    def _pop_command(self) -> Orbita2DCommand:
        """Create a gRPC command from the registers that need to be synced."""
        values = {
            "id": ComponentId(id=self.name),
        }

        reg_to_update_1 = getattr(self, self._axis1)._register_needing_sync
        reg_to_update_2 = getattr(self, self._axis2)._register_needing_sync

        for reg in set(reg_to_update_1).union(set(reg_to_update_2)):
            if reg == "compliant":
                values["compliant"] = BoolValue(value=self.compliant)
            else:
                values[reg] = self._build_2d_float_msg(reg)
        command = Orbita2DCommand(**values)

        reg_to_update_1.clear()
        reg_to_update_2.clear()
        self._need_sync.clear()

        return command

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
