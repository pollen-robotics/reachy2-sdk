import asyncio
from grpc import Channel
from typing import Dict, Any, List

from google.protobuf.wrappers_pb2 import BoolValue

from .register import Register

from reachy_sdk_api_v2.component_pb2 import ComponentId
from reachy_sdk_api_v2.orbita2d_pb2 import (
    Axis,
    Float2D,
    Orbita2DCommand,
    Orbita2DState,
    Pose2D,
    Vector2D,
)

from reachy_sdk_api_v2.orbita2d_pb2_grpc import Orbita2DServiceStub

from .orbita_utils import OrbitaJoint, OrbitaMotor, OrbitaAxis


class Orbita2d:
    compliant = Register(readonly=False, type=BoolValue, label="compliant")

    def __init__(  # noqa: C901
        self,
        uid: int,
        name: str,
        axis1: Axis,
        axis2: Axis,
        initial_state: Orbita2DState,
        grpc_channel: Channel,
    ):
        self.name = name
        self.id = uid
        self._stub = Orbita2DServiceStub(grpc_channel)

        axis1_name = Axis.DESCRIPTOR.values_by_number[axis1].name.lower()
        axis2_name = Axis.DESCRIPTOR.values_by_number[axis2].name.lower()

        self._axis1 = axis1_name
        self._axis2 = axis2_name

        self._motor_1 = None
        self._motor_2 = None

        self._axis_to_name: Dict[str, str] = {
            "axis_1": self._axis1,
            "axis_2": self._axis2,
        }

        self._state: Dict[str, bool] = {}
        init_state: Dict[str, Dict[str, float]] = {}

        self._register_needing_sync: List[str] = []

        for field, value in initial_state.ListFields():
            if field.name == "compliant":
                self._state[field.name] = value
            else:
                if isinstance(value, Pose2D):
                    for axis, val in value.ListFields():
                        if axis.name not in init_state:
                            init_state[axis.name] = {}
                        init_state[axis.name][field.name] = val
                if isinstance(value, Float2D):
                    for motor, val in value.ListFields():
                        if motor.name not in init_state:
                            init_state[motor.name] = {}
                        init_state[motor.name][field.name] = val
                if isinstance(value, Vector2D):
                    for axis, val in value.ListFields():
                        if axis.name not in init_state:
                            init_state[axis.name] = {}
                        init_state[axis.name][field.name] = val

        setattr(
            self,
            axis1_name,
            OrbitaJoint(initial_state=init_state["axis_1"], axis_type=axis1, actuator=self),
        )
        setattr(
            self,
            axis2_name,
            OrbitaJoint(initial_state=init_state["axis_2"], axis_type=axis2, actuator=self),
        )

        setattr(self, "_motor_1", OrbitaMotor(initial_state=init_state["motor_1"], actuator=self))
        setattr(self, "_motor_2", OrbitaMotor(initial_state=init_state["motor_2"], actuator=self))

        setattr(self, "_x", OrbitaAxis(initial_state=init_state["x"]))
        setattr(self, "_y", OrbitaAxis(initial_state=init_state["y"]))

    def _build_grpc_cmd_msg(self, field: str) -> Pose2D | Float2D:
        if field == "goal_position":
            axis1_attr = getattr(self, self._axis1)
            axis2_attr = getattr(self, self._axis2)
            return Pose2D(
                axis_1=getattr(axis1_attr, field),
                axis_2=getattr(axis2_attr, field),
            )

        return Float2D(
            motor_1=getattr(self._motor_1, field),
            motor_2=getattr(self._motor_2, field),
        )

    def _setup_sync_loop(self) -> None:
        """Set up the async synchronisation loop.

        The setup is done separately, as the async Event should be created in the same EventLoop than it will be used.

        The _need_sync Event is used to inform the robot that some data need to be pushed to the real robot.
        The _register_needing_sync stores a list of the register that need to be synced.
        """
        self._need_sync = asyncio.Event()
        self._loop = asyncio.get_running_loop()

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == "compliant":
            self._state[__name] = __value
            # self._motor_1._state[__name] = __value
            # self._motor_2._state[__name] = __value

            async def set_in_loop() -> None:
                self._register_needing_sync.append(__name)
                self._need_sync.set()

            fut = asyncio.run_coroutine_threadsafe(set_in_loop(), self._loop)
            fut.result()
        super().__setattr__(__name, __value)

    def set_speed_limit(self, speed_limit: float) -> None:
        self._set_motors_fields("speed_limit", speed_limit)

    def _set_motors_fields(self, field: str, value: float) -> None:
        getattr(self, "_motor_1")._state[field] = value
        getattr(self, "_motor_2")._state[field] = value

        async def set_in_loop() -> None:
            self._register_needing_sync.append(field)
            self._need_sync.set()

        fut = asyncio.run_coroutine_threadsafe(set_in_loop(), self._loop)
        fut.result()

    def _pop_command(self) -> Orbita2DCommand:
        """Create a gRPC command from the registers that need to be synced."""
        values = {
            "id": ComponentId(id=self.id),
        }

        reg_to_update_1 = getattr(self, self._axis1)._register_needing_sync
        reg_to_update_2 = getattr(self, self._axis2)._register_needing_sync
        reg_to_update_3 = self._register_needing_sync

        for reg in set(reg_to_update_1).union(set(reg_to_update_2)).union(set(reg_to_update_3)):
            if reg == "compliant":
                values["compliant"] = BoolValue(value=self._state["compliant"])
            else:
                values[reg] = self._build_grpc_cmd_msg(reg)
        command = Orbita2DCommand(**values)

        reg_to_update_1.clear()
        reg_to_update_2.clear()
        reg_to_update_3.clear()
        self._need_sync.clear()

        return command

    def _update_with(self, new_state: Orbita2DState) -> None:
        """Update the orbita with a newly received (partial) state received from the gRPC server."""
        for field, value in new_state.ListFields():
            if field.name == "compliant":
                self._state[field.name] = value
            else:
                if isinstance(value, Pose2D):
                    for joint, val in value.ListFields():
                        j = getattr(self, self._axis_to_name[joint.name])
                        j._state[field.name] = val
                if isinstance(value, Float2D):
                    for motor, val in value.ListFields():
                        m = getattr(self, "_" + motor.name)
                        m._state[field.name] = val
                if isinstance(value, Vector2D):
                    for axis, val in value.ListFields():
                        a = getattr(self, "_" + axis.name)
                        a._state[field.name] = val
