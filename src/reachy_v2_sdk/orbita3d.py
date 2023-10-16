import asyncio
from grpc import Channel
from google.protobuf.wrappers_pb2 import BoolValue
from typing import Dict

from reachy_sdk_api_v2.orbita3d_pb2 import (
    Float3D,
    Orbita3DCommand,
    Orbita3DField,
    Orbita3DState,
    Orbita3DStateRequest,
    Vector3D,
)

from reachy_sdk_api_v2.component_pb2 import ComponentId
from reachy_sdk_api_v2.kinematics_pb2 import Quaternion, Rotation3D
from reachy_sdk_api_v2.orbita3d_pb2_grpc import Orbita3DServiceStub
from .orbita_utils import OrbitaJoint, OrbitaMotor, OrbitaAxis
from .register import Register


class Orbita3d:
    compliant = Register(readonly=False, type=BoolValue, label="compliant")

    def __init__(self, uid: int, name: str, initial_state: Orbita3DState, grpc_channel: Channel):  # noqa: C901
        self.name = name
        self.id = uid
        self._stub = Orbita3DServiceStub(grpc_channel)

        self._state: Dict[str, bool] = {}
        init_state: Dict[str, Dict[str, float]] = {}

        for field, value in initial_state.ListFields():
            if field.name == "compliant":
                self._state[field.name] = value.value
            else:
                if isinstance(value, Rotation3D):
                    for _, rpy in value.ListFields():
                        for axis, val in rpy.ListFields():
                            if axis.name not in init_state:
                                init_state[axis.name] = {}
                            init_state[axis.name][field.name] = val
                if isinstance(value, Float3D):
                    for motor, val in value.ListFields():
                        if motor.name not in init_state:
                            init_state[motor.name] = {}
                        init_state[motor.name][field.name] = val
                if isinstance(value, Vector3D):
                    for axis, val in value.ListFields():
                        if axis.name not in init_state:
                            init_state[axis.name] = {}
                        init_state[axis.name][field.name] = val

        self.roll = OrbitaJoint(initial_state=init_state["roll"], axis_type="roll", actuator=self)
        self.pitch = OrbitaJoint(initial_state=init_state["pitch"], axis_type="pitch", actuator=self)
        self.yaw = OrbitaJoint(initial_state=init_state["yaw"], axis_type="yaw", actuator=self)

        self._motor_1 = OrbitaMotor(initial_state=init_state["motor_1"])
        self._motor_2 = OrbitaMotor(initial_state=init_state["motor_2"])
        self._motor_3 = OrbitaMotor(initial_state=init_state["motor_3"])

        self._x = OrbitaAxis(initial_state=init_state["x"])
        self._y = OrbitaAxis(initial_state=init_state["y"])
        self._z = OrbitaAxis(initial_state=init_state["z"])

    @property
    def temperatures(self) -> Dict[str, Register]:
        return {
            "motor_1": self._motor_1.temperature,
            "motor_2": self._motor_2.temperature,
            "motor_3": self._motor_3.temperature,
        }

    def _update_with(self, new_state: Orbita3DState) -> None:
        """Update the orbita with a newly received (partial) state received from the gRPC server."""
        for field, value in new_state.ListFields():
            if field.name == "compliant":
                self._state[field.name] = value
            else:
                if isinstance(value, Rotation3D):
                    for _, rpy in value.ListFields():
                        for joint, val in rpy.ListFields():
                            j = getattr(self, joint.name)
                            j._state[field.name] = val
                if isinstance(value, Float3D):
                    for motor, val in value.ListFields():
                        m = getattr(self, "_" + motor.name)
                        m._state[field.name] = val
                if isinstance(value, Vector3D):
                    for axis, val in value.ListFields():
                        a = getattr(self, "_" + axis.name)
                        a._state[field.name] = val

    def _build_3d_msg(self, field: str) -> Float3D:
        if field == "goal_position":
            return Rotation3D(
                q=Quaternion(
                    x=1.0,
                    y=1.0,
                    z=1.0,
                    w=1.0,
                )
            )
        return Float3D(
            motor_1=getattr(self.roll, field),
            motor_2=getattr(self.pitch, field),
            motor_3=getattr(self.yaw, field),
        )

    def _setup_sync_loop(self) -> None:
        """Set up the async synchronisation loop.

        The setup is done separately, as the async Event should be created in the same EventLoop than it will be used.

        The _need_sync Event is used to inform the robot that some data need to be pushed to the real robot.
        The _register_needing_sync stores a list of the register that need to be synced.
        """
        self._need_sync = asyncio.Event()
        self._loop = asyncio.get_running_loop()

    def _pop_command(self) -> Orbita3DCommand:
        """Create a gRPC command from the registers that need to be synced."""
        values = {
            "id": ComponentId(id=self.id),
        }

        set_reg_roll = set(self.roll._register_needing_sync)
        set_reg_pitch = set(self.pitch._register_needing_sync)
        set_reg_yaw = set(self.yaw._register_needing_sync)

        for reg in set_reg_roll.union(set_reg_pitch).union(set_reg_yaw):
            if reg == "compliant":
                values["compliant"] = BoolValue(value=self.compliant)
            else:
                values[reg] = self._build_3d_msg(reg)
        command = Orbita3DCommand(**values)

        self.roll._register_needing_sync.clear()
        self.pitch._register_needing_sync.clear()
        self.yaw._register_needing_sync.clear()
        self._need_sync.clear()

        return command

    # TODO: perform the update in a thread
    # TODO: find a smarter way to do this
    def update_3dstate(self) -> None:
        resp = self._stub.GetState(
            Orbita3DStateRequest(
                id=ComponentId(id=self.name),
                fields=[
                    Orbita3DField.PRESENT_POSITION,
                    Orbita3DField.PRESENT_SPEED,
                    Orbita3DField.PRESENT_LOAD,
                    Orbita3DField.TEMPERATURE,
                    Orbita3DField.GOAL_POSITION,
                    Orbita3DField.SPEED_LIMIT,
                    Orbita3DField.TORQUE_LIMIT,
                ],
            )
        )

        self.roll.present_position = resp.present_position.roll
        self.pitch.present_position = resp.present_position.pitch
        self.yaw.present_position = resp.present_position.yaw

        self.roll.present_speed = resp.present_speed.roll
        self.pitch.present_speed = resp.present_speed.pitch
        self.yaw.present_speed = resp.present_speed.yaw

        self.roll.present_load = resp.present_load.roll
        self.pitch.present_load = resp.present_load.pitch
        self.yaw.present_load = resp.present_load.yaw

        self.roll.goal_position = resp.goal_position.roll
        self.pitch.goal_position = resp.goal_position.pitch
        self.yaw.goal_position = resp.goal_position.yaw

        self.roll.speed_limit = resp.speed_limit.roll
        self.pitch.speed_limit = resp.speed_limit.pitch
        self.yaw.speed_limit = resp.speed_limit.yaw

        self.roll.torque_limit = resp.torque_limit.roll
        self.pitch.torque_limit = resp.torque_limit.pitch
        self.yaw.torque_limit = resp.present_position.yaw
