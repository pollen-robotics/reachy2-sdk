import asyncio
from grpc import Channel
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from typing import Dict, List, Any, Tuple

from reachy_sdk_api_v2.orbita3d_pb2 import (
    Float3D,
    Orbita3DCommand,
    Orbita3DState,
    Vector3D,
    PID3D,
)

from reachy_sdk_api_v2.component_pb2 import ComponentId, PIDGains
from reachy_sdk_api_v2.kinematics_pb2 import ExtEulerAngles, Rotation3D
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

        self._register_needing_sync: List[str] = []

        for field, value in initial_state.ListFields():
            if field.name == "compliant":
                self._state[field.name] = value
                init_state["motor_1"][field.name] = value
                init_state["motor_2"][field.name] = value
                init_state["motor_3"][field.name] = value
            else:
                if isinstance(value, Rotation3D):
                    for _, rpy in value.ListFields():
                        for axis, val in rpy.ListFields():
                            if axis.name not in init_state:
                                init_state[axis.name] = {}
                            init_state[axis.name][field.name] = val
                if isinstance(value, Float3D | PID3D):
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
        self.__joints = [self.roll, self.pitch, self.yaw]

        self._motor_1 = OrbitaMotor(initial_state=init_state["motor_1"], actuator=self)
        self._motor_2 = OrbitaMotor(initial_state=init_state["motor_2"], actuator=self)
        self._motor_3 = OrbitaMotor(initial_state=init_state["motor_3"], actuator=self)
        self.__motors = [self._motor_1, self._motor_2, self._motor_3]

        self._x = OrbitaAxis(initial_state=init_state["x"])
        self._y = OrbitaAxis(initial_state=init_state["y"])
        self._z = OrbitaAxis(initial_state=init_state["z"])
        self.__axis = [self._x, self._y, self._z]

    def set_speed_limit(self, speed_limit: float) -> None:
        self._set_motors_fields("speed_limit", speed_limit)

    def set_torque_limit(self, torque_limit: float) -> None:
        self._set_motors_fields("torque_limit", torque_limit)

    def set_pid(self, pid: Tuple[float, float, float]) -> None:
        if isinstance(pid, tuple) and len(pid) == 3:
            for m in self.__motors:
                m._tmp_pid = pid
            self._update_loop("pid")
        else:
            raise ValueError("pid should be of type Tuple[float, float, float]")

    def get_speed_limit(self) -> Dict[str, float]:
        return {
            "motor_1": getattr(self, "_motor_1").speed_limit,
            "motor_2": getattr(self, "_motor_2").speed_limit,
            "motor_3": getattr(self, "_motor_3").speed_limit,
        }

    def get_torque_limit(self) -> Dict[str, float]:
        return {
            "motor_1": getattr(self, "_motor_1").torque_limit,
            "motor_2": getattr(self, "_motor_2").torque_limit,
            "motor_3": getattr(self, "_motor_3").torque_limit,
        }

    @property
    def temperatures(self) -> Dict[str, Register]:
        return {
            "motor_1": self._motor_1.temperature,
            "motor_2": self._motor_2.temperature,
            "motor_3": self._motor_3.temperature,
        }

    def _update_with(self, new_state: Orbita3DState) -> None:  # noqa: C901
        """Update the orbita with a newly received (partial) state received from the gRPC server."""
        for field, value in new_state.ListFields():
            if field.name == "compliant":
                self._state[field.name] = value
                for m in self.__motors:
                    m._state[field.name] = value
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

    def _build_grpc_cmd_msg(self, field: str) -> Float3D:
        if field == "goal_position":
            return Rotation3D(
                rpy=ExtEulerAngles(
                    roll=FloatValue(value=getattr(self.roll, field)),
                    pitch=FloatValue(value=getattr(self.pitch, field)),
                    yaw=FloatValue(value=getattr(self.yaw, field)),
                )
            )

        elif field == "pid":
            return PID3D(
                motor_1=PIDGains(
                    p=self._motor_1._state[field].p,
                    i=self._motor_1._state[field].i,
                    d=self._motor_1._state[field].d,
                ),
                motor_2=PIDGains(
                    p=self._motor_2._state[field].p,
                    i=self._motor_2._state[field].i,
                    d=self._motor_2._state[field].d,
                ),
                motor_3=PIDGains(
                    p=self._motor_3._state[field].p,
                    i=self._motor_3._state[field].i,
                    d=self._motor_3._state[field].d,
                ),
            )

        return Float3D(
            motor_1=FloatValue(value=getattr(self._motor_1, field)),
            motor_2=FloatValue(value=getattr(self._motor_2, field)),
            motor_3=FloatValue(value=getattr(self._motor_3, field)),
        )

    def _build_grpc_cmd_msg_actuator(self, field: str) -> Float3D:
        if field == "pid":
            motor_1_gains = self.__motors[0]._tmp_pid
            motor_2_gains = self.__motors[1]._tmp_pid
            motor_3_gains = self.__motors[1]._tmp_pid
            if type(motor_1_gains) is tuple and type(motor_2_gains) is tuple and type(motor_3_gains) is tuple:
                return PID3D(
                    motor_1=PIDGains(
                        p=FloatValue(value=motor_1_gains[0]),
                        i=FloatValue(value=motor_1_gains[1]),
                        d=FloatValue(value=motor_1_gains[2]),
                    ),
                    motor_2=PIDGains(
                        p=FloatValue(value=motor_2_gains[0]),
                        i=FloatValue(value=motor_2_gains[1]),
                        d=FloatValue(value=motor_2_gains[2]),
                    ),
                    motor_3=PIDGains(
                        p=FloatValue(value=motor_3_gains[0]),
                        i=FloatValue(value=motor_3_gains[1]),
                        d=FloatValue(value=motor_3_gains[2]),
                    ),
                )

        motor_1_value = self.__motors[0]._tmp_fields[field]
        motor_2_value = self.__motors[1]._tmp_fields[field]
        motor_3_value = self.__motors[1]._tmp_fields[field]
        return Float3D(
            motor_1=FloatValue(value=motor_1_value),
            motor_2=FloatValue(value=motor_2_value),
            motor_3=FloatValue(value=motor_3_value),
        )

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == "compliant":
            self._state[__name] = __value

            async def set_in_loop() -> None:
                self._register_needing_sync.append(__name)
                self._need_sync.set()

            fut = asyncio.run_coroutine_threadsafe(set_in_loop(), self._loop)
            fut.result()
        super().__setattr__(__name, __value)

    def _setup_sync_loop(self) -> None:
        """Set up the async synchronisation loop.

        The setup is done separately, as the async Event should be created in the same EventLoop than it will be used.

        The _need_sync Event is used to inform the robot that some data need to be pushed to the real robot.
        The _register_needing_sync stores a list of the register that need to be synced.
        """
        self._need_sync = asyncio.Event()
        self._loop = asyncio.get_running_loop()

    def _set_motors_fields(self, field: str, value: float) -> None:
        for m in self.__motors:
            m._tmp_fields[field] = value

        self._update_loop(field)

    def _update_loop(self, field: str) -> None:
        async def set_in_loop() -> None:
            self._register_needing_sync.append(field)
            self._need_sync.set()

        fut = asyncio.run_coroutine_threadsafe(set_in_loop(), self._loop)
        fut.result()

    def _pop_command(self) -> Orbita3DCommand:
        """Create a gRPC command from the registers that need to be synced."""
        values = {
            "id": ComponentId(id=self.id),
        }

        set_reg_to_update = set(self._register_needing_sync)
        for reg in set_reg_to_update:
            if reg == "compliant":
                values["compliant"] = BoolValue(value=self._state["compliant"])
            else:
                values[reg] = self._build_grpc_cmd_msg_actuator(reg)

        set_reg_to_update = set()
        for obj in self.__joints + self.__motors:
            set_reg_to_update = set_reg_to_update.union(set(obj._register_needing_sync))
        for reg in set_reg_to_update:
            values[reg] = self._build_grpc_cmd_msg(reg)

        command = Orbita3DCommand(**values)

        self._register_needing_sync.clear()
        for obj in self.__joints + self.__motors:
            obj._register_needing_sync.clear()
        self._need_sync.clear()

        return command
