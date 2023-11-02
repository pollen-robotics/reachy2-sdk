import asyncio
from grpc import Channel
from typing import Dict, Any, List, Tuple

from google.protobuf.wrappers_pb2 import BoolValue, FloatValue

from .register import Register

from reachy_sdk_api_v2.component_pb2 import ComponentId, PIDGains
from reachy_sdk_api_v2.orbita2d_pb2 import (
    Axis,
    Float2D,
    Orbita2DCommand,
    Orbita2DState,
    Pose2D,
    Vector2D,
    PID2D,
)

from reachy_sdk_api_v2.orbita2d_pb2_grpc import Orbita2DServiceStub

from .orbita_utils import OrbitaJoint2D, OrbitaMotor, OrbitaAxis, _to_internal_position


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

        self._state: Dict[str, bool] = {}
        init_state: Dict[str, Dict[str, float]] = {}

        self._register_needing_sync: List[str] = []

        for field, value in initial_state.ListFields():
            if field.name == "compliant":
                self._state[field.name] = value
                init_state["motor_1"][field.name] = value
                init_state["motor_2"][field.name] = value
            else:
                if isinstance(value, Pose2D):
                    for axis, val in value.ListFields():
                        if axis.name not in init_state:
                            init_state[axis.name] = {}
                        init_state[axis.name][field.name] = val
                if isinstance(value, Float2D | PID2D):
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
            OrbitaJoint2D(initial_state=init_state["axis_1"], axis_type=axis1_name, actuator=self),
        )
        setattr(
            self,
            axis2_name,
            OrbitaJoint2D(initial_state=init_state["axis_2"], axis_type=axis2_name, actuator=self),
        )
        self._joints = {"axis_1": getattr(self, axis1_name), "axis_2": getattr(self, axis2_name)}

        self.__motor_1 = OrbitaMotor(initial_state=init_state["motor_1"], actuator=self)
        self.__motor_2 = OrbitaMotor(initial_state=init_state["motor_2"], actuator=self)
        self._motors = {"motor_1": self.__motor_1, "motor_2": self.__motor_2}

        self.__x = OrbitaAxis(initial_state=init_state["x"])
        self.__y = OrbitaAxis(initial_state=init_state["y"])
        self._axis = {"x": self.__x, "y": self.__y}

    def __repr__(self) -> str:
        """Clean representation of an Orbita2D."""
        s = "\n\t".join([str(joint) for joint in self._joints.values()])
        return f"""<Orbita2D compliant={self.compliant} joints=\n\t{
            s
        }\n>"""

    def set_speed_limit(self, speed_limit: float) -> None:
        if not isinstance(speed_limit, float | int):
            raise ValueError(f"Expected one of: float, int for speed_limit, got {type(speed_limit).__name__}")
        speed_limit = _to_internal_position(speed_limit)
        self._set_motors_fields("speed_limit", speed_limit)

    def set_torque_limit(self, torque_limit: float) -> None:
        if not isinstance(torque_limit, float | int):
            raise ValueError(f"Expected one of: float, int for torque_limit, got {type(torque_limit).__name__}")
        self._set_motors_fields("torque_limit", torque_limit)

    def set_pid(self, pid: Tuple[float, float, float]) -> None:
        if isinstance(pid, tuple) and len(pid) == 3 and all(isinstance(n, float | int) for n in pid):
            for m in self._motors.values():
                m._tmp_pid = pid
            self._update_loop("pid")
        else:
            raise ValueError("pid should be of type Tuple[float, float, float]")

    def get_speed_limit(self) -> Dict[str, float]:
        return {motor_name: m.speed_limit for motor_name, m in self._motors.items()}

    def get_torque_limit(self) -> Dict[str, float]:
        return {motor_name: m.torque_limit for motor_name, m in self._motors.items()}

    def get_pid(self) -> Dict[str, Tuple[float, float, float]]:
        return {motor_name: m.pid for motor_name, m in self._motors.items()}

    def _build_grpc_cmd_msg(self, field: str) -> Pose2D | PID2D | Float2D:
        if field == "goal_position":
            return Pose2D(
                axis_1=self._joints["axis_1"]._state["goal_position"],
                axis_2=self._joints["axis_2"]._state["goal_position"],
            )

        elif field == "pid":
            return PID2D(
                motor_1=PIDGains(
                    p=self.__motor_1._state[field].p,
                    i=self.__motor_1._state[field].i,
                    d=self.__motor_1._state[field].d,
                ),
                motor_2=PIDGains(
                    p=self.__motor_2._state[field].p,
                    i=self.__motor_2._state[field].i,
                    d=self.__motor_2._state[field].d,
                ),
            )

        return Float2D(
            motor_1=self.__motor_1._state[field],
            motor_2=self.__motor_2._state[field],
        )

    def _build_grpc_cmd_msg_actuator(self, field: str) -> Float2D:
        if field == "pid":
            motor_1_gains = self.__motor_1._tmp_pid
            motor_2_gains = self.__motor_2._tmp_pid
            if type(motor_1_gains) is tuple and type(motor_2_gains) is tuple:
                return PID2D(
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
                )

        motor_1_value = self.__motor_1._tmp_fields[field]
        motor_2_value = self.__motor_2._tmp_fields[field]
        return Float2D(
            motor_1=FloatValue(value=motor_1_value),
            motor_2=FloatValue(value=motor_2_value),
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
            if not isinstance(__value, bool):
                raise ValueError(f"Expected bool for compliant value, got {type(__value).__name__}")
            self._state[__name] = __value

            async def set_in_loop() -> None:
                self._register_needing_sync.append(__name)
                self._need_sync.set()

            fut = asyncio.run_coroutine_threadsafe(set_in_loop(), self._loop)
            fut.result()

        else:
            super().__setattr__(__name, __value)

    def _set_motors_fields(self, field: str, value: float) -> None:
        for m in self._motors.values():
            m._tmp_fields[field] = value

        self._update_loop(field)

    def _update_loop(self, field: str) -> None:
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

        set_reg_to_update = set(self._register_needing_sync)
        for reg in set_reg_to_update:
            if reg == "compliant":
                values["compliant"] = BoolValue(value=self._state["compliant"])
            else:
                values[reg] = self._build_grpc_cmd_msg_actuator(reg)

        set_reg_to_update = set()
        for obj in list(self._joints.values()) + list(self._motors.values()):
            set_reg_to_update = set_reg_to_update.union(set(obj._register_needing_sync))
        for reg in set_reg_to_update:
            values[reg] = self._build_grpc_cmd_msg(reg)

        command = Orbita2DCommand(**values)

        self._register_needing_sync.clear()
        for obj in list(self._joints.values()) + list(self._motors.values()):
            obj._register_needing_sync.clear()
        self._need_sync.clear()

        return command

    def _update_with(self, new_state: Orbita2DState) -> None:
        """Update the orbita with a newly received (partial) state received from the gRPC server."""
        for field, value in new_state.ListFields():
            if field.name == "compliant":
                self._state[field.name] = value
                for m in self._motors.values():
                    m._state[field.name] = value
            else:
                if isinstance(value, Pose2D):
                    for joint, val in value.ListFields():
                        self._joints[joint.name]._state[field.name] = val

                if isinstance(value, Float2D):
                    for motor, val in value.ListFields():
                        self._motors[motor.name]._state[field.name] = val

                if isinstance(value, Vector2D):
                    for axis, val in value.ListFields():
                        self._axis[axis.name]._state[field.name] = val
