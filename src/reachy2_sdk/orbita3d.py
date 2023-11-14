"""This module defines the Orbita3d class and its registers, joints, motors and axis."""
import asyncio
from typing import Any, Dict, List, Tuple

from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from grpc import Channel
from pyquaternion import Quaternion as pyQuat
from reachy2_sdk_api.component_pb2 import ComponentId, PIDGains
from reachy2_sdk_api.kinematics_pb2 import ExtEulerAngles, Quaternion, Rotation3D
from reachy2_sdk_api.orbita3d_pb2 import (
    PID3D,
    Float3D,
    Orbita3DCommand,
    Orbita3DGoal,
    Orbita3DState,
    Vector3D,
)
from reachy2_sdk_api.orbita3d_pb2_grpc import Orbita3DServiceStub

from .orbita_utils import OrbitaAxis, OrbitaJoint3D, OrbitaMotor, _to_internal_position
from .register import Register


class Orbita3d:
    """The Orbita3d class represents any Orbita2d actuator and its registers, joints, motors and axis.

    The Orbita3d class is used to store the up-to-date state of the actuator, especially:
        - its compliancy
        - its joints state
        - its motors state
        - its axis state

    The only register available at the actuator is the compliancy RW register.
    You can set the compliance on/off (boolean).

    You can access registers of the motors from the actuators with function that act on all the actuator's motors.
    Lower registers which can be read/write at actuator level:
    - speed limit (in degree per second, for all motors of the actuator)
    - torque limit (in %, for all motors of the actuator)
    - pid (for all motors of the actuator)
    Lower registers that are read-only but acessible at actuator level:
    - temperatures (temperatures of all motors of the actuator)
    """

    compliant = Register(readonly=False, type=BoolValue, label="compliant")

    def __init__(self, uid: int, name: str, initial_state: Orbita3DState, grpc_channel: Channel):
        """Initialize the Orbita2d with its joints, motors and axis."""
        self.name = name
        self.id = uid
        self._stub = Orbita3DServiceStub(grpc_channel)

        self._state: Dict[str, bool] = {}
        init_state: Dict[str, Dict[str, float]] = self._create_init_state(initial_state)

        self._register_needing_sync: List[str] = []

        self.roll = OrbitaJoint3D(initial_state=init_state["roll"], axis_type="roll", actuator=self)
        self.pitch = OrbitaJoint3D(initial_state=init_state["pitch"], axis_type="pitch", actuator=self)
        self.yaw = OrbitaJoint3D(initial_state=init_state["yaw"], axis_type="yaw", actuator=self)
        self._joints = {"roll": self.roll, "pitch": self.pitch, "yaw": self.yaw}

        self.__motor_1 = OrbitaMotor(initial_state=init_state["motor_1"], actuator=self)
        self.__motor_2 = OrbitaMotor(initial_state=init_state["motor_2"], actuator=self)
        self.__motor_3 = OrbitaMotor(initial_state=init_state["motor_3"], actuator=self)
        self._motors = {
            "motor_1": self.__motor_1,
            "motor_2": self.__motor_2,
            "motor_3": self.__motor_3,
        }

        self.__x = OrbitaAxis(initial_state=init_state["x"])
        self.__y = OrbitaAxis(initial_state=init_state["y"])
        self.__z = OrbitaAxis(initial_state=init_state["z"])
        self._axis = {"x": self.__x, "y": self.__y, "z": self.__z}

    def _create_init_state(self, initial_state: Orbita3DState) -> Dict[str, Dict[str, float]]:  # noqa: C901
        init_state: Dict[str, Dict[str, float]] = {}

        for field, value in initial_state.ListFields():
            if field.name == "compliant":
                self._state[field.name] = value
                init_state["motor_1"][field.name] = value
                init_state["motor_2"][field.name] = value
                init_state["motor_3"][field.name] = value
            else:
                if isinstance(value, Rotation3D):
                    for joint in ["roll", "pitch", "yaw"]:
                        if joint not in init_state:
                            init_state[joint] = {}
                        init_state[joint][field.name] = getattr(value.rpy, joint)
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
        return init_state

    def __repr__(self) -> str:
        """Clean representation of an Orbita3D."""
        s = "\n\t".join([str(joint) for _, joint in self._joints.items()])
        return f"""<Orbita3D compliant={self.compliant} joints=\n\t{
            s
        }\n>"""

    def set_speed_limit(self, speed_limit: float) -> None:
        """Set a speed_limit on all motors of the actuator"""
        if not isinstance(speed_limit, float | int):
            raise ValueError(f"Expected one of: float, int for speed_limit, got {type(speed_limit).__name__}")
        speed_limit = _to_internal_position(speed_limit)
        self._set_motors_fields("speed_limit", speed_limit)

    def set_torque_limit(self, torque_limit: float) -> None:
        """Set a torque_limit on all motors of the actuator"""
        if not isinstance(torque_limit, float | int):
            raise ValueError(f"Expected one of: float, int for torque_limit, got {type(torque_limit).__name__}")
        self._set_motors_fields("torque_limit", torque_limit)

    def set_pid(self, pid: Tuple[float, float, float]) -> None:
        """Set a pid value on all motors of the actuator"""
        if isinstance(pid, tuple) and len(pid) == 3 and all(isinstance(n, float | int) for n in pid):
            for m in self._motors.values():
                m._tmp_pid = pid
            self._update_loop("pid")
        else:
            raise ValueError("pid should be of type Tuple[float, float, float]")

    def get_speed_limit(self) -> Dict[str, float]:
        """Get speed_limit of all motors of the actuator"""
        return {motor_name: m.speed_limit for motor_name, m in self._motors.items()}

    def get_torque_limit(self) -> Dict[str, float]:
        """Get torque_limit of all motors of the actuator"""
        return {motor_name: m.torque_limit for motor_name, m in self._motors.items()}

    def get_pid(self) -> Dict[str, Tuple[float, float, float]]:
        """Get pid of all motors of the actuator"""
        return {motor_name: m.pid for motor_name, m in self._motors.items()}

    def orient(self, q: pyQuat, duration: float) -> None:
        """Orient the head to a given quaternion.

        Goal orientation is reached in a defined duration"""
        req = Orbita3DGoal(
            id=ComponentId(id=self.id, name=self.name),
            rotation=Rotation3D(q=Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)),
            duration=FloatValue(value=duration),
        )
        self._stub.GoToOrientation(req)

    def rotate_to(self, roll: float, pitch: float, yaw: float, duration: float) -> None:
        """Rotate the head to a given roll, pitch, yaw orientation.

        Goal orientation is reached in a defined duration"""
        req = Orbita3DGoal(
            id=ComponentId(id=self.id, name=self.name),
            rotation=Rotation3D(rpy=ExtEulerAngles(roll=roll, pitch=pitch, yaw=yaw)),
            duration=FloatValue(value=duration),
        )
        self._stub.GoToOrientation(req)

    @property
    def temperatures(self) -> Dict[str, Register]:
        """Get temperatures of all the motors of the actuator"""
        return {motor_name: m.temperature for motor_name, m in self._motors.items()}

    def _update_with(self, new_state: Orbita3DState) -> None:  # noqa: C901
        """Update the orbita with a newly received (partial) state received from the gRPC server."""
        for field, value in new_state.ListFields():
            if field.name == "compliant":
                self._state[field.name] = value
                for m in self._motors.values():
                    m._state[field.name] = value
            else:
                if isinstance(value, Rotation3D):
                    self.roll._state[field.name] = value.rpy.roll
                    self.pitch._state[field.name] = value.rpy.pitch
                    self.yaw._state[field.name] = value.rpy.yaw
                if isinstance(value, Float3D):
                    for motor, val in value.ListFields():
                        m = self._motors[motor.name]
                        m._state[field.name] = val
                if isinstance(value, Vector3D):
                    for axis, val in value.ListFields():
                        a = self._axis[axis.name]
                        a._state[field.name] = val

    def _build_grpc_cmd_msg(self, field: str) -> Float3D:
        """Build a gRPC message from the registers that need to be synced at the joints and
        motors level. Registers can either be goal_position, pid or speed_limit/torque_limit.
        """
        if field == "goal_position":
            return Rotation3D(
                rpy=ExtEulerAngles(
                    roll=self.roll._state["goal_position"],
                    pitch=self.pitch._state["goal_position"],
                    yaw=self.yaw._state["goal_position"],
                )
            )

        elif field == "pid":
            return PID3D(
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
                motor_3=PIDGains(
                    p=self.__motor_3._state[field].p,
                    i=self.__motor_3._state[field].i,
                    d=self.__motor_3._state[field].d,
                ),
            )

        return Float3D(
            motor_1=self.__motor_1._state[field],
            motor_2=self.__motor_2._state[field],
            motor_3=self.__motor_3._state[field],
        )

    def _build_grpc_cmd_msg_actuator(self, field: str) -> Float3D:
        """Build a gRPC message from the registers that need to be synced at the actuator level.
        Registers can either be compliant, pid, speed_limit or torque_limit."""
        if field == "pid":
            motor_1_gains = self.__motor_1._tmp_pid
            motor_2_gains = self.__motor_2._tmp_pid
            motor_3_gains = self.__motor_3._tmp_pid
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

        motor_1_value = self.__motor_1._tmp_fields[field]
        motor_2_value = self.__motor_2._tmp_fields[field]
        motor_3_value = self.__motor_3._tmp_fields[field]
        return Float3D(
            motor_1=FloatValue(value=motor_1_value),
            motor_2=FloatValue(value=motor_2_value),
            motor_3=FloatValue(value=motor_3_value),
        )

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

    def _setup_sync_loop(self) -> None:
        """Set up the async synchronisation loop.

        The setup is done separately, as the async Event should be created in the same EventLoop than it will be used.

        The _need_sync Event is used to inform the robot that some data need to be pushed to the real robot.
        The _register_needing_sync stores a list of the register that need to be synced.
        """
        self._need_sync = asyncio.Event()
        self._loop = asyncio.get_running_loop()

    def _set_motors_fields(self, field: str, value: float) -> None:
        """Set the value of the register for all motors of the actuator.

        It is used to set pid, speed_limit and torque_limit.
        """
        for m in self._motors.values():
            m._tmp_fields[field] = value

        self._update_loop(field)

    def _update_loop(self, field: str) -> None:
        """Update the registers that need to be synced.

        Set a threading event to inform the stream command thread that some data need to be pushed
        to the robot.
        """

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
        for obj in list(self._joints.values()) + list(self._motors.values()):
            set_reg_to_update = set_reg_to_update.union(set(obj._register_needing_sync))
        for reg in set_reg_to_update:
            values[reg] = self._build_grpc_cmd_msg(reg)

        command = Orbita3DCommand(**values)

        self._register_needing_sync.clear()
        for obj in list(self._joints.values()) + list(self._motors.values()):
            obj._register_needing_sync.clear()
        self._need_sync.clear()

        return command
