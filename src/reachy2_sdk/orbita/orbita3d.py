"""This module defines the Orbita3d class and its registers, joints, motors and axis."""
from typing import Dict, List

from google.protobuf.wrappers_pb2 import FloatValue
from grpc import Channel
from reachy2_sdk_api.component_pb2 import PIDGains
from reachy2_sdk_api.kinematics_pb2 import ExtEulerAngles, Rotation3d
from reachy2_sdk_api.orbita3d_pb2 import (
    Float3d,
    Orbita3dCommand,
    Orbita3dState,
    PID3d,
    Vector3d,
)
from reachy2_sdk_api.orbita3d_pb2_grpc import Orbita3dServiceStub

from .orbita import Orbita
from .orbita_axis import OrbitaAxis
from .orbita_joint import OrbitaJoint
from .orbita_motor import OrbitaMotor
from .utils import unwrapped_proto_value


class Orbita3d(Orbita):
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

    def __init__(self, uid: int, name: str, initial_state: Orbita3dState, grpc_channel: Channel):
        """Initialize the Orbita2d with its joints, motors and axis."""
        super().__init__(uid, name, "3d", Orbita3dServiceStub(grpc_channel))
        init_state: Dict[str, Dict[str, FloatValue]] = self._create_init_state(initial_state)

        self._register_needing_sync: List[str] = []

        self._roll = OrbitaJoint(initial_state=init_state["roll"], axis_type="roll", actuator=self)
        self._pitch = OrbitaJoint(initial_state=init_state["pitch"], axis_type="pitch", actuator=self)
        self._yaw = OrbitaJoint(initial_state=init_state["yaw"], axis_type="yaw", actuator=self)
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

    def _create_init_state(self, initial_state: Orbita3dState) -> Dict[str, Dict[str, FloatValue]]:  # noqa: C901
        init_state: Dict[str, Dict[str, FloatValue]] = {}

        for field, value in initial_state.ListFields():
            if field.name == "compliant":
                self._compliant = unwrapped_proto_value(value)
                init_state["motor_1"][field.name] = value
                init_state["motor_2"][field.name] = value
                init_state["motor_3"][field.name] = value
            else:
                if isinstance(value, Rotation3d):
                    for joint in ["roll", "pitch", "yaw"]:
                        if joint not in init_state:
                            init_state[joint] = {}
                        init_state[joint][field.name] = getattr(value.rpy, joint)
                if isinstance(value, Float3d | PID3d):
                    for motor, val in value.ListFields():
                        if motor.name not in init_state:
                            init_state[motor.name] = {}
                        init_state[motor.name][field.name] = val
                if isinstance(value, Vector3d):
                    for axis, val in value.ListFields():
                        if axis.name not in init_state:
                            init_state[axis.name] = {}
                        init_state[axis.name][field.name] = val
        return init_state

    @property
    def roll(self) -> OrbitaJoint:
        return self._roll

    @property
    def pitch(self) -> OrbitaJoint:
        return self._pitch

    @property
    def yaw(self) -> OrbitaJoint:
        return self._yaw

    def _build_grpc_cmd_msg(self, field: str) -> Float3d:
        """Build a gRPC message from the registers that need to be synced at the joints and
        motors level. Registers can either be goal_position, pid or speed_limit/torque_limit.
        """
        if field == "goal_position":
            req = {}
            if len(self.roll._register_needing_sync) != 0:
                req["roll"] = self.roll._tmp_state["goal_position"]
                self.roll._register_needing_sync.clear()
            if len(self.pitch._register_needing_sync) != 0:
                req["pitch"] = self.pitch._tmp_state["goal_position"]
                self.pitch._register_needing_sync.clear()
            if len(self.yaw._register_needing_sync) != 0:
                req["yaw"] = self.yaw._tmp_state["goal_position"]
                self.yaw._register_needing_sync.clear()
            return Rotation3d(rpy=ExtEulerAngles(**req))

        elif field == "pid":
            return PID3d(
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

        return Float3d(
            motor_1=self.__motor_1._state[field],
            motor_2=self.__motor_2._state[field],
            motor_3=self.__motor_3._state[field],
        )

    def _build_grpc_cmd_msg_actuator(self, field: str) -> Float3d:
        """Build a gRPC message from the registers that need to be synced at the actuator level.
        Registers can either be compliant, pid, speed_limit or torque_limit."""
        if field == "pid":
            motor_1_gains = self.__motor_1._tmp_pid
            motor_2_gains = self.__motor_2._tmp_pid
            motor_3_gains = self.__motor_3._tmp_pid
            if type(motor_1_gains) is tuple and type(motor_2_gains) is tuple and type(motor_3_gains) is tuple:
                return PID3d(
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
        return Float3d(
            motor_1=FloatValue(value=motor_1_value),
            motor_2=FloatValue(value=motor_2_value),
            motor_3=FloatValue(value=motor_3_value),
        )

    def _pop_command(self) -> Orbita3dCommand:
        """Create a gRPC command from the registers that need to be synced."""
        values = self._make_command()

        command = Orbita3dCommand(**values)

        self._reset_registers()

        return command

    def _update_with(self, new_state: Orbita3dState) -> None:  # noqa: C901
        """Update the orbita with a newly received (partial) state received from the gRPC server."""
        state: Dict[str, Dict[str, FloatValue]] = {}

        for field, value in new_state.ListFields():
            if field.name == "compliant":
                self._compliant = unwrapped_proto_value(value)
                state["motor_1"][field.name] = value
                state["motor_2"][field.name] = value
                state["motor_3"][field.name] = value
            else:
                if isinstance(value, Rotation3d):
                    for joint in ["roll", "pitch", "yaw"]:
                        if joint not in state:
                            state[joint] = {}
                        state[joint][field.name] = getattr(value.rpy, joint)
                if isinstance(value, Float3d | PID3d):
                    for motor, val in value.ListFields():
                        if motor.name not in state:
                            state[motor.name] = {}
                        state[motor.name][field.name] = val
                if isinstance(value, Vector3d):
                    for axis, val in value.ListFields():
                        if axis.name not in state:
                            state[axis.name] = {}
                        state[axis.name][field.name] = val

        for name, motor in self._motors.items():
            motor._update_with(state[name])

        for name, axis in self._axis.items():
            axis._update_with(state[name])

        for name, joints in self._joints.items():
            joints._update_with(state[name])
