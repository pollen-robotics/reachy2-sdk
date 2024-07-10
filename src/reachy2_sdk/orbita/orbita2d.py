"""This module defines the Orbita2d class and its registers, joints, motors and axis."""
from typing import Any, Dict

from google.protobuf.wrappers_pb2 import FloatValue
from grpc import Channel
from reachy2_sdk_api.component_pb2 import PIDGains
from reachy2_sdk_api.orbita2d_pb2 import (
    Axis,
    Float2d,
    Orbita2dCommand,
    Orbita2dState,
    PID2d,
    Pose2d,
    Vector2d,
)
from reachy2_sdk_api.orbita2d_pb2_grpc import Orbita2dServiceStub

from .orbita import Orbita
from .orbita_axis import OrbitaAxis
from .orbita_joint import OrbitaJoint
from .orbita_motor import OrbitaMotor
from .utils import unwrapped_proto_value


class Orbita2d(Orbita):
    """The Orbita2d class represents any Orbita2d actuator and its registers, joints, motors and axis.

    The Orbita2d class is used to store the up-to-date state of the actuator, especially:
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

    def __init__(
        self,
        uid: int,
        name: str,
        axis1: Axis,
        axis2: Axis,
        initial_state: Orbita2dState,
        grpc_channel: Channel,
    ):
        """Initialize the Orbita2d with its joints, motors and its two axis (either roll, pitch or yaw for both)."""
        super().__init__(uid, name, "2d", Orbita2dServiceStub(grpc_channel))

        axis1_name = Axis.DESCRIPTOR.values_by_number[axis1].name.lower()
        axis2_name = Axis.DESCRIPTOR.values_by_number[axis2].name.lower()

        init_state: Dict[str, Dict[str, FloatValue]] = self._create_init_state(initial_state)

        setattr(
            self,
            axis1_name,
            OrbitaJoint(initial_state=init_state["axis_1"], axis_type=axis1_name, actuator=self),
        )
        setattr(
            self,
            axis2_name,
            OrbitaJoint(initial_state=init_state["axis_2"], axis_type=axis2_name, actuator=self),
        )
        self._joints = {
            "axis_1": getattr(self, axis1_name),
            "axis_2": getattr(self, axis2_name),
        }

        self.__motor_1 = OrbitaMotor(initial_state=init_state["motor_1"], actuator=self)
        self.__motor_2 = OrbitaMotor(initial_state=init_state["motor_2"], actuator=self)
        self._motors = {"motor_1": self.__motor_1, "motor_2": self.__motor_2}

        self.__x = OrbitaAxis(initial_state=init_state["x"])
        self.__y = OrbitaAxis(initial_state=init_state["y"])
        self._axis = {"x": self.__x, "y": self.__y}

    def _create_init_state(self, initial_state: Orbita2dState) -> Dict[str, Dict[str, FloatValue]]:  # noqa: C901
        init_state: Dict[str, Dict[str, FloatValue]] = {}

        for field, value in initial_state.ListFields():
            if field.name == "compliant":
                self._compliant = unwrapped_proto_value(value)
                init_state["motor_1"][field.name] = value
                init_state["motor_2"][field.name] = value
            else:
                if isinstance(value, Pose2d):
                    for axis, val in value.ListFields():
                        if axis.name not in init_state:
                            init_state[axis.name] = {}
                        init_state[axis.name][field.name] = val
                if isinstance(value, Float2d | PID2d):
                    for motor, val in value.ListFields():
                        if motor.name not in init_state:
                            init_state[motor.name] = {}
                        init_state[motor.name][field.name] = val
                if isinstance(value, Vector2d):
                    for axis, val in value.ListFields():
                        if axis.name not in init_state:
                            init_state[axis.name] = {}
                        init_state[axis.name][field.name] = val
        return init_state

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in ["roll", "pitch", "yaw"]:
            if hasattr(self, __name):
                raise AttributeError(f"can't set attribute '{__name}'")
        super().__setattr__(__name, __value)

    def _build_grpc_cmd_msg(self, field: str) -> Pose2d | PID2d | Float2d:
        """Build a gRPC message from the registers that need to be synced at the joints and
        motors level. Registers can either be goal_position, pid or speed_limit/torque_limit.
        """
        if field == "goal_position":
            req = {}
            if len(self._joints["axis_1"]._register_needing_sync) != 0:
                req["axis_1"] = self._joints["axis_1"]._tmp_state["goal_position"]
                self._joints["axis_1"]._register_needing_sync.clear()
            if len(self._joints["axis_2"]._register_needing_sync) != 0:
                req["axis_2"] = self._joints["axis_2"]._tmp_state["goal_position"]
                self._joints["axis_2"]._register_needing_sync.clear()
            return Pose2d(**req)

        elif field == "pid":
            return PID2d(
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

        return Float2d(
            motor_1=self.__motor_1._state[field],
            motor_2=self.__motor_2._state[field],
        )

    def _build_grpc_cmd_msg_actuator(self, field: str) -> Float2d:
        """Build a gRPC message from the registers that need to be synced at the actuator level.
        Registers can either be compliant, pid, speed_limit or torque_limit."""
        if field == "pid":
            motor_1_gains = self.__motor_1._tmp_pid
            motor_2_gains = self.__motor_2._tmp_pid
            if type(motor_1_gains) is tuple and type(motor_2_gains) is tuple:
                return PID2d(
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
        return Float2d(
            motor_1=FloatValue(value=motor_1_value),
            motor_2=FloatValue(value=motor_2_value),
        )

    def _pop_command(self) -> Orbita2dCommand:
        """Create a gRPC command from the registers that need to be synced."""
        values = self._make_command()

        command = Orbita2dCommand(**values)

        self._reset_registers()

        return command

    def _update_with(self, new_state: Orbita2dState) -> None:
        """Update the orbita with a newly received (partial) state received from the gRPC server."""
        state: Dict[str, Dict[str, FloatValue]] = {}

        for field, value in new_state.ListFields():
            if field.name == "compliant":
                self._compliant = unwrapped_proto_value(value)
                state["motor_1"][field.name] = value
                state["motor_2"][field.name] = value
            else:
                if isinstance(value, Pose2d):
                    for axis, val in value.ListFields():
                        if axis.name not in state:
                            state[axis.name] = {}
                        state[axis.name][field.name] = val
                if isinstance(value, Float2d | PID2d):
                    for motor, val in value.ListFields():
                        if motor.name not in state:
                            state[motor.name] = {}
                        state[motor.name][field.name] = val
                if isinstance(value, Vector2d):
                    for axis, val in value.ListFields():
                        if axis.name not in state:
                            state[axis.name] = {}
                        state[axis.name][field.name] = val

        for name, motor in self._motors.items():
            motor._update_with(state[name])

        for name, axis in self._axis.items():
            axis._update_with(state[name])

        for name, joint in self._joints.items():
            joint._update_with(state[name])
