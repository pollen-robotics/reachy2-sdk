"""This module defines the Orbita2d class and its registers, joints, motors and axis."""
from typing import Any, Dict

from google.protobuf.wrappers_pb2 import FloatValue
from grpc import Channel
from reachy2_sdk_api.component_pb2 import ComponentId
from reachy2_sdk_api.orbita2d_pb2 import (
    Axis,
    Float2d,
    Orbita2dCommand,
    Orbita2dsCommand,
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
        self._axis_name_by_joint = {v: k for k, v in self._joints.items()}

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
                self._compliant = value.value
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

    def send_goal_positions(self) -> None:
        req_pos = {}
        for joint_axis in self._joints.keys():
            if joint_axis in self._outgoing_goal_positions:
                req_pos[joint_axis] = FloatValue(value=self._outgoing_goal_positions[joint_axis])
        pose = Pose2d(**req_pos)

        command = Orbita2dsCommand(
            cmd=[
                Orbita2dCommand(
                    id=ComponentId(id=self._id),
                    goal_position=pose,
                )
            ]
        )
        self._outgoing_goal_positions = {}
        self._stub.SendCommand(command)

    def set_speed_limits(self, speed_limit: float | int) -> None:
        """Set a speed_limit as a percentage of the max speed on all motors of the actuator"""
        super().set_speed_limits(speed_limit)
        speed_limit = speed_limit / 100.0
        command = Orbita2dsCommand(
            cmd=[
                Orbita2dCommand(
                    id=ComponentId(id=self._id),
                    speed_limit=Float2d(
                        motor_1=FloatValue(value=speed_limit),
                        motor_2=FloatValue(value=speed_limit),
                    ),
                )
            ]
        )
        self._stub.SendCommand(command)

    def set_torque_limits(self, torque_limit: float | int) -> None:
        """Set a torque_limit as a percentage of the max torque on all motors of the actuator"""
        super().set_torque_limits(torque_limit)
        torque_limit = torque_limit / 100.0
        command = Orbita2dsCommand(
            cmd=[
                Orbita2dCommand(
                    id=ComponentId(id=self._id),
                    torque_limit=Float2d(
                        motor_1=FloatValue(value=torque_limit),
                        motor_2=FloatValue(value=torque_limit),
                    ),
                )
            ]
        )
        print(command)
        self._stub.SendCommand(command)

    def _update_with(self, new_state: Orbita2dState) -> None:  # noqa: C901
        """Update the orbita with a newly received (partial) state received from the gRPC server."""
        state: Dict[str, Dict[str, FloatValue]] = {}

        for field, value in new_state.ListFields():
            if field.name == "compliant":
                self._compliant = value.value
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
