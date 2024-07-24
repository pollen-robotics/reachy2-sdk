"""This module defines the Orbita3d class and its registers, joints, motors and axis."""
from typing import Dict

from google.protobuf.wrappers_pb2 import FloatValue
from grpc import Channel
from reachy2_sdk_api.component_pb2 import ComponentId
from reachy2_sdk_api.kinematics_pb2 import ExtEulerAngles, Rotation3d
from reachy2_sdk_api.orbita3d_pb2 import (
    Float3d,
    Orbita3dCommand,
    Orbita3dsCommand,
    Orbita3dState,
    PID3d,
    Vector3d,
)
from reachy2_sdk_api.orbita3d_pb2_grpc import Orbita3dServiceStub

from .orbita import Orbita
from .orbita_axis import OrbitaAxis
from .orbita_joint import OrbitaJoint
from .orbita_motor import OrbitaMotor


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
        init_state: Dict[str, Dict[str, FloatValue]] = self._create_dict_state(initial_state)

        self._roll = OrbitaJoint(initial_state=init_state["roll"], axis_type="roll", actuator=self)
        self._pitch = OrbitaJoint(initial_state=init_state["pitch"], axis_type="pitch", actuator=self)
        self._yaw = OrbitaJoint(initial_state=init_state["yaw"], axis_type="yaw", actuator=self)
        self._joints = {"roll": self.roll, "pitch": self.pitch, "yaw": self.yaw}
        self._axis_name_by_joint = {v: k for k, v in self._joints.items()}

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

    def _create_dict_state(self, initial_state: Orbita3dState) -> Dict[str, Dict[str, FloatValue]]:  # noqa: C901
        init_state: Dict[str, Dict[str, FloatValue]] = {}

        for field, value in initial_state.ListFields():
            if field.name == "compliant":
                self._compliant = value.value
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

    def send_goal_positions(self) -> None:
        req_pos = {}
        for joint_axis in self._joints.keys():
            if joint_axis in self._outgoing_goal_positions:
                req_pos[joint_axis] = FloatValue(value=self._outgoing_goal_positions[joint_axis])
        pose = Rotation3d(rpy=ExtEulerAngles(**req_pos))

        command = Orbita3dsCommand(
            cmd=[
                Orbita3dCommand(
                    id=ComponentId(id=self._id),
                    goal_position=pose,
                )
            ]
        )
        self._outgoing_goal_positions = {}
        self._stub.SendCommand(command)
