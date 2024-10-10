"""Reachy Orbita3d module.

Handles all specific methods to Orbita3d.
"""

from typing import Dict, List

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

from ..parts.part import Part
from .orbita import Orbita
from .orbita_axis import OrbitaAxis
from .orbita_joint import OrbitaJoint
from .orbita_motor import OrbitaMotor


class Orbita3d(Orbita):
    """The Orbita3d class represents any Orbita3d actuator and its joints, motors and axis.

    The Orbita3d class is used to store the up-to-date state of the actuator, especially:
        - its compliancy
        - its joints state
        - its motors state
        - its axis state

    You can access properties of the motors from the actuators with function that act on all the actuator's motors:
    - speed limit (in percentage, for all motors of the actuator)
    - torque limit (in percentage, for all motors of the actuator)
    - pid (for all motors of the actuator)
    - compliancy (for all motors of the actuator)
    Lower properties that are read-only but acessible at actuator level:
    - temperatures (temperatures of all motors of the actuator)
    """

    def __init__(
        self,
        uid: int,
        name: str,
        initial_state: Orbita3dState,
        grpc_channel: Channel,
        part: Part,
        joints_position_order: List[int],
    ):
        """Initialize the Orbita3d actuator with its joints, motors, and axes.

        Args:
            uid: The unique identifier for the actuator.
            name: The name of the actuator.
            initial_state: The initial state of the Orbita3d actuator, containing the states
                of the joints, motors, and axes.
            grpc_channel: The gRPC communication channel used for interfacing with the
                Orbita3d actuator.
            part: The robot part that this actuator belongs to.
            joints_position_order: A list defining the order of the joint positions in the
                containing part, used to map the actuator's joint positions correctly.
        """
        super().__init__(uid, name, "3d", Orbita3dServiceStub(grpc_channel), part)
        init_state: Dict[str, Dict[str, FloatValue]] = self._create_dict_state(initial_state)

        self._roll = OrbitaJoint(
            initial_state=init_state["roll"], axis_type="roll", actuator=self, position_order_in_part=joints_position_order[0]
        )
        self._pitch = OrbitaJoint(
            initial_state=init_state["pitch"], axis_type="pitch", actuator=self, position_order_in_part=joints_position_order[1]
        )
        self._yaw = OrbitaJoint(
            initial_state=init_state["yaw"], axis_type="yaw", actuator=self, position_order_in_part=joints_position_order[2]
        )
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
        """Create a dictionary representation of the state for the actuator.

        The method processes the fields in the given Orbita2dState and converts them into a nested dictionary
        structure, where the top-level keys are the axis, motor and joints names, and the inner dictionaries contain
        field names and corresponding FloatValue objects.

        Args:
            initial_state: An Orbita2dState object representing the initial state of the actuator.

        Returns:
            A dictionary where the keys represent the axis, motors and joints, and the values are dictionaries
            containing field names and corresponding FloatValue objects.

        Raises:
            ValueError: If the field type is not recognized or supported.
        """
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
        """Get the roll joint of the actuator."""
        return self._roll

    @property
    def pitch(self) -> OrbitaJoint:
        """Get the pitch joint of the actuator."""
        return self._pitch

    @property
    def yaw(self) -> OrbitaJoint:
        """Get the yaw joint of the actuator."""
        return self._yaw

    def send_goal_positions(self) -> None:
        """Send goal positions to the actuator's joints.

        If goal positions have been specified for any joint of this actuator, sends them to the actuator.
        """
        if self._outgoing_goal_positions:
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
            self._post_send_goal_positions()

    def set_speed_limits(self, speed_limit: float | int) -> None:
        """Set the speed limit as a percentage of the maximum speed for all motors of the actuator.

        Args:
            speed_limit: The desired speed limit as a percentage (0-100) of the maximum speed. Can be
                specified as a float or int.
        """
        super().set_speed_limits(speed_limit)
        speed_limit = speed_limit / 100.0
        command = Orbita3dsCommand(
            cmd=[
                Orbita3dCommand(
                    id=ComponentId(id=self._id),
                    speed_limit=Float3d(
                        motor_1=FloatValue(value=speed_limit),
                        motor_2=FloatValue(value=speed_limit),
                        motor_3=FloatValue(value=speed_limit),
                    ),
                )
            ]
        )
        self._stub.SendCommand(command)

    def set_torque_limits(self, torque_limit: float | int) -> None:
        """Set the torque limit as a percentage of the maximum torque for all motors of the actuator.

        Args:
            torque_limit: The desired torque limit as a percentage (0-100) of the maximum torque. Can be
                specified as a float or int.
        """
        super().set_torque_limits(torque_limit)
        torque_limit = torque_limit / 100.0
        command = Orbita3dsCommand(
            cmd=[
                Orbita3dCommand(
                    id=ComponentId(id=self._id),
                    torque_limit=Float3d(
                        motor_1=FloatValue(value=torque_limit),
                        motor_2=FloatValue(value=torque_limit),
                        motor_3=FloatValue(value=torque_limit),
                    ),
                )
            ]
        )
        self._stub.SendCommand(command)
