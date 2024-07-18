from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.component_pb2 import ComponentId
from reachy2_sdk_api.orbita2d_pb2 import (
    Orbita2dCommand,
    Orbita2dsCommand,
    Orbita2dState,
)
from reachy2_sdk_api.orbita2d_pb2_grpc import Orbita2dServiceStub
from reachy2_sdk_api.orbita3d_pb2 import Orbita3dState
from reachy2_sdk_api.orbita3d_pb2_grpc import Orbita3dServiceStub

from .orbita_motor import OrbitaMotor


class Orbita(ABC):
    """The Orbita class is an abstract class to represent any Orbita actuator and its registers, joints, motors and axis.

    The Orbita class is used to store the up-to-date state of the actuator, especially:
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

    This class is meant to be derived by Orbita2d and Orbita3d
    """

    def __init__(
        self,
        uid: int,
        name: str,
        orbita_type: str,
        stub: Orbita2dServiceStub | Orbita3dServiceStub,
    ):
        """Initialize the common attributes."""
        self._name = name
        self._id = uid
        self._orbita_type = orbita_type
        self._stub = stub

        self._compliant: bool

        self._joints: Dict[str, Any] = {}
        self._axis_name_by_joint: Dict[Any, str] = {}
        self._motors: Dict[str, OrbitaMotor] = {}
        self._outgoing_goal_positions: Dict[str, float] = {}

    @abstractmethod
    def _create_init_state(self, initial_state: Orbita2dState | Orbita3dState) -> Dict[str, Dict[str, FloatValue]]:
        pass

    def __repr__(self) -> str:
        """Clean representation of an Orbita."""
        s = "\n\t".join([str(joint) for joint in self._joints.values()])
        return f"""<Orbita{self._orbita_type} on={self.is_on()} joints=\n\t{
            s
        }\n>"""

    @abstractmethod
    def set_speed_limit(self, speed_limit: float | int) -> None:
        if not isinstance(speed_limit, float | int):
            raise ValueError(f"Expected one of: float, int for speed_limit, got {type(speed_limit).__name__}")
        if not (0 < speed_limit < 100):
            raise ValueError(f"speed_limit must be in [0, 100], got {speed_limit}.")
        speed_limit = speed_limit / 100.0

    @abstractmethod
    def set_torque_limit(self, torque_limit: float | int) -> None:
        if not isinstance(torque_limit, float | int):
            raise ValueError(f"Expected one of: float, int for torque_limit, got {type(torque_limit).__name__}")
        if not (0 < torque_limit < 100):
            raise ValueError(f"torque_limit must be in [0, 100], got {torque_limit}.")
        torque_limit = torque_limit / 100.0

    # def set_pid(self, pid: Tuple[float, float, float]) -> None:
    #     """Set a pid value on all motors of the actuator"""
    #     if isinstance(pid, tuple) and len(pid) == 3 and all(isinstance(n, float | int) for n in pid):
    #         for m in self._motors.values():
    #             m._tmp_pid = pid
    #         self._update_loop("pid")
    #     else:
    #         raise ValueError("pid should be of type Tuple[float, float, float]")

    def get_speed_limit(self) -> Dict[str, float]:
        """Get speed_limit of all motors of the actuator"""
        return {motor_name: m.speed_limit for motor_name, m in self._motors.items()}

    def get_torque_limit(self) -> Dict[str, float]:
        """Get torque_limit of all motors of the actuator"""
        return {motor_name: m.torque_limit for motor_name, m in self._motors.items()}

    def get_pid(self) -> Dict[str, Tuple[float, float, float]]:
        """Get pid of all motors of the actuator"""
        return {motor_name: m.pid for motor_name, m in self._motors.items()}

    def turn_on(self) -> None:
        """Turn all motors of the orbita2d on.
        All orbita2d's motors will then be stiff.
        """
        self._set_compliant(False)

    def turn_off(self) -> None:
        """Turn all motors of the orbita2d on.
        All orbita2d's motors will then be stiff.
        """
        self._set_compliant(True)

    def is_on(self) -> bool:
        """Get compliancy of the actuator"""
        return not self._compliant

    @property
    def temperatures(self) -> Dict[str, float]:
        """Get temperatures of all the motors of the actuator"""
        return {motor_name: m.temperature for motor_name, m in self._motors.items()}

    def _set_compliant(self, compliant: bool) -> None:
        command = Orbita2dsCommand(
            cmd=[
                Orbita2dCommand(
                    id=ComponentId(id=self._id),
                    compliant=BoolValue(value=compliant),
                )
            ]
        )
        self._stub.SendCommand(command)

    def _set_outgoing_goal_position(self, axis_name: str, goal_position: float) -> None:
        joint = getattr(self, axis_name)
        axis = self._axis_name_by_joint[joint]
        self._outgoing_goal_positions[axis] = goal_position

    @abstractmethod
    def send_goal_positions(self) -> None:
        pass
