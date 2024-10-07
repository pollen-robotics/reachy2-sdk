import logging
import time
from abc import ABC, abstractmethod
from threading import Thread
from typing import Any, Dict, Optional, Tuple

import numpy as np
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.component_pb2 import ComponentId
from reachy2_sdk_api.orbita2d_pb2 import (
    Orbita2dCommand,
    Orbita2dsCommand,
    Orbita2dState,
    Orbita2dStatus,
)
from reachy2_sdk_api.orbita2d_pb2_grpc import Orbita2dServiceStub
from reachy2_sdk_api.orbita3d_pb2 import Orbita3dState, Orbita3dStatus
from reachy2_sdk_api.orbita3d_pb2_grpc import Orbita3dServiceStub

from ..parts.part import Part
from .orbita_axis import OrbitaAxis
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

    def __init__(self, uid: int, name: str, orbita_type: str, stub: Orbita2dServiceStub | Orbita3dServiceStub, part: Part):
        """Initialize the common attributes.

        Arguments:
        - uid: id of the actuator
        - name: name of the actuator
        - orbita_type: discriminate the orbita type, which can be "2d" or "3d"
        - stub: stub to call Orbitas methods
        - part: refers to the part the Orbita belongs to, in order to retrieve the parent part of the actuator.
        """
        self._logger = logging.getLogger(__name__)
        self._name = name
        self._id = uid
        self._orbita_type = orbita_type
        self._stub = stub
        self._part = part

        self._compliant: bool

        self._joints: Dict[str, Any] = {}
        self._axis_name_by_joint: Dict[Any, str] = {}
        self._motors: Dict[str, OrbitaMotor] = {}
        self._outgoing_goal_positions: Dict[str, float] = {}
        self._axis: Dict[str, OrbitaAxis] = {}

        self._error_status: Optional[str] = None

        self._thread_check_position: Optional[Thread] = None
        self._cancel_check = False

    @abstractmethod
    def _create_dict_state(self, initial_state: Orbita2dState | Orbita3dState) -> Dict[str, Dict[str, FloatValue]]:
        pass

    def __repr__(self) -> str:
        """Clean representation of an Orbita."""
        s = "\n\t".join([str(joint) for joint in self._joints.values()])
        return f"""<Orbita{self._orbita_type} on={self.is_on()} joints=\n\t{
            s
        }\n>"""

    @abstractmethod
    def set_speed_limits(self, speed_limit: float | int) -> None:
        if not isinstance(speed_limit, float | int):
            raise ValueError(f"Expected one of: float, int for speed_limit, got {type(speed_limit).__name__}")
        if not (0 <= speed_limit <= 100):
            raise ValueError(f"speed_limit must be in [0, 100], got {speed_limit}.")

    @abstractmethod
    def set_torque_limits(self, torque_limit: float | int) -> None:
        if not isinstance(torque_limit, float | int):
            raise ValueError(f"Expected one of: float, int for torque_limit, got {type(torque_limit).__name__}")
        if not (0 <= torque_limit <= 100):
            raise ValueError(f"torque_limit must be in [0, 100], got {torque_limit}.")

    # def set_pid(self, pid: Tuple[float, float, float]) -> None:
    #     """Set a pid value on all motors of the actuator"""
    #     if isinstance(pid, tuple) and len(pid) == 3 and all(isinstance(n, float | int) for n in pid):
    #         for m in self._motors.values():
    #             m._tmp_pid = pid
    #         self._update_loop("pid")
    #     else:
    #         raise ValueError("pid should be of type Tuple[float, float, float]")

    def get_speed_limits(self) -> Dict[str, float]:
        """Get speed_limit of all motors of the actuator, as a percentage of the max speed"""
        return {motor_name: m.speed_limit for motor_name, m in self._motors.items()}

    def get_torque_limits(self) -> Dict[str, float]:
        """Get torque_limit of all motors of the actuator, as a percentage of the max torque"""
        return {motor_name: m.torque_limit for motor_name, m in self._motors.items()}

    def get_pids(self) -> Dict[str, Tuple[float, float, float]]:
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

    def _post_send_goal_positions(self) -> None:
        if self._thread_check_position is None or not self._thread_check_position.is_alive():
            self._cancel_check = True
            if self._thread_check_position is not None:
                self._thread_check_position.join()
            self._thread_check_position = Thread(target=self._check_goal_positions, daemon=True)
            self._thread_check_position.start()

    def _check_goal_positions(self) -> None:
        """Send command does not return a status. Manual check of the present position vs the goal position"""
        self._cancel_check = False
        t1 = time.time()
        while time.time() - t1 < 1:
            time.sleep(0.05)
            if self._cancel_check:
                # in case of multiple send_goal_positions we'll check the next call
                return

        for joint, orbitajoint in self._joints.items():
            # precision is low we are looking for unreachable positions
            if not np.isclose(orbitajoint.present_position, orbitajoint.goal_position, atol=1):
                self._logger.warning(
                    f"{self._name}.{joint} has not reached the goal position ({orbitajoint.present_position} instead"
                    f" of {orbitajoint.goal_position})."
                )

    def _update_with(self, new_state: Orbita2dState | Orbita3dState) -> None:
        state: Dict[str, Dict[str, FloatValue]] = self._create_dict_state(new_state)

        for name, motor in self._motors.items():
            motor._update_with(state[name])

        for name, axis in self._axis.items():
            axis._update_with(state[name])

        for name, joints in self._joints.items():
            joints._update_with(state[name])

    @property
    def audit(self) -> Optional[str]:
        return self._error_status

    def _update_audit_status(self, new_status: Orbita2dStatus | Orbita3dStatus) -> None:
        self._error_status = new_status.errors[0].details
