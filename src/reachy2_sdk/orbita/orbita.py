import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.component_pb2 import ComponentId
from reachy2_sdk_api.orbita2d_pb2 import Float2d, Orbita2dState, PID2d, Pose2d
from reachy2_sdk_api.orbita2d_pb2_grpc import Orbita2dServiceStub
from reachy2_sdk_api.orbita3d_pb2 import Float3d, Orbita3dState
from reachy2_sdk_api.orbita3d_pb2_grpc import Orbita3dServiceStub

from ..register import Register
from .orbita_motor import OrbitaMotor
from .utils import to_internal_position


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

    compliant = Register(readonly=False, type=BoolValue, label="compliant")

    def __init__(
        self,
        uid: int,
        name: str,
        orbita_type: str,
        stub: Orbita2dServiceStub | Orbita3dServiceStub,
    ):
        """Initialize the common attributes."""
        self.name = name
        self.id = uid
        self._orbita_type = orbita_type
        self._stub = stub

        self._state: Dict[str, bool] = {}
        self._register_needing_sync: List[str] = []
        self._joints: Dict[str, Any] = {}
        self._motors: Dict[str, OrbitaMotor] = {}

    @abstractmethod
    def _create_init_state(self, initial_state: Orbita2dState | Orbita3dState) -> Dict[str, Dict[str, FloatValue]]:
        pass

    def __repr__(self) -> str:
        """Clean representation of an Orbita."""
        s = "\n\t".join([str(joint) for joint in self._joints.values()])
        return f"""<Orbita{self._orbita_type} compliant={self.compliant} joints=\n\t{
            s
        }\n>"""

    def set_speed_limit(self, speed_limit: float) -> None:
        """Set a speed_limit on all motors of the actuator"""
        if not isinstance(speed_limit, float | int):
            raise ValueError(f"Expected one of: float, int for speed_limit, got {type(speed_limit).__name__}")
        speed_limit = to_internal_position(speed_limit)
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

    @property
    def temperatures(self) -> Dict[str, Register]:
        """Get temperatures of all the motors of the actuator"""
        return {motor_name: m.temperature for motor_name, m in self._motors.items()}

    def __setattr__(self, __name: str, __value: Any) -> None:
        """Set the value of the register."""
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
        """Set the value of the register for all motors of the actuator.

        It is used to set pid, speed_limit and torque_limit.
        """
        for m in self._motors.values():
            m._tmp_fields[field] = value

        self._update_loop(field)

    def _setup_sync_loop(self) -> None:
        """Set up the async synchronisation loop.

        The setup is done separately, as the async Event should be created in the same EventLoop than it will be used.

        The _need_sync Event is used to inform the robot that some data need to be pushed to the real robot.
        The _register_needing_sync stores a list of the register that need to be synced.
        """
        self._need_sync = asyncio.Event()
        self._loop = asyncio.get_running_loop()

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

    @abstractmethod
    def _build_grpc_cmd_msg_actuator(self, field: str) -> Float2d | Float3d:
        pass

    @abstractmethod
    def _build_grpc_cmd_msg(self, field: str) -> Pose2d | PID2d | Float2d | Float3d:
        pass

    def _make_command(self) -> Dict[str, Any]:
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

        return values

    def _reset_registers(self) -> None:
        self._register_needing_sync.clear()
        for obj in list(self._motors.values()):
            obj._register_needing_sync.clear()
        self._need_sync.clear()
