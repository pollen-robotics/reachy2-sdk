"""ReachySDK package.

This package provides remote access (via socket) to a Reachy robot.
It automatically handles the synchronization with the robot.
In particular, you can easily get an always up-to-date robot state (joint positions, sensors value).
You can also send joint commands, compute forward or inverse kinematics.

"""

import asyncio
import atexit
import threading
import time
from logging import getLogger
from typing import Any, Dict, List

import grpc
from google.protobuf.empty_pb2 import Empty
from grpc._channel import _InactiveRpcError
from reachy_sdk_api_v2 import reachy_pb2, reachy_pb2_grpc
from reachy_sdk_api_v2.orbita2d_pb2 import Orbita2DsCommand

# from reachy_sdk_api_v2.dynamixel_motor_pb2 import DynamixelMotorsCommand
from reachy_sdk_api_v2.orbita2d_pb2_grpc import Orbita2DServiceStub
from reachy_sdk_api_v2.orbita3d_pb2 import Orbita3DsCommand
from reachy_sdk_api_v2.orbita3d_pb2_grpc import Orbita3DServiceStub

from .arm import Arm
from .head import Head
from .orbita2d import Orbita2d
from .orbita3d import Orbita3d
from .reachy import ReachyInfo, get_config

# from reachy_sdk_api_v2.dynamixel_motor_pb2_grpc import DynamixelMotorServiceStub


# from .dynamixel_motor import DynamixelMotor


def singleton(cls: Any, *args: Any, **kw: Any) -> Any:
    instances = {}

    def _singleton(*args: Any, **kw: Any) -> Any:
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        else:
            raise ConnectionError("Cannot open 2 robot connections in the same kernel.")
        return instances[cls]

    return _singleton


@singleton
class ReachySDK:
    """The ReachySDK class handles the connection with your robot.

    It holds:

    - all joints (can be accessed directly via their name or via the joints list).
    - all force sensors (can be accessed directly via their name or via the force_sensors list).
    - all fans (can be accessed directly via their name or via the fans list).

    The synchronisation with the robot is automatically launched at instanciation and is handled in background automatically.
    """

    def __init__(
        self,
        host: str,
        sdk_port: int = 50051,
    ) -> None:
        """Set up the connection with the robot."""
        self._logger = getLogger()
        self._host = host
        self._sdk_port = sdk_port

        self._grpc_connected = False
        self.connect()

    def connect(self) -> None:
        if self._grpc_status == "connected":
            print("Already connected to Reachy.")
            return

        self._grpc_channel = grpc.insecure_channel(f"{self._host}:{self._sdk_port}")

        self._enabled_parts: Dict[str, Any] = {}
        self._disabled_parts: List[str] = []

        self._stop_flag = threading.Event()
        self._ready = threading.Event()
        self._pushed_2dcommand = threading.Event()
        self._pushed_3dcommand = threading.Event()
        # self._pushed_dmcommand = threading.Event()

        try:
            self._get_info()
        except ConnectionError:
            print(
                f"Could not connect to Reachy with on IP address {self._host}, check that the sdk server \
is running and that the IP is correct."
            )
            self._grpc_status = "disconnected"
            return

        self._setup_parts()

        self._sync_thread = threading.Thread(target=self._start_sync_in_bg)
        self._sync_thread.daemon = True
        self._sync_thread.start()

        self._grpc_status = "connected"
        _open_connection.append(self)
        print("Connected to Reachy.")

    def disconnect(self) -> None:
        if self._grpc_status == "disconnected":
            print("Already disconnected from Reachy.")
            return

        for part in self._enabled_parts.values():
            for actuator in part._actuators.values():
                actuator._need_sync.clear()

        self._stop_flag.set()
        time.sleep(0.1)
        self._grpc_status = "disconnected"

        self._grpc_channel.close()
        attributs = [attr for attr in dir(self) if not attr.startswith("_")]
        for attr in attributs:
            if attr not in [
                "grpc_status",
                "connect",
                "disconnect",
                "turn_on",
                "turn_off",
                "enabled_parts",
                "disabled_parts",
            ]:
                delattr(self, attr)

        for task in asyncio.all_tasks(loop=self._loop):
            task.cancel()

        print("Disconnected from Reachy.")

    def __repr__(self) -> str:
        """Clean representation of a Reachy."""
        s = "\n\t".join([part_name + ": " + str(part) for part_name, part in self._enabled_parts.items()])
        return f"""<Reachy host="{self._host}"\n grpc_status={self.grpc_status} \n enabled_parts=\n\t{
            s
        }\n\tdisabled_parts={self._disabled_parts}\n>"""

    @property
    def enabled_parts(self) -> List[str]:
        if self._grpc_status == "disconnected":
            print("Cannot get enabled parts, not connected to Reachy.")
            return []
        return list(self._enabled_parts.keys())

    @property
    def disabled_parts(self) -> List[str]:
        if self._grpc_status == "disconnected":
            print("Cannot get disabled parts, not connected to Reachy.")
            return []
        return self._disabled_parts

    @property
    def grpc_status(self) -> str:
        """Get the status of the connection with the robot.

        Can be either 'connected' or 'disconnected'.
        """
        return self._grpc_status

    @property
    def _grpc_status(self) -> str:
        if self._grpc_connected:
            return "connected"
        else:
            return "disconnected"

    @_grpc_status.setter
    def _grpc_status(self, status: str) -> None:
        if status == "connected":
            self._grpc_connected = True
        elif status == "disconnected":
            self._grpc_connected = False
        else:
            raise ValueError("_grpc_status can only be set to 'connected' or 'disconnected'")

    def _get_info(self) -> None:
        config_stub = reachy_pb2_grpc.ReachyServiceStub(self._grpc_channel)
        try:
            self._robot = config_stub.GetReachy(Empty())
        except _InactiveRpcError:
            raise ConnectionError()

        self.info = ReachyInfo(self._host, self._robot.info)
        self.config = get_config(self._robot)
        self._grpc_status = "connected"

    def _setup_parts(self) -> None:
        setup_stub = reachy_pb2_grpc.ReachyServiceStub(self._grpc_channel)
        initial_state = setup_stub.GetReachyState(self._robot.id)

        if self._robot.HasField("r_arm"):
            if initial_state.r_arm_state.activated:
                r_arm = Arm(self._robot.r_arm, initial_state.r_arm_state, self._grpc_channel)
                setattr(self, "r_arm", r_arm)
                self._enabled_parts["r_arm"] = getattr(self, "r_arm")
                # if self._robot.HasField("r_hand"):
                #     right_hand = Hand(self._grpc_channel, self._robot.r_hand)
                #     setattr(self.r_arm, "gripper", right_hand)
            else:
                self._disabled_parts.append("r_arm")

        if self._robot.HasField("l_arm"):
            if initial_state.l_arm_state.activated:
                l_arm = Arm(self._robot.l_arm, initial_state.l_arm_state, self._grpc_channel)
                setattr(self, "l_arm", l_arm)
                self._enabled_parts["l_arm"] = getattr(self, "l_arm")
            else:
                self._disabled_parts.append("l_arm")

        if self._robot.HasField("head"):
            if initial_state.head_state.activated:
                head = Head(self._robot.head, initial_state.head_state, self._grpc_channel)
                setattr(self, "head", head)
                self._enabled_parts["head"] = getattr(self, "head")
            else:
                self._disabled_parts.append("head")

        # if self._robot.HasField("mobile_base"):
        #     pass

    async def _wait_for_stop(self) -> None:
        while not self._stop_flag.is_set():
            await asyncio.sleep(0.1)
        raise ConnectionError("Connection with Reachy lost, check the sdk server status.")

    async def _poll_waiting_2dcommands(self) -> Orbita2DsCommand:
        tasks = []

        for part in self._enabled_parts.values():
            for actuator in part._actuators.values():
                if isinstance(actuator, Orbita2d):
                    # tasks.append(asyncio.create_task(actuator._need_sync.wait(), name=f"Task for {actuator.name}"))
                    tasks.append(asyncio.create_task(actuator._need_sync.wait()))

        if len(tasks) > 0:
            await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            commands = []

            for part in self._enabled_parts.values():
                for actuator in part._actuators.values():
                    if isinstance(actuator, Orbita2d) and actuator._need_sync.is_set():
                        commands.append(actuator._pop_command())

            return Orbita2DsCommand(cmd=commands)

        else:
            pass

    async def _poll_waiting_3dcommands(self) -> Orbita3DsCommand:
        tasks = []

        for part in self._enabled_parts.values():
            for actuator in part._actuators.values():
                if isinstance(actuator, Orbita3d):
                    tasks.append(asyncio.create_task(actuator._need_sync.wait()))
                    # tasks.append(asyncio.create_task(actuator._need_sync.wait(), name=f"Task for {actuator.name}"))

        if len(tasks) > 0:
            await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            commands = []

            for part in self._enabled_parts.values():
                for actuator in part._actuators.values():
                    if isinstance(actuator, Orbita3d) and actuator._need_sync.is_set():
                        commands.append(actuator._pop_command())

            return Orbita3DsCommand(cmd=commands)

        else:
            pass

    # async def _poll_waiting_dmcommands(self) -> DynamixelMotorsCommand:
    #     tasks = []

    #     for part in self._enabled_parts.values():
    #         for actuator in part._actuators.values():
    #             if isinstance(actuator, DynamixelMotor):
    #                 tasks.append(asyncio.create_task(actuator._need_sync.wait(), name=f"Task for {actuator.name}"))

    #     if len(tasks) > 0:
    #         await asyncio.wait(
    #             tasks,
    #             return_when=asyncio.FIRST_COMPLETED,
    #         )

    #         commands = []

    #         for part in self._enabled_parts.values():
    #             for actuator in part._actuators.values():
    #                 if isinstance(actuator, DynamixelMotor) and actuator._need_sync.is_set():
    #                     commands.append(actuator._pop_command())

    #         return DynamixelMotorsCommand(cmd=commands)

    #     else:
    #         pass

    def _start_sync_in_bg(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._sync_loop())
            self.disconnect()
        except asyncio.CancelledError:
            print("Sync loop cancelled.")

    async def _sync_loop(self) -> None:
        if hasattr(self, "r_arm"):
            for actuator in self.r_arm._actuators.values():
                actuator._setup_sync_loop()

        if hasattr(self, "l_arm"):
            for actuator in self.l_arm._actuators.values():
                actuator._setup_sync_loop()

        if hasattr(self, "head"):
            for actuator in self.head._actuators.values():
                actuator._setup_sync_loop()

        async_channel = grpc.aio.insecure_channel(f"{self._host}:{self._sdk_port}")
        reachy_stub = reachy_pb2_grpc.ReachyServiceStub(async_channel)
        orbita2d_stub = Orbita2DServiceStub(async_channel)
        orbita3d_stub = Orbita3DServiceStub(async_channel)
        # dynamixel_motor_stub = DynamixelMotorServiceStub(async_channel)

        try:
            await asyncio.gather(
                self._stream_orbita2d_commands_loop(orbita2d_stub, freq=100),
                self._stream_orbita3d_commands_loop(orbita3d_stub, freq=100),
                # self._stream_dynamixel_motor_commands_loop(dynamixel_motor_stub, freq=100),
                self._get_stream_update_loop(reachy_stub, freq=1),
                self._wait_for_stop(),
            )
        except ConnectionError:
            print("Connection with Reachy lost, check the sdk server status.")
        except asyncio.CancelledError:
            print("Stopped streaming commands.")

    async def _get_stream_update_loop(self, reachy_stub: reachy_pb2_grpc.ReachyServiceStub, freq: float) -> None:
        stream_req = reachy_pb2.ReachyStreamStateRequest(id=self._robot.id, publish_frequency=freq)
        try:
            async for state_update in reachy_stub.StreamReachyState(stream_req):
                if hasattr(self, "l_arm"):
                    self.l_arm._update_with(state_update.l_arm_state)
                    if hasattr(self.l_arm, "l_hand"):
                        self.l_arm.gripper._update_with(state_update.l_hand_state)
                if hasattr(self, "r_arm"):
                    self.r_arm._update_with(state_update.r_arm_state)
                    if hasattr(self, "r_hand"):
                        self.r_arm.gripper._update_with(state_update.r_hand_state)
                if hasattr(self, "head"):
                    self.head._update_with(state_update.head_state)
                if hasattr(self, "mobile_base"):
                    self.mobile_base._update_with(state_update.mobile_base_state)
        except grpc.aio._call.AioRpcError:
            raise ConnectionError("")

    async def _stream_orbita2d_commands_loop(self, orbita2d_stub: Orbita2DServiceStub, freq: float) -> None:
        async def command_poll_2d() -> Orbita2DsCommand:
            last_pub = 0.0
            dt = 1.0 / freq

            while True:
                elapsed_time = time.time() - last_pub
                if elapsed_time < dt:
                    await asyncio.sleep(dt - elapsed_time)

                commands = await self._poll_waiting_2dcommands()
                yield commands
                self._pushed_2dcommand.set()
                self._pushed_2dcommand.clear()
                last_pub = time.time()

        try:
            await orbita2d_stub.StreamCommand(command_poll_2d())
        except grpc.aio._call.AioRpcError:
            pass

    async def _stream_orbita3d_commands_loop(self, orbita3d_stub: Orbita3DServiceStub, freq: float) -> None:
        async def command_poll_3d() -> Orbita3DsCommand:
            last_pub = 0.0
            dt = 1.0 / freq

            while True:
                elapsed_time = time.time() - last_pub
                if elapsed_time < dt:
                    await asyncio.sleep(dt - elapsed_time)

                commands = await self._poll_waiting_3dcommands()
                yield commands
                self._pushed_3dcommand.set()
                self._pushed_3dcommand.clear()
                last_pub = time.time()

        try:
            await orbita3d_stub.StreamCommand(command_poll_3d())
        except grpc.aio._call.AioRpcError:
            pass

    # async def _stream_dynamixel_motor_commands_loop(self, dynamixel_motor_stub: DynamixelMotorServiceStub, freq: float) -> None:  # noqa: E501
    #     async def command_poll_dm() -> DynamixelMotorsCommand:
    #         last_pub = 0.0
    #         dt = 1.0 / freq

    #         while True:
    #             elapsed_time = time.time() - last_pub
    #             if elapsed_time < dt:
    #                 await asyncio.sleep(dt - elapsed_time)

    #             commands = await self._poll_waiting_dmcommands()
    #             yield commands
    #             self._pushed_dmcommand.set()
    #             self._pushed_dmcommand.clear()
    #             last_pub = time.time()

    #     await dynamixel_motor_stub.StreamCommand(command_poll_dm())

    def turn_on(self) -> None:
        if self._grpc_status == "disconnected":
            print("Cannot turn on Reachy, not connected.")
            return
        for part in self._enabled_parts.values():
            part.turn_on()

    def turn_off(self) -> None:
        if self._grpc_status == "disconnected":
            print("Cannot turn off Reachy, not connected.")
            return
        for part in self._enabled_parts.values():
            part.turn_off()


_open_connection: List[ReachySDK] = []


def flush_connection() -> None:
    """Flush communication before leaving.

    We make sure all buffered commands have been sent before actually leaving.
    Cancel any pending asyncio task.
    """
    for reachy in _open_connection:
        reachy._pushed_2dcommand.wait(timeout=0.5)
        reachy._pushed_3dcommand.wait(timeout=0.5)

        for task in asyncio.all_tasks(loop=reachy._loop):
            task.cancel()


atexit.register(flush_connection)
