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

# from typing import List
from logging import getLogger

import grpc

from typing import Optional

# from grpc._channel import _InactiveRpcError
from google.protobuf.empty_pb2 import Empty

from reachy_sdk_api_v2 import reachy_pb2_grpc
from reachy_sdk_api_v2.orbita2d_pb2_grpc import Orbita2DServiceStub
from reachy_sdk_api_v2.orbita3d_pb2_grpc import Orbita3DServiceStub

# from reachy_v2_sdk_api import config_pb2_grpc
from .reachy import ReachyInfo, get_config
from .arm import Arm


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
        self._grpc_channel = grpc.insecure_channel(f"{self._host}:{self._sdk_port}")

        self.l_arm: Optional[Arm] = None
        self.r_arm: Optional[Arm] = None
        # self.head: Optional[Head] = None
        # self.mobile_base: Optional[MobileBase] = None

        self._ready = threading.Event()
        self._pushed_2dcommand = threading.Event()
        self._pushed_3dcommand = threading.Event()

        self._get_info()
        self._setup_parts()
        self._available_parts = []

    def __repr__(self) -> str:
        """Clean representation of a Reachy."""
        return f'<Reachy host="{self._host}">'

    def _get_info(self) -> None:
        config_stub = reachy_pb2_grpc.ReachyServiceStub(self._grpc_channel)
        self._robot = config_stub.GetReachy(Empty())
        self.info = ReachyInfo(self._host, self._robot.info)
        self.config = get_config(self._robot)

    def _setup_parts(self) -> None:
        if self._robot.HasField("r_arm"):
            self.r_arm = Arm(self._robot.r_arm, self._grpc_channel)
            self._available_parts.append(self.r_arm)
            # if self._robot.HasField("r_hand"):
            #     right_hand = Hand(self._grpc_channel, self._robot.r_hand)
            #     setattr(self.r_arm, "gripper", right_hand)

        # if self._robot.HasField("l_arm"):
        #     self.l_arm = Arm(self._grpc_channel, self._robot.l_arm)
        #     for articulation in self._robot.l_arm.DESCRIPTOR.fields:
        #         actuator = getattr(self._robot.l_arm, articulation.name)
        #         setattr(self.l_arm, articulation.name, self._actuators_dict[actuator.info.id])

        #     if self._robot.HasField("l_hand"):
        #         left_hand = Hand(self._grpc_channel, self._robot.l_hand)
        #         setattr(self.l_arm, "gripper", left_hand)

        # if self._robot.HasField("head"):
        #     self.head = Head(self._grpc_channel, self._robot.head)

        # if self._robot.HasField("mobile_base"):
        #     pass

    async def _poll_waiting_2dcommands(self):
        tasks = []

        for part in self._available_parts:
            for actuator, type in part._actuators.items():
                if type == "orbita2d":
                    tasks.append(asyncio.create_task(actuator._need_sync.wait()))

        await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        commands = [actuator._pop_command() for actuator in self._available_parts if actuator._need_sync.is_set()]
        return commands

    async def _poll_waiting_3dcommands(self):
        tasks = []

        for part in self._available_parts:
            for actuator, type in part._actuators.items():
                if type == "orbita2d":
                    tasks.append(asyncio.create_task(actuator._need_sync.wait()))

        await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        commands = [actuator._pop_command() for actuator in self._available_parts if actuator._need_sync.is_set()]
        return commands

    def _start_sync_in_bg(self) -> None:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self._sync_loop())

    async def _sync_loop(self) -> None:
        async_channel = grpc.aio.insecure_channel(f"{self._host}:{self._sdk_port}")
        # reachy_stub = reachy_pb2_grpc.ReachyServiceStub(async_channel)
        orbita2d_stub = Orbita2DServiceStub(async_channel)
        orbita3d_stub = Orbita3DServiceStub(async_channel)

        await asyncio.gather(
            self._stream_orbita2d_commands_loop(orbita2d_stub, freq=100),
            self._stream_orbita3d_commands_loop(orbita3d_stub, freq=100),
        )

    async def _stream_orbita2d_commands_loop(self, orbita2d_stub, freq: float):
        async def command_poll():
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

        await orbita2d_stub.StreamCommand(command_poll())

    async def _stream_orbita3d_commands_loop(self, orbita2d_stub, freq: float):
        async def command_poll():
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

        await orbita2d_stub.StreamCommand(command_poll())


def flush_communication() -> None:
    """Flush communication before leaving.

    We make sure all buffered commands have been sent before actually leaving.
    """
    print("Kezaco")


atexit.register(flush_communication)
