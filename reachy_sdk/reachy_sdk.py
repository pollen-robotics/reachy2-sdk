"""ReachySDK package.

This package provides remote access (via socket) to a Reachy robot.
It automatically handles the synchronization with the robot.
In particular, you can easily get an always up-to-date robot state (joint positions, sensors value).
You can also send joint commands, compute forward or inverse kinematics.

"""

# import asyncio
import atexit
import threading

# import time
# from typing import List
from enum import Enum
from logging import getLogger

import grpc

# from grpc._channel import _InactiveRpcError
from google.protobuf.empty_pb2 import Empty

from reachy_sdk_api import config_pb2_grpc


class ReachyParts(Enum):
    """Reachy parts options."""

    REACHY = 1
    L_ARM = 2
    R_ARM = 3
    HEAD = 4


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
        sdk_port: int = 50055,
    ) -> None:
        """Set up the connection with the robot."""
        self._logger = getLogger()
        self._host = host
        self._sdk_port = sdk_port
        self._grpc_channel = grpc.insecure_channel(f"{self._host}:{self._sdk_port}")

        self._ready = threading.Event()

        self._get_config()

        self._ready.wait()

    def __repr__(self) -> str:
        """Clean representation of a Reachy."""
        return f'<Reachy host="{self._host}">'
        # s = '\n\t'.join([str(j) for j in self._joints])
        # return f'''<Reachy host="{self._host}" joints=\n\t{
        #     s
        # }\n>'''

    def _get_config(self) -> None:
        config_stub = config_pb2_grpc.ConfigServiceStub(self._grpc_channel)
        self._config = config_stub.GetReachyConfig(Empty()).config

    def _start_sync_in_bg(self) -> None:
        # loop = asyncio.new_event_loop()
        # loop.run_until_complete(self._sync_loop())
        pass


def flush_communication() -> None:
    """Flush communication before leaving.

    We make sure all buffered commands have been sent before actually leaving.
    """
    print("Kezaco")


atexit.register(flush_communication)
