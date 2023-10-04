"""ReachySDK package.

This package provides remote access (via socket) to a Reachy robot.
It automatically handles the synchronization with the robot.
In particular, you can easily get an always up-to-date robot state (joint positions, sensors value).
You can also send joint commands, compute forward or inverse kinematics.

"""

import asyncio
import atexit

import threading

# import time
# from typing import List
from logging import getLogger

import grpc

# from typing import Optional

# from grpc._channel import _InactiveRpcError
from google.protobuf.empty_pb2 import Empty

from reachy_sdk_api_v2 import reachy_pb2, reachy_pb2_grpc

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

        # self.l_arm: Optional[Arm] = None
        # self.r_arm: Optional[Arm] = None
        # self.head: Optional[Head] = None
        # self.mobile_base: Optional[MobileBase] = None

        self._get_info()
        self._setup_parts()

        self._sync_thread = threading.Thread(target=self._start_sync_in_bg)
        self._sync_thread.daemon = True
        self._sync_thread.start()

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
            r_arm = Arm(self._robot.r_arm, self._grpc_channel)
            setattr(self, "r_arm", r_arm)
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

    def _start_sync_in_bg(self) -> None:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self._sync_loop())

    async def _sync_loop(self) -> None:
        async_channel = grpc.aio.insecure_channel(f"{self._host}:{self._sdk_port}")
        reachy_stub = reachy_pb2_grpc.ReachyServiceStub(async_channel)

        await self._get_stream_update_loop(reachy_stub, freq=100)

    async def _get_stream_update_loop(self, reachy_stub: reachy_pb2_grpc.ReachyServiceStub, freq: float) -> None:
        stream_req = reachy_pb2.ReachyStreamStateRequest(id=self._robot.id, publish_frequency=freq)
        async for state_update in reachy_stub.StreamReachyState(stream_req):
            if hasattr(self, "l_arm"):
                self.l_arm._update_with(state_update.l_arm_state)
                # if (hasattr(self.l_arm, 'l_hand')):
                #     self.l_arm.gripper._update_with(state_update.l_hand_state)
            if hasattr(self, "r_arm"):
                self.r_arm._update_with(state_update.r_arm_state)
                if hasattr(self, "r_hand"):
                    self.r_arm.gripper._update_with(state_update.r_hand_state)
            if hasattr(self, "head"):
                self.head._update_with(state_update.head_state)
            # if (hasattr(self, 'mobile_base')):
            # self.mobile_base._update_with(state_update.mobile_base_state)


def flush_communication() -> None:
    """Flush communication before leaving.

    We make sure all buffered commands have been sent before actually leaving.
    """
    print("Kezaco")


atexit.register(flush_communication)
