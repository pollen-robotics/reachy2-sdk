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
from logging import getLogger

import grpc

from typing import Optional

# from grpc._channel import _InactiveRpcError
from google.protobuf.empty_pb2 import Empty

from reachy_sdk_api_v2 import reachy_pb2_grpc

# from reachy_v2_sdk_api import config_pb2_grpc
from .reachy import ReachyInfo, get_config
from .arm import Arm
from .head import Head
from .hand import Hand

from .orbita2d import Orbita2DSDK
from .orbita3d import Orbita3DSDK


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

        self._ready = threading.Event()

        self._get_info()

        self._setup_actuators()
        self._setup_parts()

        self.l_arm: Optional[Arm] = None
        self.r_arm: Optional[Arm] = None
        self.head: Optional[Head] = None
        # self.mobile_base: Optional[MobileBase] = None

        # self._ready.wait()

    def __repr__(self) -> str:
        """Clean representation of a Reachy."""
        return f'<Reachy host="{self._host}">'
        # s = '\n\t'.join([str(j) for j in self._joints])
        # return f'''<Reachy host="{self._host}" joints=\n\t{
        #     s
        # }\n>'''

    def _get_info(self) -> None:
        config_stub = reachy_pb2_grpc.ReachyServiceStub(self._grpc_channel)
        self._robot = config_stub.GetReachy(Empty())
        self.info = ReachyInfo(self._host, self._robot.info)
        self.config = get_config(self._robot)

    def _setup_actuators(self) -> None:
        self._orbita2d_holder = Orbita2DSDK(self._host)
        self._orbita3d_holder = Orbita3DSDK(self._host)

        self._actuators_list = self._orbita2d_holder.get_list() + self._orbita3d_holder.get_list()

        self._actuators_dict = dict(zip([actuator.id for actuator in self._actuators_list], self._actuators_list))

    def _setup_parts(self) -> None:
        if self._robot.HasField("l_arm"):
            self.l_arm = Arm(self._grpc_channel, self._robot.l_arm)
            for articulation in self._robot.l_arm.DESCRIPTOR.fields:
                actuator = getattr(self._robot.l_arm, articulation.name)
                setattr(self.l_arm, articulation.name, self._actuators_dict[actuator.info.id])

            if self._robot.HasField("l_hand"):
                left_hand = Hand(self._grpc_channel, self._robot.l_hand)
                setattr(self.l_arm, "gripper", left_hand)

        if self._robot.HasField("r_arm"):
            self.r_arm = Arm(self._grpc_channel, self._robot.r_arm)
            for articulation in self._robot.l_arm.DESCRIPTOR.fields:
                actuator = getattr(self._robot.l_arm, articulation.name)
                setattr(self.l_arm, articulation.name, self._actuators_dict[actuator.info.id])

            if self._robot.HasField("r_hand"):
                right_hand = Hand(self._grpc_channel, self._robot.r_hand)
                setattr(self.r_arm, "gripper", right_hand)

        if self._robot.HasField("head"):
            self.head = Head(self._grpc_channel, self._robot.head)

        if self._robot.HasField("mobile_base"):
            pass

    def _start_sync_in_bg(self) -> None:
        # loop = asyncio.new_event_loop()
        # loop.run_until_complete(self._sync_loop())
        pass

    def turn_on(self) -> None:
        pass

    def turn_off(self) -> None:
        pass


def flush_communication() -> None:
    """Flush communication before leaving.

    We make sure all buffered commands have been sent before actually leaving.
    """
    print("Kezaco")


atexit.register(flush_communication)
