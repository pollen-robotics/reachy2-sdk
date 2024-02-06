"""ReachySDK package.

This package provides remote access (via socket) to a Reachy robot.
It automatically handles the synchronization with the robot.
In particular, you can easily get an always up-to-date robot state (joint positions, sensors value).
You can also send joint commands, compute forward or inverse kinematics.

"""

# from reachy2_sdk_api.dynamixel_motor_pb2_grpc import DynamixelMotorServiceStub
# from .dynamixel_motor import DynamixelMotor


import asyncio
import atexit
import threading
import time
from collections import namedtuple
from logging import getLogger
from typing import Any, Dict, List, Optional

import grpc
from google.protobuf.empty_pb2 import Empty
from grpc._channel import _InactiveRpcError
from mobile_base_sdk import MobileBaseSDK
from reachy2_sdk_api import reachy_pb2, reachy_pb2_grpc
from reachy2_sdk_api.goto_pb2 import GoalStatus, GoToAck, GoToGoalStatus, GoToId
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.orbita2d_pb2 import Orbita2dsCommand

# from reachy2_sdk_api.dynamixel_motor_pb2 import DynamixelMotorsCommand
from reachy2_sdk_api.orbita2d_pb2_grpc import Orbita2dServiceStub
from reachy2_sdk_api.orbita3d_pb2 import Orbita3dsCommand
from reachy2_sdk_api.orbita3d_pb2_grpc import Orbita3dServiceStub

from .config.reachy_info import ReachyInfo
from .media.audio import Audio
from .orbita.orbita2d import Orbita2d
from .orbita.orbita3d import Orbita3d
from .orbita.orbita_joint import OrbitaJoint
from .parts.arm import Arm
from .parts.head import Head
from .utils.singleton import Singleton
from .utils.utils import (
    arm_position_to_list,
    ext_euler_angles_to_list,
    get_interpolation_mode,
)

SimplifiedRequest = namedtuple("SimplifiedRequest", ["part", "goal_positions", "duration", "mode"])
"""Named tuple for easy access to request variables"""

GoToHomeId = namedtuple("GoToHomeId", ["head", "r_arm", "l_arm"])
"""Named tuple for easy access to goto request on full body"""


class ReachySDK(metaclass=Singleton):
    """The ReachySDK class handles the connection with your robot.
    Only one instance of this class can be created in a session.

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
        audio_port: int = 50063,
    ) -> None:
        """Set up the connection with the robot."""
        self._logger = getLogger(__name__)
        self._host = host
        self._sdk_port = sdk_port
        self._audio_port = audio_port

        self._grpc_connected = False

        # declared to help mypy. actually filled in self._setup_parts()
        self._r_arm: Optional[Arm] = None
        self._l_arm: Optional[Arm] = None
        self._head: Optional[Head] = None

        self.connect()

    def connect(self) -> None:
        """Connects the SDK to the server."""
        if self._grpc_connected:
            self._logger.warning("Already connected to Reachy.")
            return

        self._grpc_channel = grpc.insecure_channel(f"{self._host}:{self._sdk_port}")

        self._stop_flag = threading.Event()
        self._ready = threading.Event()
        self._pushed_2dcommand = threading.Event()
        self._pushed_3dcommand = threading.Event()
        # self._pushed_dmcommand = threading.Event()

        try:
            self._get_info()
        except ConnectionError:
            self._logger.error(
                f"Could not connect to Reachy with on IP address {self._host}, check that the sdk server \
is running and that the IP is correct."
            )
            self._grpc_connected = False
            return

        self._setup_parts()
        # self._setup_audio()

        self._sync_thread = threading.Thread(target=self._start_sync_in_bg)
        self._sync_thread.daemon = True
        self._sync_thread.start()

        self._grpc_connected = True
        _open_connection.append(self)
        self._logger.info("Connected to Reachy.")

    def disconnect(self) -> None:
        """Disconnects the SDK from the server."""
        if not self._grpc_connected:
            self._logger.warning("Already disconnected from Reachy.")
            return

        for part in self.info._enabled_parts.values():
            for actuator in part._actuators.values():
                actuator._need_sync.clear()

        self._stop_flag.set()
        time.sleep(0.1)
        self._grpc_connected = False

        self._grpc_channel.close()
        attributs = [attr for attr in dir(self) if not attr.startswith("_")]
        for attr in attributs:
            if attr not in [
                "is_connected",
                "connect",
                "disconnect",
                "home",
                "turn_on",
                "turn_off",
                "enabled_parts",
                "disabled_parts",
                "joints",
                "actuators",
                "head",
                "r_arm",
                "l_arm",
                "cancel_all_moves",
                "cancel_move_by_id",
                "_get_move_state",
                "get_move_joints_request",
                "is_move_finished",
                "is_move_playing",
            ]:
                delattr(self, attr)

        self._head = None
        self._r_arm = None
        self._l_arm = None

        for task in asyncio.all_tasks(loop=self._loop):
            task.cancel()

        self._logger.info("Disconnected from Reachy.")

    def __repr__(self) -> str:
        """Clean representation of a Reachy."""
        s = "\n\t".join([part_name + ": " + str(part) for part_name, part in self.info._enabled_parts.items()])
        return f"""<Reachy host="{self._host}"\n connected={self._grpc_connected} \n enabled_parts=\n\t{
            s
        }\n\tdisabled_parts={self.info._disabled_parts}\n>"""

    @property
    def head(self) -> Optional[Head]:
        """Get Reachy's head."""
        if self._head is None:
            raise AttributeError("head does not exist with this configuration")
        return self._head

    @property
    def r_arm(self) -> Optional[Arm]:
        """Get Reachy's right arm."""
        if self._r_arm is None:
            raise AttributeError("r_arm does not exist with this configuration")
        return self._r_arm

    @property
    def l_arm(self) -> Optional[Arm]:
        """Get Reachy's left arm."""
        if self._l_arm is None:
            raise AttributeError("l_arm does not exist with this configuration")
        return self._l_arm

    @property
    def joints(self) -> Dict[str, OrbitaJoint]:
        """Get all joints of the robot."""
        if not self._grpc_connected:
            self._logger.warning("Cannot get joints, not connected to Reachy.")
            return {}
        _joints: Dict[str, OrbitaJoint] = {}
        for part_name in self.info._enabled_parts:
            part = getattr(self, part_name)
            for joint_name, joint in part.joints.items():
                _joints[part_name + "_" + joint_name] = joint
        return _joints

    @property
    def actuators(self) -> Dict[str, Orbita2d | Orbita3d]:
        """Get all actuators of the robot."""
        if not self._grpc_connected:
            self._logger.warning("Cannot get actuators, not connected to Reachy.")
            return {}
        _actuators: Dict[str, Orbita2d | Orbita3d] = {}
        for part_name in self.info._enabled_parts:
            part = getattr(self, part_name)
            for actuator_name, actuator in part.actuators.items():
                _actuators[part_name + "_" + actuator_name] = actuator
        return _actuators

    def is_connected(self) -> bool:
        """Get the status of the connection with the robot server.

        Can be either 'connected' or 'disconnected'.
        """
        return self._grpc_connected

    @property
    def _grpc_status(self) -> str:
        """Get the status of the connection with the robot server."""
        if self._grpc_connected:
            return "connected"
        else:
            return "disconnected"

    @_grpc_status.setter
    def _grpc_status(self, status: str) -> None:
        """Set the status of the connection with the robot server."""
        if status == "connected":
            self._grpc_connected = True
        elif status == "disconnected":
            self._grpc_connected = False
        else:
            raise ValueError("_grpc_status can only be set to 'connected' or 'disconnected'")

    def _get_info(self) -> None:
        """Get main description of the robot.

        First connection to the robot. Information get:
        - robot's parts
        - robot's sofware and hardware version
        - robot's serial number
        """
        config_stub = reachy_pb2_grpc.ReachyServiceStub(self._grpc_channel)
        try:
            self._robot = config_stub.GetReachy(Empty())
        except _InactiveRpcError:
            raise ConnectionError()

        self.info = ReachyInfo(self._robot)
        self._grpc_connected = True

    def _setup_audio(self) -> None:
        """Internal function to set up the audio server."""
        try:
            self.audio = Audio(self._host, self._audio_port)
        except Exception:
            self._logger.error("Failed to connect to audio server. ReachySDK.audio will not be available.")

    def _setup_parts(self) -> None:
        """Setup all parts of the robot.

        Get the state of each part of the robot, create an instance for each of them and add
        it to the ReachySDK instance.
        """
        setup_stub = reachy_pb2_grpc.ReachyServiceStub(self._grpc_channel)
        self._goto_stub = GoToServiceStub(self._grpc_channel)
        initial_state = setup_stub.GetReachyState(self._robot.id)

        if self._robot.HasField("r_arm"):
            if initial_state.r_arm_state.activated:
                r_arm = Arm(self._robot.r_arm, initial_state.r_arm_state, self._grpc_channel, self._goto_stub)
                self._r_arm = r_arm
                self.info._enabled_parts["r_arm"] = self._r_arm
                if self._robot.HasField("r_hand"):
                    self._r_arm._init_hand(self._robot.r_hand, initial_state.r_hand_state)
            else:
                self.info._disabled_parts.append("r_arm")

        if self._robot.HasField("l_arm"):
            if initial_state.l_arm_state.activated:
                l_arm = Arm(self._robot.l_arm, initial_state.l_arm_state, self._grpc_channel, self._goto_stub)
                self._l_arm = l_arm
                self.info._enabled_parts["l_arm"] = self._l_arm
                if self._robot.HasField("l_hand"):
                    self._l_arm._init_hand(self._robot.l_hand, initial_state.l_hand_state)
            else:
                self.info._disabled_parts.append("l_arm")

        if self._robot.HasField("head"):
            if initial_state.head_state.activated:
                head = Head(self._robot.head, initial_state.head_state, self._grpc_channel, self._goto_stub)
                self._head = head
                self.info._enabled_parts["head"] = self._head
            else:
                self.info._disabled_parts.append("head")

        if self._robot.HasField("mobile_base"):
            self.mobile_base = MobileBaseSDK(self._host)

    async def _wait_for_stop(self) -> None:
        while not self._stop_flag.is_set():
            await asyncio.sleep(0.1)
        raise ConnectionError("Connection with Reachy lost, check the sdk server status.")

    async def _poll_waiting_2dcommands(self) -> Orbita2dsCommand:
        """Poll registers to update for Orbita2d actuators of the robot."""
        tasks = []

        for part in self.info._enabled_parts.values():
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

            for part in self.info._enabled_parts.values():
                for actuator in part._actuators.values():
                    if isinstance(actuator, Orbita2d) and actuator._need_sync.is_set():
                        commands.append(actuator._pop_command())

            return Orbita2dsCommand(cmd=commands)

        else:
            pass

    async def _poll_waiting_3dcommands(self) -> Orbita3dsCommand:
        """Poll registers to update for Orbita3d actuators of the robot."""
        tasks = []

        for part in self.info._enabled_parts.values():
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

            for part in self.info._enabled_parts.values():
                for actuator in part._actuators.values():
                    if isinstance(actuator, Orbita3d) and actuator._need_sync.is_set():
                        commands.append(actuator._pop_command())

            return Orbita3dsCommand(cmd=commands)

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
        """Start the synchronization asyncio tasks with the robot in background."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._sync_loop())
            self.disconnect()
        except asyncio.CancelledError:
            self._logger.error("Sync loop cancelled.")

    async def _sync_loop(self) -> None:
        """Define the synchronization loop.

        The synchronization loop is used to:
            - stream commands to the robot
            - update the state of the robot
        """
        if self._r_arm is not None:
            for actuator in self._r_arm._actuators.values():
                actuator._setup_sync_loop()

        if self._l_arm is not None:
            for actuator in self._l_arm._actuators.values():
                actuator._setup_sync_loop()

        if self._head is not None:
            for actuator in self._head._actuators.values():
                actuator._setup_sync_loop()

        async_channel = grpc.aio.insecure_channel(f"{self._host}:{self._sdk_port}")
        reachy_stub = reachy_pb2_grpc.ReachyServiceStub(async_channel)
        orbita2d_stub = Orbita2dServiceStub(async_channel)
        orbita3d_stub = Orbita3dServiceStub(async_channel)
        # dynamixel_motor_stub = DynamixelMotorServiceStub(async_channel)

        try:
            await asyncio.gather(
                self._stream_orbita2d_commands_loop(orbita2d_stub, freq=80),
                self._stream_orbita3d_commands_loop(orbita3d_stub, freq=80),
                # self._stream_dynamixel_motor_commands_loop(dynamixel_motor_stub, freq=100),
                self._get_stream_update_loop(reachy_stub, freq=100),
                self._wait_for_stop(),
            )
        except ConnectionError:
            self._logger.error("Connection with Reachy lost, check the sdk server status.")
        except asyncio.CancelledError:
            self._logger.error("Stopped streaming commands.")

    async def _get_stream_update_loop(self, reachy_stub: reachy_pb2_grpc.ReachyServiceStub, freq: float) -> None:
        """Update the state of the robot at a given frequency."""
        stream_req = reachy_pb2.ReachyStreamStateRequest(id=self._robot.id, publish_frequency=freq)
        try:
            async for state_update in reachy_stub.StreamReachyState(stream_req):
                if self._l_arm is not None:
                    self._l_arm._update_with(state_update.l_arm_state)
                    if hasattr(self._l_arm, "gripper"):
                        self._l_arm.gripper._update_with(state_update.l_hand_state)
                if self._r_arm is not None:
                    self._r_arm._update_with(state_update.r_arm_state)
                    if hasattr(self._r_arm, "gripper"):
                        self._r_arm.gripper._update_with(state_update.r_hand_state)
                if self._head is not None:
                    self._head._update_with(state_update.head_state)
        except grpc.aio._call.AioRpcError:
            raise ConnectionError("")

    async def _stream_orbita2d_commands_loop(self, orbita2d_stub: Orbita2dServiceStub, freq: float) -> None:
        """Stream commands for the 2d actuators of the robot at a given frequency.

        Poll the waiting commands at a given frequency and stream them to the server.
        Catch if the server is not reachable anymore and set the status of the connection to 'disconnected'.
        """

        async def command_poll_2d() -> Orbita2dsCommand:
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

    async def _stream_orbita3d_commands_loop(self, orbita3d_stub: Orbita3dServiceStub, freq: float) -> None:
        """Stream commands for the 3d actuators of the robot at a given frequency.

        Poll the waiting commands at a given frequency and stream them to the server.
        Catch if the server is not reachable anymore and set the status of the connection to 'disconnected'.
        """

        async def command_poll_3d() -> Orbita3dsCommand:
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

    def turn_on(self) -> bool:
        """Turn all motors of enabled parts on.

        All enabled parts' motors will then be stiff.
        """
        if not self._grpc_connected:
            self._logger.warning("Cannot turn on Reachy, not connected.")
            return False
        for part in self.info._enabled_parts.values():
            part.turn_on()

        return True

    def turn_off(self) -> bool:
        """Turn all motors of enabled parts off.

        All enabled parts' motors will then be compliant.
        """
        if not self._grpc_connected:
            self._logger.warning("Cannot turn off Reachy, not connected.")
            return False
        for part in self.info._enabled_parts.values():
            part.turn_off()

        return True

    def is_on(self) -> bool:
        """Return True if all actuators of the arm are stiff"""
        for part in self.info._enabled_parts.values():
            if not part.is_on():
                return False
        return True

    def is_off(self) -> bool:
        """Return True if all actuators of the arm are stiff"""
        for part in self.info._enabled_parts.values():
            if part.is_on():
                return False
        return True

    def home(self, wait_for_goto_end: bool = True, duration: float = 2, interpolation_mode: str = "minimum_jerk") -> GoToHomeId:
        """Send all joints to 0 in specified duration.

        Setting wait_for_goto_end to False will cancel all gotos on all parts and immediately send the 0 commands.
        Otherwise, the 0 commands will be sent to a part when all gotos of its queue has been played.
        """
        head_id = None
        r_arm_id = None
        l_arm_id = None
        if not wait_for_goto_end:
            self.cancel_all_moves()
        if self.head is not None:
            head_id = self.head.rotate_to(0, 0, 0, duration, interpolation_mode)
        if self.r_arm is not None:
            r_arm_id = self.r_arm.goto_joints([0, 0, 0, 0, 0, 0, 0], duration, interpolation_mode)
        if self.l_arm is not None:
            l_arm_id = self.l_arm.goto_joints([0, 0, 0, 0, 0, 0, 0], duration, interpolation_mode)
        ids = GoToHomeId(
            head=head_id,
            r_arm=r_arm_id,
            l_arm=l_arm_id,
        )
        return ids

    def is_move_finished(self, id: GoToId) -> bool:
        """Return True if goto has been played and has been cancelled, False otherwise."""
        state = self._get_move_state(id)
        result = bool(
            state.goal_status == GoalStatus.STATUS_ABORTED
            or state.goal_status == GoalStatus.STATUS_CANCELED
            or state.goal_status == GoalStatus.STATUS_SUCCEEDED
        )
        return result

    def is_move_playing(self, id: GoToId) -> bool:
        """Return True if goto is currently playing, False otherwise."""
        state = self._get_move_state(id)
        return bool(state.goal_status == GoalStatus.STATUS_EXECUTING)

    def cancel_all_moves(self) -> GoToAck:
        """Cancel all the goto tasks."""
        response = self._goto_stub.CancelAllGoTo(Empty())
        return response

    def _get_move_state(self, goto_id: GoToId) -> GoToGoalStatus:
        """Return the current state of a goto, given its id."""
        response = self._goto_stub.GetGoToState(goto_id)
        return response

    def cancel_move_by_id(self, goto_id: GoToId) -> GoToAck:
        """Ask the cancellation of a single goto on the arm, given its id"""
        response = self._goto_stub.CancelGoTo(goto_id)
        return response

    def get_move_joints_request(self, goto_id: GoToId) -> SimplifiedRequest:
        """Returns the part affected, the joints goal positions, duration and mode of the corresponding GoToId

        Part can be either 'r_arm', 'l_arm' or 'head'
        Goal_position is returned as a list in degrees
        """
        response = self._goto_stub.GetGoToRequest(goto_id)
        if response.joints_goal.HasField("arm_joint_goal"):
            part = response.joints_goal.arm_joint_goal.id.name
            mode = get_interpolation_mode(response.interpolation_mode.interpolation_type)
            goal_positions = arm_position_to_list(response.joints_goal.arm_joint_goal.joints_goal, degrees=True)
            duration = response.joints_goal.arm_joint_goal.duration.value
        elif response.joints_goal.HasField("neck_joint_goal"):
            part = response.joints_goal.neck_joint_goal.id.name
            mode = get_interpolation_mode(response.interpolation_mode.interpolation_type)
            goal_positions = ext_euler_angles_to_list(
                response.joints_goal.neck_joint_goal.joints_goal.rotation.rpy, degrees=True
            )
            duration = response.joints_goal.neck_joint_goal.duration.value

        request = SimplifiedRequest(
            part=part,
            goal_positions=goal_positions,
            duration=duration,
            mode=mode,
        )
        return request


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
