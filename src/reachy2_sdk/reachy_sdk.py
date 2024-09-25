"""ReachySDK package.

This package provides remote access (via socket) to a Reachy robot.
It automatically handles the synchronization with the robot.
In particular, you can easily get an always up-to-date robot state (joint positions, sensors value).
You can also send joint commands, compute forward or inverse kinematics.

"""

# from reachy2_sdk_api.dynamixel_motor_pb2_grpc import DynamixelMotorServiceStub
# from .dynamixel_motor import DynamixelMotor

import threading
import time
from collections import namedtuple
from logging import getLogger
from typing import Dict, Optional

import grpc
from google.protobuf.empty_pb2 import Empty
from grpc._channel import _InactiveRpcError
from reachy2_sdk_api import reachy_pb2, reachy_pb2_grpc
from reachy2_sdk_api.goto_pb2 import GoalStatus, GoToAck, GoToGoalStatus, GoToId
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.reachy_pb2 import ReachyState

from .config.reachy_info import ReachyInfo
from .media.audio import Audio
from .media.camera_manager import CameraManager
from .orbita.orbita2d import Orbita2d
from .orbita.orbita3d import Orbita3d
from .orbita.orbita_joint import OrbitaJoint
from .parts.arm import Arm
from .parts.head import Head
from .parts.joints_based_part import JointsBasedPart
from .parts.mobile_base import MobileBase
from .utils.custom_dict import CustomDict
from .utils.utils import (
    SimplifiedRequest,
    arm_position_to_list,
    ext_euler_angles_to_list,
    get_interpolation_mode,
)

GoToHomeId = namedtuple("GoToHomeId", ["head", "r_arm", "l_arm"])
"""Named tuple for easy access to goto request on full body"""


class ReachySDK:
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
        video_port: int = 50065,
    ) -> None:
        """Set up the connection with the robot."""
        self._logger = getLogger(__name__)
        self._host = host
        self._sdk_port = sdk_port
        self._audio_port = audio_port
        self._video_port = video_port

        self._grpc_connected = False

        # declared to help mypy. actually filled in self._setup_parts()
        self._r_arm: Optional[Arm] = None
        self._l_arm: Optional[Arm] = None
        self._head: Optional[Head] = None
        self._cameras: Optional[CameraManager] = None
        self._mobile_base: Optional[MobileBase] = None
        self._info: Optional[ReachyInfo] = None

        self.connect()

    def connect(self) -> None:
        """Connects the SDK to the server."""
        if self._grpc_connected:
            self._logger.warning("Already connected to Reachy.")
            return

        self._grpc_channel = grpc.insecure_channel(f"{self._host}:{self._sdk_port}")

        self._stop_flag = threading.Event()

        try:
            self._get_info()
        except ConnectionError:
            self._logger.error(
                f"Could not connect to Reachy with on IP address {self._host}, "
                "check that the sdk server is running and that the IP is correct."
            )
            self._grpc_connected = False
            return

        self._setup_parts()
        # self._setup_audio()
        self._cameras = self._setup_video()

        self._sync_thread = threading.Thread(target=self._start_sync_in_bg)
        self._sync_thread.daemon = True
        self._sync_thread.start()

        self._audit_thread = threading.Thread(target=self._audit)
        self._audit_thread.daemon = True
        self._audit_thread.start()

        self._grpc_connected = True
        self._logger.info("Connected to Reachy.")

    def disconnect(self, lost_connection: bool = False) -> None:
        """Disconnects the SDK from the server."""
        if not self._grpc_connected:
            self._logger.warning("Already disconnected from Reachy.")
            return

        self._grpc_connected = False
        self._grpc_channel.close()

        self._head = None
        self._r_arm = None
        self._l_arm = None
        self._mobile_base = None

        self._logger.info("Disconnected from Reachy.")

    def __repr__(self) -> str:
        """Clean representation of a Reachy."""

        if not self._grpc_connected or self.info is None:
            return "Reachy is not connected"

        s = "\n\t".join([part_name + ": " + str(part) for part_name, part in self.info._enabled_parts.items()])
        repr_template = (
            '<Reachy host="{host}" connected={connected} on={on} \n'
            " battery_voltage={battery_voltage} \n"
            " parts=\n\t{parts} \n>"
        )
        return repr_template.format(
            host=self._host,
            connected=self._grpc_connected,
            on=self.is_on(),
            battery_voltage=self.info.battery_voltage,
            parts=s,
        )

    @property
    def info(self) -> Optional[ReachyInfo]:
        if not self._grpc_connected:
            self._logger.error("Cannot get info, not connected to Reachy")
            return None
        return self._info

    @property
    def head(self) -> Optional[Head]:
        """Get Reachy's head."""
        if not self._grpc_connected:
            self._logger.error("Cannot get head, not connected to Reachy")
            return None
        if self._head is None:
            self._logger.error("head does not exist with this configuration")
            return None
        return self._head

    @property
    def r_arm(self) -> Optional[Arm]:
        """Get Reachy's right arm."""
        if not self._grpc_connected:
            self._logger.error("Cannot get r_arm, not connected to Reachy")
            return None
        if self._r_arm is None:
            self._logger.error("r_arm does not exist with this configuration")
            return None
        return self._r_arm

    @property
    def l_arm(self) -> Optional[Arm]:
        """Get Reachy's left arm."""
        if not self._grpc_connected:
            self._logger.error("Cannot get l_arm, not connected to Reachy")
            return None
        if self._l_arm is None:
            self._logger.error("l_arm does not exist with this configuration")
            return None
        return self._l_arm

    @property
    def mobile_base(self) -> Optional[MobileBase]:
        """Get Reachy's mobile base."""
        if not self._grpc_connected:
            self._logger.error("Cannot get mobile_base, not connected to Reachy")
            return None
        if self._mobile_base is None:
            self._logger.error("mobile_base does not exist with this configuration")
            return None
        return self._mobile_base

    @property
    def joints(self) -> CustomDict[str, OrbitaJoint]:
        """Get all joints of the robot."""
        if not self._grpc_connected or not self.info:
            self._logger.warning("Cannot get joints, not connected to Reachy.")
            return CustomDict({})
        _joints: CustomDict[str, OrbitaJoint] = CustomDict({})
        for part_name in self.info._enabled_parts:
            part = getattr(self, part_name)
            for joint_name, joint in part.joints.items():
                _joints[part_name + "." + joint_name] = joint
        return _joints

    @property
    def _actuators(self) -> Dict[str, Orbita2d | Orbita3d]:
        """Get all actuators of the robot."""
        if not self._grpc_connected or not self.info:
            self._logger.warning("Cannot get actuators, not connected to Reachy.")
            return {}
        _actuators: Dict[str, Orbita2d | Orbita3d] = {}
        for part_name in self.info._enabled_parts:
            part = getattr(self, part_name)
            for actuator_name, actuator in part._actuators.items():
                _actuators[part_name + "." + actuator_name] = actuator
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

    @property
    def cameras(self) -> Optional[CameraManager]:
        """Get Reachy's cameras."""
        if not self._grpc_connected:
            self._logger.error("Cannot get cameras, not connected to Reachy")
            return None
        return self._cameras

    def _get_info(self) -> None:
        """Get main description of the robot.

        First connection to the robot. Information get:
        - robot's parts
        - robot's sofware and hardware version
        - robot's serial number
        """
        self._stub = reachy_pb2_grpc.ReachyServiceStub(self._grpc_channel)
        try:
            self._robot = self._stub.GetReachy(Empty())
        except _InactiveRpcError:
            raise ConnectionError()

        self._info = ReachyInfo(self._robot)
        self._grpc_connected = True

    def _setup_audio(self) -> None:
        """Internal function to set up the audio server."""
        try:
            self.audio = Audio(self._host, self._audio_port)
        except Exception:
            self._logger.error("Failed to connect to audio server. ReachySDK.audio will not be available.")

    def _setup_video(self) -> Optional[CameraManager]:
        try:
            return CameraManager(self._host, self._video_port)

        except Exception as e:
            self._logger.error(f"Failed to connect to video server with error: {e}.\nReachySDK.video will not be available.")
            return None

    def _setup_part_r_arm(self, initial_state: ReachyState) -> None:
        if not self.info:
            self._logger.warning("Reachy is not connected")
            return None

        if self._robot.HasField("r_arm"):
            if initial_state.r_arm_state.activated:
                r_arm = Arm(self._robot.r_arm, initial_state.r_arm_state, self._grpc_channel, self._goto_stub)
                self._r_arm = r_arm
                self.info._enabled_parts["r_arm"] = self._r_arm
                if self._robot.HasField("r_hand"):
                    self._r_arm._init_hand(self._robot.r_hand, initial_state.r_hand_state)
            else:
                self.info._disabled_parts.append("r_arm")

    def _setup_part_l_arm(self, initial_state: ReachyState) -> None:
        if not self.info:
            self._logger.warning("Reachy is not connected")
            return None

        if self._robot.HasField("l_arm"):
            if initial_state.l_arm_state.activated:
                l_arm = Arm(self._robot.l_arm, initial_state.l_arm_state, self._grpc_channel, self._goto_stub)
                self._l_arm = l_arm
                self.info._enabled_parts["l_arm"] = self._l_arm
                if self._robot.HasField("l_hand"):
                    self._l_arm._init_hand(self._robot.l_hand, initial_state.l_hand_state)
            else:
                self.info._disabled_parts.append("l_arm")

    def _setup_part_mobile_base(self, initial_state: ReachyState) -> None:
        if not self.info:
            self._logger.warning("Reachy is not connected")
            return None

        if self._robot.HasField("mobile_base"):
            self._mobile_base = MobileBase(self._robot.head, initial_state.mobile_base_state, self._grpc_channel)
            self.info._set_mobile_base(self._mobile_base)

    def _setup_part_head(self, initial_state: ReachyState) -> None:
        if not self.info:
            self._logger.warning("Reachy is not connected")
            return None

        if self._robot.HasField("head"):
            if initial_state.head_state.activated:
                head = Head(self._robot.head, initial_state.head_state, self._grpc_channel, self._goto_stub)
                self._head = head
                self.info._enabled_parts["head"] = self._head
            else:
                self.info._disabled_parts.append("head")

    def _setup_parts(self) -> None:
        """Setup all parts of the robot.

        Get the state of each part of the robot, create an instance for each of them and add
        it to the ReachySDK instance.
        """
        setup_stub = reachy_pb2_grpc.ReachyServiceStub(self._grpc_channel)
        self._goto_stub = GoToServiceStub(self._grpc_channel)
        initial_state = setup_stub.GetReachyState(self._robot.id)

        self._setup_part_r_arm(initial_state)
        self._setup_part_l_arm(initial_state)
        self._setup_part_head(initial_state)
        self._setup_part_mobile_base(initial_state)

    def _start_sync_in_bg(self) -> None:
        """Start the synchronization asyncio tasks with the robot in background."""
        channel = grpc.insecure_channel(f"{self._host}:{self._sdk_port}")
        reachy_stub = reachy_pb2_grpc.ReachyServiceStub(channel)
        self._get_stream_update_loop(reachy_stub, freq=100)

    def _get_stream_update_loop(self, reachy_stub: reachy_pb2_grpc.ReachyServiceStub, freq: float) -> None:
        """Update the state of the robot at a given frequency."""
        stream_req = reachy_pb2.ReachyStreamStateRequest(id=self._robot.id, publish_frequency=freq)
        try:
            for state_update in reachy_stub.StreamReachyState(stream_req):
                if self._l_arm is not None:
                    self._l_arm._update_with(state_update.l_arm_state)
                    if self._l_arm.gripper is not None:
                        self._l_arm.gripper._update_with(state_update.l_hand_state)
                if self._r_arm is not None:
                    self._r_arm._update_with(state_update.r_arm_state)
                    if self._r_arm.gripper is not None:
                        self._r_arm.gripper._update_with(state_update.r_hand_state)
                if self._head is not None:
                    self._head._update_with(state_update.head_state)
                if self._mobile_base is not None:
                    self._mobile_base._update_with(state_update.mobile_base_state)
        except grpc._channel._MultiThreadedRendezvous:
            self._grpc_connected = False
            raise ConnectionError(f"Connection with Reachy ip:{self._host} lost, check the sdk server status.")

    def _audit(self) -> None:
        while self._grpc_connected:
            audit_status = self._stub.Audit(self._robot.id)
            if self._l_arm is not None and audit_status.HasField("l_arm_status"):
                self._l_arm._update_audit_status(audit_status.l_arm_status)
                if self._l_arm.gripper is not None and audit_status.HasField("l_hand_status"):
                    self._l_arm.gripper._update_audit_status(audit_status.l_hand_status)
            if self._r_arm is not None and audit_status.HasField("r_arm_status"):
                self._r_arm._update_audit_status(audit_status.r_arm_status)
                if self._r_arm.gripper is not None and audit_status.HasField("r_hand_status"):
                    self._r_arm.gripper._update_audit_status(audit_status.r_hand_status)
            if self._head is not None and audit_status.HasField("head_status"):
                self._head._update_audit_status(audit_status.head_status)
            if self._mobile_base is not None and audit_status.HasField("mobile_base_status"):
                self._mobile_base._update_audit_status(audit_status.mobile_base_status)
            time.sleep(1)

    @property
    def audit(self) -> Dict[str, Dict[str, str]]:
        audit_dict: Dict[str, Dict[str, str]] = {}
        if not self._grpc_connected or not self.info:
            self._logger.warning("Reachy is not connected!")
        if self.info is not None:
            for part in self.info._enabled_parts.values():
                audit_dict[part._part_id.name] = part.audit
        return audit_dict

    def turn_on(self) -> bool:
        """Turn all motors of enabled parts on.

        All enabled parts' motors will then be stiff.
        """
        if not self._grpc_connected or not self.info:
            self._logger.warning("Cannot turn on Reachy, not connected.")
            return False
        for part in self.info._enabled_parts.values():
            part._turn_on()
        if self._mobile_base is not None:
            self._mobile_base._turn_on()
        time.sleep(0.5)

        return True

    def turn_off(self) -> bool:
        """Turn all motors of enabled parts off.

        All enabled parts' motors will then be compliant.
        """
        if not self._grpc_connected or not self.info:
            self._logger.warning("Cannot turn off Reachy, not connected.")
            return False
        for part in self.info._enabled_parts.values():
            part._turn_off()
        if self._mobile_base is not None:
            self._mobile_base._turn_off()
        time.sleep(0.5)

        return True

    def turn_off_smoothly(self, duration: float = 2) -> bool:
        """Turn all motors of enabled parts off.

        All enabled parts' motors will then be compliant.
        """
        if not self._grpc_connected or not self.info:
            self._logger.warning("Cannot turn off Reachy, not connected.")
            return False
        if hasattr(self, "_mobile_base") and self._mobile_base is not None:
            self._mobile_base._turn_off()
        for part in self.info._enabled_parts.values():
            if "arm" in part._part_id.name:
                part.set_torque_limits(20)
            else:
                part._turn_off()
        time.sleep(duration)
        for part in self.info._enabled_parts.values():
            if "arm" in part._part_id.name:
                part._turn_off()
                part.set_torque_limits(100)
        time.sleep(0.5)
        return True

    def is_on(self) -> bool:
        """Return True if all actuators of the arm are stiff"""
        if not self.info:
            self._logger.warning("Reachy is not connected!")
            return False

        for part in self.info._enabled_parts.values():
            if not part.is_on():
                return False
        if self._mobile_base is not None and self._mobile_base.is_off():
            return False
        return True

    def is_off(self) -> bool:
        """Return True if all actuators of the arm are stiff"""

        if not self.info:
            self._logger.warning("Reachy is not connected!")
            return True

        for part in self.info._enabled_parts.values():
            if part.is_on():
                return False
        if self._mobile_base is not None and self._mobile_base.is_on():
            return False
        return True

    def send_goal_positions(self) -> None:
        if not self.info:
            self._logger.warning("Reachy is not connected!")
            return

        for part in self.info._enabled_parts.values():
            if issubclass(type(part), JointsBasedPart):
                part.send_goal_positions()

    def set_pose(
        self,
        common_pose: str = "default",
        wait: bool = False,
        wait_for_moves_end: bool = True,
        duration: float = 2,
        interpolation_mode: str = "minimum_jerk",
    ) -> GoToHomeId:
        """Send all joints to standard positions in specified duration.

        common_pose can be 'default', arms being straight, or 'elbow_90'.
        Setting wait_for_goto_end to False will cancel all gotos on all parts and immediately send the commands.
        Otherwise, the commands will be sent to a part when all gotos of its queue has been played.
        """
        if common_pose not in ["default", "elbow_90"]:
            raise ValueError(f"common_pose {interpolation_mode} not supported! Should be 'default' or 'elbow_90'")
        head_id = None
        r_arm_id = None
        l_arm_id = None
        if not wait_for_moves_end:
            self.cancel_all_moves()
        if self.head is not None:
            head_id = self.head.set_pose(wait_for_moves_end, duration, interpolation_mode)
        if self.r_arm is not None:
            r_arm_id = self.r_arm.set_pose(common_pose, wait_for_moves_end, duration, interpolation_mode)
        if self.l_arm is not None:
            l_arm_id = self.l_arm.set_pose(common_pose, wait_for_moves_end, duration, interpolation_mode)
        ids = GoToHomeId(
            head=head_id,
            r_arm=r_arm_id,
            l_arm=l_arm_id,
        )
        return ids

    def is_move_finished(self, id: GoToId) -> bool:
        """Return True if goto has been played and has been cancelled, False otherwise."""
        if not self._grpc_connected:
            self._logger.warning("Reachy is not connected!")
            return False
        state = self._get_move_state(id)
        result = bool(
            state.goal_status == GoalStatus.STATUS_ABORTED
            or state.goal_status == GoalStatus.STATUS_CANCELED
            or state.goal_status == GoalStatus.STATUS_SUCCEEDED
        )
        return result

    def is_move_playing(self, id: GoToId) -> bool:
        """Return True if goto is currently playing, False otherwise."""
        if not self._grpc_connected:
            self._logger.warning("Reachy is not connected!")
            return False
        state = self._get_move_state(id)
        return bool(state.goal_status == GoalStatus.STATUS_EXECUTING)

    def cancel_all_moves(self) -> GoToAck:
        """Cancel all the goto tasks."""
        if not self._grpc_connected:
            self._logger.warning("Reachy is not connected!")
            return None
        response = self._goto_stub.CancelAllGoTo(Empty())
        return response

    def _get_move_state(self, goto_id: GoToId) -> GoToGoalStatus:
        """Return the current state of a goto, given its id."""
        response = self._goto_stub.GetGoToState(goto_id)
        return response

    def cancel_move_by_id(self, goto_id: GoToId) -> GoToAck:
        """Ask the cancellation of a single goto on the arm, given its id"""
        if not self._grpc_connected:
            self._logger.warning("Reachy is not connected!")
            return None
        response = self._goto_stub.CancelGoTo(goto_id)
        return response

    def get_move_joints_request(self, goto_id: GoToId) -> Optional[SimplifiedRequest]:
        """Returns the part affected, the joints goal positions, duration and mode of the corresponding GoToId

        Part can be either 'r_arm', 'l_arm' or 'head'
        Goal_position is returned as a list in degrees
        """
        if not self._grpc_connected:
            self._logger.warning("Reachy is not connected!")
            return None
        if goto_id.id == -1:
            raise ValueError("No answer was found for given move, goto_id is -1")

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
