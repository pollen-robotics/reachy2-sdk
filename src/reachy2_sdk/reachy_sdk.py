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
from google.protobuf.timestamp_pb2 import Timestamp
from grpc._channel import _InactiveRpcError
from reachy2_sdk_api import reachy_pb2, reachy_pb2_grpc
from reachy2_sdk_api.goto_pb2 import GoalStatus, GoToAck, GoToGoalStatus, GoToId
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.reachy_pb2 import ReachyState

from .config.reachy_info import ReachyInfo
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
    """The ReachySDK class manages the connection and interaction with a Reachy robot.

    This class handles:
    - Establishing and maintaining a connection with the robot via gRPC.
    - Accessing and controlling various parts of the robot, such as the arms, head, and mobile base.
    - Managing robot components including actuators, joints, cameras, and audio.
    - Synchronizing robot state with the server in the background to keep data up-to-date.
    - Providing utility functions for common tasks such as turning on/off motors, sending goal positions,
        and performing movements.
    """

    def __init__(
        self,
        host: str,
        sdk_port: int = 50051,
        audio_port: int = 50063,
        video_port: int = 50065,
    ) -> None:
        """Initialize a connection to the robot.

        Args:
            host: The IP address or hostname of the robot.
            sdk_port: The gRPC port for the SDK. Default is 50051.
            audio_port: The gRPC port for audio services. Default is 50063.
            video_port: The gRPC port for video services. Default is 50065.
        """
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

        self._update_timestamp: Timestamp = Timestamp(seconds=0)

        self.connect()

    def connect(self) -> None:
        """Connects the SDK to the robot."""
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
        """Disconnect the SDK from the robot's server.

        Args:
            lost_connection: If `True`, indicates that the connection was lost unexpectedly.
        """
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
        """Get ReachyInfo if connected."""
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
        """Return a dictionary of all joints of the robot.

        The dictionary keys are the joint names, and the values are the corresponding OrbitaJoint objects.
        """
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
        """Return a dictionary of all actuators of the robot.

        The dictionary keys are the actuator names, and the values are the corresponding actuator objects.
        """
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
        """Check if the SDK is connected to the robot.

        Returns:
            `True` if connected, `False` otherwise.
        """
        return self._grpc_connected

    @property
    def _grpc_status(self) -> str:
        """Get the current connection status of the robot's gRPC server.

        Returns:
            "connected" if connected, "disconnected" otherwise.
        """
        if self._grpc_connected:
            return "connected"
        else:
            return "disconnected"

    @_grpc_status.setter
    def _grpc_status(self, status: str) -> None:
        """Set the connection status of the robot's gRPC server.

        Args:
            status: The connection status to set, must be either "connected" or "disconnected".

        Raises:
            ValueError: If the status is not "connected" or "disconnected".
        """
        if status == "connected":
            self._grpc_connected = True
        elif status == "disconnected":
            self._grpc_connected = False
        else:
            raise ValueError("_grpc_status can only be set to 'connected' or 'disconnected'")

    @property
    def cameras(self) -> Optional[CameraManager]:
        """Get the camera manager if available and connected."""
        if not self._grpc_connected:
            self._logger.error("Cannot get cameras, not connected to Reachy")
            return None
        return self._cameras

    def _get_info(self) -> None:
        """Retrieve basic information about the robot.

        Gathers data on the robot's parts, hardware and software versions, and serial number.
        """
        self._stub = reachy_pb2_grpc.ReachyServiceStub(self._grpc_channel)
        try:
            self._robot = self._stub.GetReachy(Empty())
        except _InactiveRpcError:
            raise ConnectionError()

        self._info = ReachyInfo(self._robot)
        self._grpc_connected = True

    def _setup_audio(self) -> None:
        """Set up the audio server for the robot.

        Attempts to connect to the audio server and initializes the audio-related components.
        """
        # try:
        #     self.audio = Audio(self._host, self._audio_port)
        # except Exception:
        #     self._logger.error("Failed to connect to audio server. ReachySDK.audio will not be available.")
        pass

    def _setup_video(self) -> Optional[CameraManager]:
        """Set up the video server for the robot.

        Returns:
            A CameraManager instance if the video server connection is successful, otherwise None.
        """
        try:
            return CameraManager(self._host, self._video_port)

        except Exception as e:
            self._logger.error(f"Failed to connect to video server with error: {e}.\nReachySDK.video will not be available.")
            return None

    def _setup_part_r_arm(self, initial_state: ReachyState) -> None:
        """Set up the robot's right arm based on the initial state."""
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
        """Set up the robot's left arm based on the initial state."""
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
        """Set up the robot's mobile base based on the initial state."""
        if not self.info:
            self._logger.warning("Reachy is not connected")
            return None

        if self._robot.HasField("mobile_base"):
            self._mobile_base = MobileBase(self._robot.head, initial_state.mobile_base_state, self._grpc_channel)
            self.info._set_mobile_base(self._mobile_base)

    def _setup_part_head(self, initial_state: ReachyState) -> None:
        """Set up the robot's head based on the initial state."""
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
        """Initialize all parts of the robot.

        Retrieves the state of each part, creates instances, and adds them to the ReachySDK instance.
        """
        setup_stub = reachy_pb2_grpc.ReachyServiceStub(self._grpc_channel)
        self._goto_stub = GoToServiceStub(self._grpc_channel)
        initial_state = setup_stub.GetReachyState(self._robot.id)

        self._setup_part_r_arm(initial_state)
        self._setup_part_l_arm(initial_state)
        self._setup_part_head(initial_state)
        self._setup_part_mobile_base(initial_state)

    def get_update_timestamp(self) -> int:
        return self._update_timestamp.ToNanoseconds()

    def _start_sync_in_bg(self) -> None:
        """Start background synchronization with the robot."""
        channel = grpc.insecure_channel(f"{self._host}:{self._sdk_port}")
        reachy_stub = reachy_pb2_grpc.ReachyServiceStub(channel)
        self._get_stream_update_loop(reachy_stub, freq=100)

    def _get_stream_update_loop(self, reachy_stub: reachy_pb2_grpc.ReachyServiceStub, freq: float) -> None:
        """Update the robot's state at a specified frequency.

        Args:
            reachy_stub: The gRPC stub for communication with the robot.
            freq: The frequency (in Hz) at which to update the robot's state.
        """
        stream_req = reachy_pb2.ReachyStreamStateRequest(id=self._robot.id, publish_frequency=freq)
        try:
            for state_update in reachy_stub.StreamReachyState(stream_req):
                self._update_timestamp = state_update.timestamp
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
        """Periodically perform an audit of the robot's components."""
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
        """Return the audit status of all enabled parts of the robot."""
        audit_dict: Dict[str, Dict[str, str]] = {}
        if not self._grpc_connected or not self.info:
            self._logger.warning("Reachy is not connected!")
        if self.info is not None:
            for part in self.info._enabled_parts.values():
                audit_dict[part._part_id.name] = part.audit
        return audit_dict

    def turn_on(self) -> bool:
        """Activate all motors of the robot's parts.

        Returns:
            `True` if successful, `False` otherwise.
        """
        if not self._grpc_connected or not self.info:
            self._logger.warning("Cannot turn on Reachy, not connected.")
            return False
        for part in self.info._enabled_parts.values():
            part.set_speed_limits(1)
        time.sleep(0.05)
        for part in self.info._enabled_parts.values():
            part._turn_on()
        if self._mobile_base is not None:
            self._mobile_base._turn_on()
        time.sleep(0.05)
        for part in self.info._enabled_parts.values():
            part.set_speed_limits(100)
        time.sleep(0.4)

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

    def turn_off_smoothly(self) -> bool:
        """Turn all motors of robot parts off.

        Arm torques are reduced during 3 seconds, then all parts' motors will be compliant.
        """
        if not self._grpc_connected or not self.info:
            self._logger.warning("Cannot turn off Reachy, not connected.")
            return False

        torque_limit_low = 35
        torque_limit_high = 100
        duration = 3
        arms_list = []

        tic = time.time()

        if hasattr(self, "_mobile_base") and self._mobile_base is not None:
            self._mobile_base._turn_off()
        for part in self.info._enabled_parts.values():
            if "arm" in part._part_id.name:
                part.set_torque_limits(torque_limit_low)
                part.goto_posture(duration=duration, wait_for_goto_end=False)
                arms_list.append(part)
            else:
                part._turn_off()
        elapsed_time = time.time() - tic
        print(f"1 : {elapsed_time}")

        countingTime = 0
        while countingTime < duration:
            time.sleep(1)
            torque_limit_low -= 10
            for arm_part in arms_list:
                arm_part.set_torque_limits(torque_limit_low)
            countingTime += 1

        elapsed_time = time.time() - tic
        print(f"2 : {elapsed_time}")

        for arm_part in arms_list:
            arm_part._turn_off()
            arm_part.set_torque_limits(torque_limit_high)

        elapsed_time = time.time() - tic
        print(f"3 : {elapsed_time}")

        time.sleep(0.5)
        return True

    def is_on(self) -> bool:
        """Check if all actuators of Reachy parts are on (stiff).

        Returns:
            `True` if all are stiff, `False` otherwise.
        """
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
        """Check if all actuators of Reachy parts are off (compliant).

        Returns:
            `True` if all are compliant, `False` otherwise.
        """
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
        """Send the goal positions to the robot.

        If goal positions have been specified for any joint of the robot, sends them to the robot.
        """
        if not self.info:
            self._logger.warning("Reachy is not connected!")
            return

        for part in self.info._enabled_parts.values():
            if issubclass(type(part), JointsBasedPart):
                part.send_goal_positions()

    def goto_posture(
        self,
        common_posture: str = "default",
        wait: bool = False,
        wait_for_goto_end: bool = True,
        duration: float = 2,
        interpolation_mode: str = "minimum_jerk",
    ) -> GoToHomeId:
        """Move the robot to a predefined posture.

        Args:
            common_posture: The name of the posture. It can be 'default' or 'elbow_90'. Defaults to 'default'.
            wait: Determines whether the program should wait for the movement to finish before
                returning. If set to `True`, the program waits for the movement to complete before continuing
                execution. Defaults to `False`.
            wait_for_goto_end: Specifies whether commands will be sent to a part immediately or
                only after all previous commands in the queue have been executed. If set to `False`, the program
                will cancel all executing moves and queues. Defaults to `True`.
            duration: The time duration in seconds for the robot to move to the specified posture.
                Defaults to 2.
            interpolation_mode: The type of interpolation used when moving the arm's joints.
                Can be 'minimum_jerk' or 'linear'. Defaults to 'minimum_jerk'.

        Returns:
            A GoToHomeId containing movement GoToIds for each part.
        """
        if common_posture not in ["default", "elbow_90"]:
            raise ValueError(f"common_posture {common_posture} not supported! Should be 'default' or 'elbow_90'")
        head_id = None
        r_arm_id = None
        l_arm_id = None
        if not wait_for_goto_end:
            self.cancel_all_goto()
        if self.head is not None:
            is_last_commmand = self.r_arm is None and self.l_arm is None
            wait_head = wait and is_last_commmand
            head_id = self.head.goto_posture(
                duration=duration, wait=wait_head, wait_for_goto_end=wait_for_goto_end, interpolation_mode=interpolation_mode
            )
        if self.r_arm is not None:
            is_last_commmand = self.l_arm is None
            wait_r_arm = wait and is_last_commmand
            r_arm_id = self.r_arm.goto_posture(
                common_posture,
                duration=duration,
                wait=wait_r_arm,
                wait_for_goto_end=wait_for_goto_end,
                interpolation_mode=interpolation_mode,
            )
        if self.l_arm is not None:
            l_arm_id = self.l_arm.goto_posture(
                common_posture,
                duration=duration,
                wait=wait,
                wait_for_goto_end=wait_for_goto_end,
                interpolation_mode=interpolation_mode,
            )
        ids = GoToHomeId(
            head=head_id,
            r_arm=r_arm_id,
            l_arm=l_arm_id,
        )
        return ids

    def reset_default_limits(self) -> None:
        """Reset speed and torque limits of all parts to the maximum value (100)."""
        if not self.info:
            self._logger.warning("Reachy is not connected!")
            return

        for part in self.info._enabled_parts.values():
            if issubclass(type(part), JointsBasedPart):
                part.set_speed_limits(100)
                part.set_torque_limits(100)
        time.sleep(0.5)

    def is_goto_finished(self, goto_id: GoToId) -> bool:
        """Check if a goto command has completed.

        Args:
            goto_id: The unique GoToId of the goto command.

        Returns:
            `True` if the command is complete, `False` otherwise.
        """
        if not self._grpc_connected:
            self._logger.warning("Reachy is not connected!")
            return False
        if not isinstance(goto_id, GoToId):
            raise TypeError(f"goto_id must be a GoToId, got {type(goto_id).__name__}")
        if goto_id.id == -1:
            self._logger.error("is_goto_finished() asked for unvalid movement. Goto not played.")
            return True
        state = self._get_goto_state(goto_id)
        result = bool(
            state.goal_status == GoalStatus.STATUS_ABORTED
            or state.goal_status == GoalStatus.STATUS_CANCELED
            or state.goal_status == GoalStatus.STATUS_SUCCEEDED
        )
        return result

    def cancel_all_goto(self) -> GoToAck:
        """Cancel all active goto commands.

        Returns:
             A `GoToAck` object indicating whether the cancellation was acknowledged.
        """
        if not self._grpc_connected:
            self._logger.warning("Reachy is not connected!")
            return None
        response = self._goto_stub.CancelAllGoTo(Empty())
        return response

    def _get_goto_state(self, goto_id: GoToId) -> GoToGoalStatus:
        """Retrieve the current state of a goto command.

        Args:
            goto_id: The unique GoToId of the goto command.

        Returns:
            The current state of the command.
        """
        response = self._goto_stub.GetGoToState(goto_id)
        return response

    def cancel_goto_by_id(self, goto_id: GoToId) -> GoToAck:
        """Request the cancellation of a specific goto command based on its GoToId.

        Args:
            goto_id: The ID of the goto command to cancel.

        Returns:
            A `GoToAck` object indicating whether the cancellation was acknowledged.
            If the robot is not connected, returns None.

        Raises:
            TypeError: If `goto_id` is not an instance of `GoToId`.
        """
        if not self._grpc_connected:
            self._logger.warning("Reachy is not connected!")
            return None
        if not isinstance(goto_id, GoToId):
            raise TypeError(f"goto_id must be a GoToId, got {type(goto_id).__name__}")
        if goto_id.id == -1:
            self._logger.error("cancel_goto_by_id() asked for unvalid movement. Goto not played.")
            return GoToAck(ack=True)
        response = self._goto_stub.CancelGoTo(goto_id)
        return response

    def get_goto_joints_request(self, goto_id: GoToId) -> Optional[SimplifiedRequest]:
        """Retrieve the details of a goto command based on its GoToId.

        Args:
            goto_id: The ID of the goto command for which details are requested.

        Returns:
            A `SimplifiedRequest` object containing the part name, joint goal positions
            (in degrees), movement duration, and interpolation mode.
            Returns `None` if the robot is not connected or if the `goto_id` is invalid.

        Raises:
            TypeError: If `goto_id` is not an instance of `GoToId`.
            ValueError: If `goto_id` is -1, indicating an invalid command.
        """
        if not self._grpc_connected:
            self._logger.warning("Reachy is not connected!")
            return None
        if not isinstance(goto_id, GoToId):
            raise TypeError(f"goto_id must be a GoToId, got {type(goto_id).__name__}")
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
