"""Reachy IGoToBasedElement interface.

Handles common interface for elements (parts or components) performing movements using goto mechanism.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from reachy2_sdk_api.component_pb2 import ComponentId
from reachy2_sdk_api.goto_pb2 import GoalStatus, GoToAck, GoToId, GoToRequest
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.part_pb2 import PartId

from ..utils.utils import (
    EllipticalParameters,
    JointsRequest,
    OdometryRequest,
    SimplifiedRequest,
    TargetJointsRequest,
    arm_position_to_list,
    ext_euler_angles_to_list,
    get_arc_direction,
    get_interpolation_mode,
    get_interpolation_space,
)


class IGoToBasedElement(ABC):
    """Interface for elements (parts or components) of Reachy that use goto functions.

    The `IGoToBasedElements` class defines a common interface for handling goto-based movements. It is
    designed to be implemented by any parts or components of the robot that perform movements via the goto mechanism.
    """

    def __init__(
        self,
        element_id: ComponentId | PartId,
        goto_stub: GoToServiceStub,
    ) -> None:
        """Initialize the IGoToBasedElement interface.

        Sets up the common attributes needed for handling goto-based movements. This includes
        associating the component with the interface and setting up the gRPC stub for performing
        goto commands.

        Args:
            element_id: The robot component or part that uses this interface.
            goto_stub: The gRPC stub used to send goto commands to the robot element.
        """
        self._element_id = element_id
        self._goto_stub = goto_stub
        self._logger_goto = logging.getLogger(__name__)  # not using self._logger to avoid name conflict in multiple inheritance

    @abstractmethod
    def get_goto_playing(self) -> GoToId:
        """Return the GoToId of the currently playing goto movement on a specific element."""
        pass

    @abstractmethod
    def get_goto_queue(self) -> List[GoToId]:
        """Return a list of all GoToIds waiting to be played on a specific element."""
        pass

    @abstractmethod
    def cancel_all_goto(self) -> GoToAck:
        """Request the cancellation of all playing and waiting goto commands for a specific element.

        Returns:
            A GoToAck acknowledging the cancellation of all goto commands.
        """
        pass

    def _get_goto_request(self, goto_id: GoToId) -> Optional[SimplifiedRequest]:
        """Retrieve the details of a goto command based on its GoToId.

        Args:
            goto_id: The ID of the goto command for which details are requested.

        Returns:
            A `SimplifiedRequest` object containing the element name, joint goal positions
            (in degrees), movement duration, and interpolation mode.
            Returns `None` if the robot is not connected or if the `goto_id` is invalid.

        Raises:
            TypeError: If `goto_id` is not an instance of `GoToId`.
            ValueError: If `goto_id` is -1, indicating an invalid command.
        """
        if not isinstance(goto_id, GoToId):
            raise TypeError(f"goto_id must be a GoToId, got {type(goto_id).__name__}")
        if goto_id.id == -1:
            raise ValueError("No answer was found for given move, goto_id is -1")

        response = self._goto_stub.GetGoToRequest(goto_id)

        full_request = process_goto_request(response)

        return full_request

    def _is_goto_finished(self, id: GoToId) -> bool:
        """Check if the goto movement has been completed or cancelled.

        Returns:
           `True` if the goto has been played or cancelled, `False` otherwise.
        """
        state = self._goto_stub.GetGoToState(id)
        result = bool(
            state.goal_status == GoalStatus.STATUS_ABORTED
            or state.goal_status == GoalStatus.STATUS_CANCELED
            or state.goal_status == GoalStatus.STATUS_SUCCEEDED
        )
        return result

    def _wait_goto(self, id: GoToId, duration: float) -> None:
        """Wait for a goto to finish. timeout is in seconds."""
        self._logger_goto.info(f"Waiting for movement with {id}.")

        if not self._is_goto_already_over(id, duration):
            info_gotos = [self._get_goto_request(id)]
            ids_queue = self.get_goto_queue()
            for goto_id in ids_queue:
                info_gotos.append(self._get_goto_request(goto_id))

            timeout = 1  # adding one more sec
            for igoto in info_gotos:
                if igoto is not None:
                    if type(igoto.request) is JointsRequest:
                        timeout += igoto.request.duration
                    elif type(igoto.request) is OdometryRequest:
                        timeout += igoto.request.timeout

            self._logger_goto.debug(f"timeout is set to {timeout}")

            t_start = time.time()  # timeout for others
            while not self._is_goto_finished(id):
                time.sleep(0.1)

                if time.time() - t_start > timeout:
                    self._logger_goto.warning(f"Waiting time for movement with {id} is timeout.")
                    return

            self._logger_goto.info(f"Movement with {id} finished.")

    def _is_goto_already_over(self, id: GoToId, timeout: float) -> bool:
        """Check if the goto movement is already over."""
        t0 = time.time()
        id_playing = self.get_goto_playing()
        while id_playing.id == -1:
            time.sleep(0.005)
            id_playing = self.get_goto_playing()

            if self._is_goto_finished(id):
                return True

            # manage an id_playing staying at -1
            if time.time() - t0 > timeout:
                self._logger_goto.warning(f"Waiting time for movement with {id} is timeout.")
                return True
        return False

    @abstractmethod
    def _check_goto_parameters(self, target: Any, duration: Optional[float], q0: Optional[List[float]] = None) -> None:
        """Check the validity of the parameters for a goto movement."""
        pass  # pragma: no cover

    @abstractmethod
    def goto_posture(
        self,
        common_posture: str = "default",
        duration: float = 2,
        wait: bool = False,
        wait_for_goto_end: bool = True,
        interpolation_mode: str = "minimum_jerk",
    ) -> GoToId:
        """Send all joints to standard positions with optional parameters for duration, waiting, and interpolation mode."""
        pass  # pragma: no cover


def process_goto_request(response: GoToRequest) -> Optional[SimplifiedRequest]:
    """Process the response from a goto request and return a SimplifiedRequest object."""
    if response.HasField("cartesian_goal"):
        request_dict = process_arm_cartesian_goal(response)
        return build_simplified_joints_request(request_dict)

    elif response.HasField("joints_goal"):
        match response.joints_goal.WhichOneof("joints_goal"):
            case "arm_joint_goal":
                request_dict = process_arm_joint_goal(response)
            case "neck_joint_goal":
                request_dict = process_neck_joint_goal(response)
            case "antenna_joint_goal":
                request_dict = process_antenna_joint_goal(response)
            case "hand_joint_goal":
                request_dict = process_hand_joint_goal(response)
            case _:
                raise ValueError("No valid joint goal found in the response")
        return build_simplified_joints_request(request_dict)

    elif response.HasField("odometry_goal"):
        request_dict = process_odometry_goal(response)
        return build_simplified_odometry_request(request_dict)

    else:
        raise ValueError("No valid request found in the response")


def process_arm_cartesian_goal(response: GoToRequest) -> Dict[str, Any]:
    """Process the response from a goto request containing an arm cartesian goal."""
    request_dict = {}
    request_dict["part"] = response.cartesian_goal.arm_cartesian_goal.id.name
    request_dict["mode"] = get_interpolation_mode(response.interpolation_mode.interpolation_type)
    request_dict["interpolation_space"] = get_interpolation_space(response.interpolation_space.interpolation_space)
    request_dict["duration"] = response.cartesian_goal.arm_cartesian_goal.duration.value
    request_dict["target_pose"] = np.reshape(response.cartesian_goal.arm_cartesian_goal.goal_pose.data, (4, 4))
    if request_dict["mode"] == "elliptical":
        arc_direction = get_arc_direction(response.elliptical_parameters.arc_direction)
        secondary_radius = response.elliptical_parameters.secondary_radius.value
        request_dict["elliptical_params"] = EllipticalParameters(arc_direction, secondary_radius)
    else:
        request_dict["elliptical_params"] = None
    return request_dict


def process_arm_joint_goal(response: GoToRequest) -> Dict[str, Any]:
    """Process the response from a goto request containing an arm joint goal."""
    request_dict = {}
    request_dict["part"] = response.joints_goal.arm_joint_goal.id.name
    request_dict["mode"] = get_interpolation_mode(response.interpolation_mode.interpolation_type)
    request_dict["interpolation_space"] = get_interpolation_space(response.interpolation_space.interpolation_space)
    request_dict["duration"] = response.joints_goal.arm_joint_goal.duration.value
    request_dict["target_joints"] = arm_position_to_list(response.joints_goal.arm_joint_goal.joints_goal, degrees=True)
    request_dict["elliptical_params"] = None
    return request_dict


def process_neck_joint_goal(response: GoToRequest) -> Dict[str, Any]:
    """Process the response from a goto request containing a neck joint goal."""
    request_dict = {}
    request_dict["part"] = response.joints_goal.neck_joint_goal.id.name
    request_dict["mode"] = get_interpolation_mode(response.interpolation_mode.interpolation_type)
    request_dict["interpolation_space"] = get_interpolation_space(response.interpolation_space.interpolation_space)
    request_dict["duration"] = response.joints_goal.neck_joint_goal.duration.value
    request_dict["target_joints"] = ext_euler_angles_to_list(
        response.joints_goal.neck_joint_goal.joints_goal.rotation.rpy, degrees=True
    )
    request_dict["elliptical_params"] = None
    return request_dict


def process_antenna_joint_goal(response: GoToRequest) -> Dict[str, Any]:
    """Process the response from a goto request containing an antenna joint goal."""
    request_dict = {}
    request_dict["part"] = response.joints_goal.antenna_joint_goal.antenna.id.name
    if request_dict["part"] == "antenna_right":
        request_dict["part"] = "r_antenna"
    elif request_dict["part"] == "antenna_left":
        request_dict["part"] = "l_antenna"
    request_dict["mode"] = get_interpolation_mode(response.interpolation_mode.interpolation_type)
    request_dict["interpolation_space"] = "joint_space"
    request_dict["duration"] = response.joints_goal.antenna_joint_goal.duration.value
    request_dict["target_joints"] = np.rad2deg(response.joints_goal.antenna_joint_goal.joint_goal.value)
    request_dict["elliptical_params"] = None
    return request_dict


def process_hand_joint_goal(response: GoToRequest) -> Dict[str, Any]:
    """Process the response from a goto request containing a hand joint goal."""
    request_dict = {}
    request_dict["part"] = response.joints_goal.hand_joint_goal.goal_request.id.name
    request_dict["mode"] = get_interpolation_mode(response.interpolation_mode.interpolation_type)
    request_dict["interpolation_space"] = "joint_space"
    request_dict["duration"] = response.joints_goal.hand_joint_goal.duration.value
    request_dict["target_joints"] = np.rad2deg(
        response.joints_goal.hand_joint_goal.goal_request.position.parallel_gripper.position.value
    )
    request_dict["elliptical_params"] = None
    return request_dict


def process_odometry_goal(response: GoToRequest) -> Dict[str, Any]:
    """Process the response from a goto request containing an odometry goal."""
    request_dict = {}
    request_dict["part"] = response.odometry_goal.odometry_goal.id.name
    request_dict["timeout"] = response.odometry_goal.timeout.value
    request_dict["distance_tolerance"] = response.odometry_goal.distance_tolerance.value
    request_dict["angle_tolerance"] = np.rad2deg(response.odometry_goal.angle_tolerance.value)
    request_dict["target"] = {
        "x": response.odometry_goal.odometry_goal.direction.x.value,
        "y": response.odometry_goal.odometry_goal.direction.y.value,
        "theta": np.rad2deg(response.odometry_goal.odometry_goal.direction.theta.value),
    }
    return request_dict


def build_simplified_joints_request(request_dict: Dict[str, Any]) -> SimplifiedRequest:
    """Build a SimplifiedRequest object from a dictionary of joints request details."""
    target = TargetJointsRequest(
        joints=request_dict.get("target_joints", None),
        pose=request_dict.get("target_pose", None),
    )

    joints_request = JointsRequest(
        target=target,
        duration=request_dict["duration"],
        mode=request_dict["mode"],
        interpolation_space=request_dict["interpolation_space"],
        elliptical_parameters=request_dict["elliptical_params"],
    )

    full_request = SimplifiedRequest(
        part=request_dict["part"],
        request=joints_request,
    )

    return full_request


def build_simplified_odometry_request(request_dict: Dict[str, Any]) -> SimplifiedRequest:
    """Build a SimplifiedRequest object from a dictionary of odomztry request details."""
    odometry_request = OdometryRequest(
        target=request_dict["target"],
        timeout=request_dict["timeout"],
        distance_tolerance=request_dict["distance_tolerance"],
        angle_tolerance=request_dict["angle_tolerance"],
    )

    full_request = SimplifiedRequest(
        part=request_dict["part"],
        request=odometry_request,
    )

    return full_request
