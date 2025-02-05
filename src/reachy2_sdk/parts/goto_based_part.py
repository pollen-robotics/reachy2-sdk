"""Reachy IGoToBasedPart interface.

Handles common interface for parts performing movement using goto mechanism.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from reachy2_sdk_api.goto_pb2 import GoalStatus, GoToAck, GoToId
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub

from ..utils.utils import (
    JointsRequest,
    OdometryRequest,
    SimplifiedRequest,
    arm_position_to_list,
    ext_euler_angles_to_list,
    get_interpolation_mode,
)
from .part import Part


class IGoToBasedPart(ABC):
    """Interface for parts of Reachy that use goto functions.

    The `IGoToBasedPart` class defines a common interface for handling goto-based movements. It is
    designed to be implemented by parts of the robot that perform movements via the goto mechanism,
    such as the Arm, Head, or potentially the MobileBase in the future.
    """

    def __init__(
        self,
        part: Part,
        goto_stub: GoToServiceStub,
    ) -> None:
        """Initialize the IGoToBasedPart interface.

        Sets up the common attributes needed for handling goto-based movements. This includes
        associating the part with the interface and setting up the gRPC stub for performing
        goto commands.

        Args:
            part: The robot part that uses this interface, such as an Arm or Head.
            goto_stub: The gRPC stub used to send goto commands to the robot part.
        """
        self.part = part
        self._goto_stub = goto_stub
        self._logger_goto = logging.getLogger(__name__)  # not using self._logger to avoid name conflict in multiple inheritance

    def get_goto_playing(self) -> GoToId:
        """Return the GoToId of the currently playing goto movement on a specific part."""
        response = self._goto_stub.GetPartGoToPlaying(self.part._part_id)
        return response

    def get_goto_queue(self) -> List[GoToId]:
        """Return a list of all GoToIds waiting to be played on a specific part."""
        response = self._goto_stub.GetPartGoToQueue(self.part._part_id)
        return [goal_id for goal_id in response.goto_ids]

    def cancel_all_goto(self) -> GoToAck:
        """Request the cancellation of all playing and waiting goto commands for a specific part.

        Returns:
            A GoToAck acknowledging the cancellation of all goto commands.
        """
        response = self._goto_stub.CancelPartAllGoTo(self.part._part_id)
        return response

    def _get_goto_request(self, goto_id: GoToId) -> Optional[SimplifiedRequest]:
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
        if not isinstance(goto_id, GoToId):
            raise TypeError(f"goto_id must be a GoToId, got {type(goto_id).__name__}")
        if goto_id.id == -1:
            raise ValueError("No answer was found for given move, goto_id is -1")

        response = self._goto_stub.GetGoToRequest(goto_id)

        if response.HasField("joints_goal"):
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

            joints_request = JointsRequest(
                goal_positions=goal_positions,
                duration=duration,
                mode=mode,
            )

            full_request = SimplifiedRequest(
                part=part,
                request=joints_request,
            )
        elif response.HasField("odometry_goal"):
            part = response.odometry_goal.odometry_goal.id.name
            odom_goal_positions = {}
            odom_goal_positions["x"] = response.odometry_goal.odometry_goal.direction.x.value
            odom_goal_positions["y"] = response.odometry_goal.odometry_goal.direction.y.value
            odom_goal_positions["theta"] = response.odometry_goal.odometry_goal.direction.theta.value
            odom_request = OdometryRequest(
                goal_positions=odom_goal_positions,
                timeout=response.odometry_goal.timeout.value,
                distance_tolerance=response.odometry_goal.distance_tolerance.value,
                angle_tolerance=response.odometry_goal.angle_tolerance.value,
            )
            full_request = SimplifiedRequest(
                part=part,
                request=odom_request,
            )

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
        t0 = time.time()
        self._logger_goto.info(f"Waiting for movement with {id}.")

        id_playing = self.get_goto_playing()
        while id_playing.id == -1:
            time.sleep(0.005)
            id_playing = self.get_goto_playing()

            # manage an id_playing staying at -1
            if time.time() - t0 > duration:
                self._logger_goto.warning(f"Waiting time for movement with {id} is timeout.")
                return

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
