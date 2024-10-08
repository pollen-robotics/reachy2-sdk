import logging
import time
from abc import ABC
from typing import List, Optional

from reachy2_sdk_api.goto_pb2 import GoalStatus, GoToAck, GoToId
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub

from ..utils.utils import (
    SimplifiedRequest,
    arm_position_to_list,
    ext_euler_angles_to_list,
    get_interpolation_mode,
)
from .part import Part


class IGoToBasedPart(ABC):
    """The IGoToBasedPart class is an interface to define default behavior of all parts using goto functions.

    This interface is meant to be implemented by any relevant part of the robot : Arm, Head, (MobileBase in the future)
    """

    def __init__(
        self,
        part: Part,
        goto_stub: GoToServiceStub,
    ) -> None:
        """Initialize the common attributes."""
        self.part = part
        self._goto_stub = goto_stub
        self._logger_goto = logging.getLogger(__name__)  # not using self._logger to avoid name conflict in multiple inheritance

    def get_goto_playing(self) -> GoToId:
        """Return the id of the goto currently playing on the part"""
        response = self._goto_stub.GetPartGoToPlaying(self.part._part_id)
        return response

    def get_goto_queue(self) -> List[GoToId]:
        """Return the list of all goto ids waiting to be played on the part"""
        response = self._goto_stub.GetPartGoToQueue(self.part._part_id)
        return [goal_id for goal_id in response.goto_ids]

    def cancel_all_goto(self) -> GoToAck:
        """Ask the cancellation of all waiting goto on the part"""
        response = self._goto_stub.CancelPartAllGoTo(self.part._part_id)
        return response

    def _get_goto_joints_request(self, goto_id: GoToId) -> Optional[SimplifiedRequest]:
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

    def _is_goto_finished(self, id: GoToId) -> bool:
        """Return True if goto has been played and has been cancelled, False otherwise."""
        state = self._goto_stub.GetGoToState(id)
        result = bool(
            state.goal_status == GoalStatus.STATUS_ABORTED
            or state.goal_status == GoalStatus.STATUS_CANCELED
            or state.goal_status == GoalStatus.STATUS_SUCCEEDED
        )
        return result

    def _wait_goto(self, id: GoToId, timeout: float = 10) -> None:
        """Wait for a goto to finish. timeout is in seconds."""
        self._logger_goto.info(f"Waiting for movement with {id}.")

        t_goto = None  # timeout for this goto
        t_all = time.time()  # timeout for others
        while not self._is_goto_finished(id):
            time.sleep(0.1)

            if t_goto is None:
                if self.get_goto_playing() == id:
                    t_goto = time.time()
                elif time.time() - t_all > 60:  # ToDo: we need to know how long to wait for
                    self._logger_goto.warning(
                        f"Waiting time for movement with {id} is timeout. Previous movements are not finished."
                    )
                    return

            elif time.time() - t_goto > timeout:
                self._logger_goto.warning(f"Waiting time for movement with {id} is timeout.")
                return
        self._logger_goto.info(f"Movement with {id} finished.")
