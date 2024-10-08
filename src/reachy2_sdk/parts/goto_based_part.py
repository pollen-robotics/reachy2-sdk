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

    def get_goto_playing(self) -> GoToId:
        """
        Returns the id of the goto currently playing on a specific part.

        Returns:
          the `GoToId` of the goto currently playing on the part.
        """
        response = self._goto_stub.GetPartGoToPlaying(self.part._part_id)
        return response

    def get_goto_queue(self) -> List[GoToId]:
        """
        Returns a list of all goto ids waiting to be played on a specific part.

        Returns:
          a list of all `GoToIds` waiting to be played on the part.
        """
        response = self._goto_stub.GetPartGoToQueue(self.part._part_id)
        return [goal_id for goal_id in response.goto_ids]

    def cancel_all_goto(self) -> GoToAck:
        """
        Requests the cancellation of all playing and waiting goto commands for a specific part.

        Returns:
          a `GoToAck` object, acknowledging the cancellation of all goto commands.
        """
        response = self._goto_stub.CancelPartAllGoTo(self.part._part_id)
        return response

    def _get_goto_joints_request(self, goto_id: GoToId) -> Optional[SimplifiedRequest]:
        """
        Returns the part affected, the joints goal positions, duration and mode of the corresponding GoToId. Part can be either
        'r_arm', 'l_arm' or 'head'.

        Returns:
            a `SimplifiedRequest` object containing the part, goal_positions, duration and mode of the corresponding GoToId.
        `goal_position` is returned as a list in degrees
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
        """
        Returns `True` if goto has been played or has been cancelled, `False` otherwise.

        Returns:
          a boolean value indicating whether the goto is finished.
        """
        state = self._goto_stub.GetGoToState(id)
        result = bool(
            state.goal_status == GoalStatus.STATUS_ABORTED
            or state.goal_status == GoalStatus.STATUS_CANCELED
            or state.goal_status == GoalStatus.STATUS_SUCCEEDED
        )
        return result
