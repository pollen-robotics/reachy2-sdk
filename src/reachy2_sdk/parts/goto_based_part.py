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
        # fmt: off
        """
        Return the ID of the currently playing goto movement on a specific part.

        Returns:  
            GoToId: The ID of the goto currently playing on the part.  
        """
        # fmt: on
        response = self._goto_stub.GetPartGoToPlaying(self.part._part_id)
        return response

    def get_goto_queue(self) -> List[GoToId]:
        # fmt: off
        """
        Return a list of all goto IDs waiting to be played on a specific part.

        Returns:  
            List[GoToId]: A list of all goto IDs waiting to be played on the part.  
        """
        # fmt: on
        response = self._goto_stub.GetPartGoToQueue(self.part._part_id)
        return [goal_id for goal_id in response.goto_ids]

    def cancel_all_goto(self) -> GoToAck:
        # fmt: off
        """
        Request the cancellation of all playing and waiting goto commands for a specific part.

        Returns:  
            GoToAck: An object acknowledging the cancellation of all goto commands.  
        """
        # fmt: on
        response = self._goto_stub.CancelPartAllGoTo(self.part._part_id)
        return response

    def _get_goto_joints_request(self, goto_id: GoToId) -> Optional[SimplifiedRequest]:
        # fmt: off
        """
        Return the part affected, joint goal positions, duration, and mode for the given GoToId.

        The part can be 'r_arm', 'l_arm', or 'head'.

        Args:  
            - **goto_id** (GoToId): The ID of the goto command for which to retrieve the details.  

        Returns:  
            Optional[SimplifiedRequest]: A `SimplifiedRequest` object containing the part, goal_positions, 
            duration, and mode for the corresponding GoToId. The `goal_positions` are returned as a list in degrees.  
        """
        # fmt: on
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
        # fmt: off
        """
        Check if the goto movement has been completed or cancelled.

        Returns:  
            bool: `True` if the goto has been played or cancelled, `False` otherwise.  
        """
        # fmt: on
        state = self._goto_stub.GetGoToState(id)
        result = bool(
            state.goal_status == GoalStatus.STATUS_ABORTED
            or state.goal_status == GoalStatus.STATUS_CANCELED
            or state.goal_status == GoalStatus.STATUS_SUCCEEDED
        )
        return result
