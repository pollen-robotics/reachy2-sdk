"""Reachy IGoToBasedPart interface.

Handles common interface for parts performing movement using goto mechanism.
"""

from typing import List

from reachy2_sdk_api.goto_pb2 import GoToAck, GoToId
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.part_pb2 import PartId

from ..utils.goto_based_element import IGoToBasedElement


class IGoToBasedPart(IGoToBasedElement):
    """Interface for parts of Reachy that use goto functions.

    The `IGoToBasedPart` class defines a common interface for handling goto-based movements. It is
    designed to be implemented by parts of the robot that perform movements via the goto mechanism,
    such as the Arm, Head, or potentially the MobileBase in the future.
    """

    def __init__(
        self,
        part: PartId,
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
        super().__init__(part, goto_stub)

    def get_goto_playing(self) -> GoToId:
        """Return the GoToId of the currently playing goto movement on a specific part."""
        response = self._goto_stub.GetPartGoToPlaying(self._element_id)
        return response

    def get_goto_queue(self) -> List[GoToId]:
        """Return a list of all GoToIds waiting to be played on a specific part."""
        response = self._goto_stub.GetPartGoToQueue(self._element_id)
        return [goal_id for goal_id in response.goto_ids]

    def cancel_all_goto(self) -> GoToAck:
        """Request the cancellation of all playing and waiting goto commands for a specific part.

        Returns:
            A GoToAck acknowledging the cancellation of all goto commands.
        """
        response = self._goto_stub.CancelPartAllGoTo(self._element_id)
        return response
