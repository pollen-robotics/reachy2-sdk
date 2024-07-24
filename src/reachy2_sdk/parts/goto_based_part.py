from abc import ABC
from typing import List

from reachy2_sdk_api.goto_pb2 import GoToAck, GoToId
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub

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

    def get_move_playing(self) -> GoToId:
        """Return the id of the goto currently playing on the part"""
        response = self._goto_stub.GetPartGoToPlaying(self.part._part_id)
        return response

    def get_moves_queue(self) -> List[GoToId]:
        """Return the list of all goto ids waiting to be played on the part"""
        response = self._goto_stub.GetPartGoToQueue(self.part._part_id)
        return [goal_id for goal_id in response.goto_ids]

    def cancel_all_moves(self) -> GoToAck:
        """Ask the cancellation of all waiting goto on the part"""
        response = self._goto_stub.CancelPartAllGoTo(self.part._part_id)
        return response
