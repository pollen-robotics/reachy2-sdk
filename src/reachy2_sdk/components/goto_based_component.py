"""Reachy IGoToBasedComponent interface.

Handles common interface for components performing movement using goto mechanism.
"""

from typing import List

from reachy2_sdk_api.component_pb2 import ComponentId
from reachy2_sdk_api.goto_pb2 import GoToAck, GoToId
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub

from ..utils.goto_based_element import IGoToBasedElement


class IGoToBasedComponent(IGoToBasedElement):
    """Interface for components of Reachy that use goto functions.

    The `IGoToBasedComponent` class defines a common interface for handling goto-based movements. It is
    designed to be implemented by components of the robot that perform movements via the goto mechanism.
    It is used by the Antenna class to handle goto-based movements, and is based on the IGoToBasedElement.
    """

    def __init__(
        self,
        component_id: ComponentId,
        goto_stub: GoToServiceStub,
    ) -> None:
        """Initialize the IGoToBasedComponent interface.

        Sets up the common attributes needed for handling goto-based movements. This includes
        associating the component with the interface and setting up the gRPC stub for performing
        goto commands.

        Args:
            component_id: The robot component that uses this interface.
            goto_stub: The gRPC stub used to send goto commands to the robot component.
        """
        super().__init__(component_id, goto_stub)

    def get_goto_playing(self) -> GoToId:
        """Return the GoToId of the currently playing goto movement on a specific component."""
        response = self._goto_stub.GetComponentGoToPlaying(self._element_id)
        return response

    def get_goto_queue(self) -> List[GoToId]:
        """Return a list of all GoToIds waiting to be played on a specific component."""
        response = self._goto_stub.GetComponentGoToQueue(self._element_id)
        return [goal_id for goal_id in response.goto_ids]

    def cancel_all_goto(self) -> GoToAck:
        """Request the cancellation of all playing and waiting goto commands for a specific component.

        Returns:
            A GoToAck acknowledging the cancellation of all goto commands.
        """
        response = self._goto_stub.CancelComponentAllGoTo(self._element_id)
        return response
