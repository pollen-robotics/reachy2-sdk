"""Reachy Tripod module.

Handles all specific methods to the tripod.
"""

import logging
from typing import Tuple

import grpc
import numpy as np
from google.protobuf.wrappers_pb2 import FloatValue
from reachy2_sdk_api.part_pb2 import PartId
from reachy2_sdk_api.tripod_pb2 import Tripod as Tripod_proto
from reachy2_sdk_api.tripod_pb2 import TripodCommand, TripodState
from reachy2_sdk_api.tripod_pb2_grpc import TripodServiceStub


class Tripod:
    """The Tripod class represents the fixed tripod of the robot.

    The Tripod class is used to update manually the robot tripod's height value.
    """

    def __init__(
        self,
        proto_msg: Tripod_proto,
        initial_state: TripodState,
        grpc_channel: grpc.Channel,
    ):
        """Initialize the Tripod with its initial state and configuration.

        This sets up the tripod by assigning its state based on the provided initial values.

        Args:
            proto_msg: The protobuf message containing configuration details for the part.
            initial_state: The initial state of the tripod's joints.
            grpc_channel: The gRPC channel used to communicate with the DynamixelMotor service.
        """
        self._grpc_channel = grpc_channel
        self._stub = TripodServiceStub(grpc_channel)
        self._part_id = PartId(id=proto_msg.part_id.id, name=proto_msg.part_id.name)
        self._logger = logging.getLogger(__name__)

        self._present_position: float
        self._goal_position: float
        self._update_with(initial_state)

    def __repr__(self) -> str:
        """Clean representation of the Tripod."""
        repr_template = "<Tripod height={height} >"
        return repr_template.format(
            height=round(self.height, 3),
        )

    @property
    def height(self) -> float:
        """Get the current height of the robot torso in meters."""
        return float(np.round(self._present_position, 3))

    def set_height(self, height: float) -> None:
        """Set the height of the tripod.

        Args:
            height: The height of the tripod in meters.

        Raises:
            TypeError: If the height is not a float or int.
        """
        if not isinstance(height, float) and not isinstance(height, int):
            raise TypeError(f"height should be a float or int (got {type(height)} instead)!")

        limit_min, limit_max = self.get_limits()
        if not limit_min <= height <= limit_max:
            self._logger.warning(f"Height value {height} is out of bounds. ")
            height = np.clip(height, limit_min, limit_max)
            self._logger.warning(f"Setting height to {height}.")
        command = TripodCommand(
            part_id=self._part_id,
            height_position=FloatValue(value=height),
        )
        self._stub.SendCommand(command)

    def reset_height(self) -> None:
        """Reset the height of the tripod to its default position."""
        self._stub.ResetDefaultValues(self._part_id)

    def get_limits(self) -> Tuple[float, float]:
        """Get the limits of the tripod's height.

        Returns:
            A tuple containing the minimum and maximum height values.
        """
        response = self._stub.GetJointsLimits(self._part_id)
        return np.round(response.height_limit.min.value, 3), np.round(response.height_limit.max.value, 3)

    def _update_with(self, new_state: TripodState) -> None:
        """Update the present and goal positions of the joint with new state values.

        Args:
            new_state: A TripodState containing the new state values for the joint. Should include
                "present_position" and "goal_position", with corresponding FloatValue objects as values.
        """
        self._present_position = new_state.height.present_position.value
        self._goal_position = new_state.height.goal_position.value
