"""Reachy Arm module.

Handles all specific method to an Arm (left and/or right) especially:
- the forward kinematics
- the inverse kinematics
"""
import grpc
from reachy_sdk_api_v2.hand_pb2 import Hand as Hand_proto
from reachy_sdk_api_v2.hand_pb2 import HandState
from reachy_sdk_api_v2.hand_pb2_grpc import HandServiceStub
from reachy_sdk_api_v2.part_pb2 import PartId


class Hand:
    """Arm abstract class used for both left/right arms.

    It exposes the kinematics of the arm:
    - you can access the joints actually used in the kinematic chain,
    - you can compute the forward and inverse kinematics
    """

    def __init__(self, hand: Hand_proto, grpc_channel: grpc.Channel) -> None:
        """Set up the arm with its kinematics."""
        self._hand_stub = HandServiceStub(grpc_channel)
        self.part_id = PartId(id=hand.part_id)

    def open(self) -> None:
        self._hand_stub.OpenHand(self.part_id)

    def close(self) -> None:
        self._hand_stub.CloseHand(self.part_id)

    def turn_on(self) -> None:
        self._hand_stub.TurnOn(self.part_id)

    def turn_off(self) -> None:
        self._hand_stub.TurnOff(self.part_id)

    def _update_with(self, new_state: HandState) -> None:
        """Update the orbita with a newly received (partial) state received from the gRPC server."""
        pass
