from typing import List

import grpc
from reachy2_sdk_api.arm_pb2 import Arm as Arm_proto
from reachy2_sdk_api.arm_pb2_grpc import ArmServiceStub
from reachy2_sdk_api.goto_pb2 import GoToAck, GoToId
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.head_pb2 import Head as Head_proto
from reachy2_sdk_api.head_pb2_grpc import HeadServiceStub
from reachy2_sdk_api.mobile_base_utility_pb2 import MobileBase as MobileBase_proto
from reachy2_sdk_api.mobile_base_utility_pb2_grpc import MobileBaseUtilityServiceStub

from .part import Part


class GoToBasedPart(Part):
    """The Part class is an abstract class to represent any part of the robot.

    This class is meant to be derived by any part of the robot : Arm, Hand, Head, MobileBase
    """

    def __init__(
        self,
        proto_msg: Arm_proto | Head_proto | MobileBase_proto,
        grpc_channel: grpc.Channel,
        stub: ArmServiceStub | HeadServiceStub | MobileBaseUtilityServiceStub,
        goto_stub: GoToServiceStub,
    ) -> None:
        """Initialize the common attributes."""
        super().__init__(proto_msg, grpc_channel, stub)
        self._goto_stub = goto_stub

    def get_move_playing(self) -> GoToId:
        """Return the id of the goto currently playing on the arm"""
        response = self._goto_stub.GetPartGoToPlaying(self._part_id)
        return response

    def get_moves_queue(self) -> List[GoToId]:
        """Return the list of all goto ids waiting to be played on the arm"""
        response = self._goto_stub.GetPartGoToQueue(self._part_id)
        return [goal_id for goal_id in response.goto_ids]

    def cancel_all_moves(self) -> GoToAck:
        """Ask the cancellation of all waiting goto on the arm"""
        response = self._goto_stub.CancelPartAllGoTo(self._part_id)
        return response
