"""Reachy Arm module.

Handles all specific method to an Arm (left and/or right) especially:
- the forward kinematics
- the inverse kinematics
"""
import grpc

from typing import List, Optional, Set

from reachy_sdk_api_v2.hand_pb2_grpc import HandStub
from reachy_sdk_api_v2.hand_pb2 import Hand
from reachy_sdk_api_v2.part_pb2 import PartId

import numpy as np


class Hand():
    """Arm abstract class used for both left/right arms.

    It exposes the kinematics of the arm:
    - you can access the joints actually used in the kinematic chain,
    - you can compute the forward and inverse kinematics
    """

    def __init__(self, hand: Hand, grpc_channel: grpc.Channel) -> None:
        """Set up the arm with its kinematics."""
        self._hand_stub = HandStub(grpc_channel)
        self.part_id = PartId(id=hand.part_id)

    def open(self):
        self._hand_stub.OpenHand(self.part_id)

    def close(self):
        self._hand_stub.CloseHand(self.part_id)

    def turn_on(self):
        self._hand_stub.TurnOn(self.part_id)
    
    def turn_off(self):
        self._hand_stub.TurnOff(self.part_id)