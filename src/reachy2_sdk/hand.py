import grpc
from reachy2_sdk_api.hand_pb2 import Hand as Hand_proto
from reachy2_sdk_api.hand_pb2 import HandState
from reachy2_sdk_api.hand_pb2_grpc import HandServiceStub
from reachy2_sdk_api.part_pb2 import PartId

from register import Register
from typing import Any


def _to_position(internal_pos: float) -> Any:
    return round(np.rad2deg(internal_pos), 2)


def _to_internal_position(pos: float) -> Any:
    try:
        return np.deg2rad(pos)
    except TypeError:
        raise TypeError(f"Excepted one of: int, float, got {type(pos).__name__}")


class Hand:
    def __init__(self, hand_msg: Hand_proto, initial_state: HandState, grpc_channel: grpc.Channel) -> None:
        """Set up the arm with its kinematics."""
        self._hand_stub = HandServiceStub(grpc_channel)
        self.type = "gripper"
        self.part_id = PartId(id=hand_msg.part_id)
        self.joints = []
        self.motors = []

    def __repr__(self) -> str:
        return ""

    def open(self, percentage: float) -> None:
        if not percentage:
            self._hand_stub.OpenHand(self.part_id)
        else:
            # Compute goal position depending on percentage
            pass

    def close(self, percentage: float) -> None:
        if not percentage:
            self._hand_stub.CloseHand(self.part_id)
        else:
            # Compute goal position depending on percentage
            pass

    def turn_on(self) -> None:
        self._hand_stub.TurnOn(self.part_id)

    def turn_off(self) -> None:
        self._hand_stub.TurnOff(self.part_id)

    def _update_with(self, new_state: HandState) -> None:
        pass

    @property
    def is_holding_object(self) -> bool:
        return False

    @property
    def force_sensor(self) -> float:
        return 1.0


class HandJoint:
    goal_position = Register(readonly=False, type=float, label="goal_position")
    present_position = Register(readonly=True, type=float, label="present_position")

    def __init__(self) -> None:
        pass


class HandMotor:
    temperature = Register(readonly=True, type=FloatValue, label="temperature")
    speed_limit = Register(
        readonly=False, type=FloatValue, label="speed_limit", conversion=(_to_internal_position, _to_position)
    )
    torque_limit = Register(readonly=False, type=FloatValue, label="torque_limit")
    compliant = Register(readonly=True, type=BoolValue, label="compliant")

    # pid = Register(readonly=False, type=PIDGains, label="pid")

    def __init__(self) -> None:
        pass


class ForceSensor:
    def __init__(self) -> None:
        pass