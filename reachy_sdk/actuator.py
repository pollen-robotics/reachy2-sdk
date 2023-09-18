from abc import ABC
from typing import List
import numpy as np

from .joint import Joint
from .register import Register, MetaRegisterABCMeta

from google.protobuf.wrappers_pb2 import BoolValue, FloatValue, UInt32Value
from reachy_sdk_api.joint_pb2 import PIDValue

# from reachy_v2_sdk_api.joint_pb2 import PIDValue


def _to_position(internal_pos: float) -> float:
    result: float
    result = round(np.rad2deg(internal_pos), 2)
    return result


def _to_internal_position(pos: float) -> float:
    result: float
    result = np.deg2rad(pos)
    return result


class Actuator(ABC, metaclass=MetaRegisterABCMeta):
    name = Register(readonly=True, type=str)
    uid = Register(readonly=True, type=UInt32Value)

    present_speed = Register(readonly=True, type=FloatValue, conversion=(_to_internal_position, _to_position))
    present_load = Register(readonly=True, type=FloatValue)
    temperatures = Register(readonly=True, type=FloatValue)

    compliant = Register(readonly=False, type=BoolValue)
    speed_limit = Register(
        readonly=False,
        type=FloatValue,
        conversion=(_to_internal_position, _to_position),
    )
    torque_limit = Register(readonly=False, type=FloatValue)

    pid = Register(readonly=False, type=PIDValue)

    def __init__(self) -> None:
        self.joints: List[Joint] = []
