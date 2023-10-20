import asyncio
from typing import Any, Dict, List

from google.protobuf.wrappers_pb2 import FloatValue

from .register import Register

from google.protobuf.wrappers_pb2 import BoolValue


class OrbitaJoint:
    present_position = Register(readonly=True, type=FloatValue, label="present_position")
    goal_position = Register(readonly=False, type=FloatValue, label="goal_position")

    def __init__(self, initial_state: Dict[str, float], axis_type: str, actuator: Any) -> None:
        self._actuator = actuator
        self.axis_type = axis_type

        self._state = initial_state

        for field in dir(self):
            value = getattr(self, field)
            if isinstance(value, Register):
                value.label = field

        self._register_needing_sync: List[str] = []

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in ["goal_position"]:
            self._state[__name] = __value

            async def set_in_loop() -> None:
                self._register_needing_sync.append(__name)
                self._actuator._need_sync.set()

            fut = asyncio.run_coroutine_threadsafe(set_in_loop(), self._actuator._loop)
            fut.result()
        super().__setattr__(__name, __value)

    def __getitem__(self, field: str) -> float:
        return self._state[field]

    # def __setitem__(self, field: str, value: float) -> None:
    #     self._state[field] = value


class OrbitaMotor:
    temperature = Register(readonly=True, type=FloatValue, label="temperature")
    speed_limit = Register(readonly=False, type=FloatValue, label="speed_limit")
    torque_limit = Register(readonly=False, type=FloatValue, label="torque_limit")
    compliant = Register(readonly=True, type=BoolValue, label="compliant")

    def __init__(self, initial_state: Dict[str, float], actuator: Any) -> None:
        self._actuator = actuator
        self.pid = PID(p=0.0, i=0.0, d=0.0)

        self._state = initial_state

        for field in dir(self):
            value = getattr(self, field)
            if isinstance(value, Register):
                value.label = field

        self._register_needing_sync: List[str] = []

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in ["speed_limit", "torque_limit"]:
            self._state[__name] = __value

            async def set_in_loop() -> None:
                self._register_needing_sync.append(__name)
                self._actuator._need_sync.set()

            fut = asyncio.run_coroutine_threadsafe(set_in_loop(), self._actuator._loop)
            fut.result()
        super().__setattr__(__name, __value)

    def __getitem__(self, field: str) -> float:
        return self._state[field]

    # def __setitem__(self, field: str, value: float) -> None:
    #     self._state[field] = value


class OrbitaAxis:
    present_speed = Register(readonly=True, type=FloatValue, label="present_speed")
    present_load = Register(readonly=True, type=FloatValue, label="present_load")

    def __init__(self, initial_state: Dict[str, float]) -> None:
        self._state = initial_state

        for field in dir(self):
            value = getattr(self, field)
            if isinstance(value, Register):
                value.label = field

    def __getitem__(self, field: str) -> float:
        return self._state[field]

    def __setitem__(self, field: str, value: float) -> None:
        self._state[field] = value


class PID:
    def __init__(self, p: float, i: float, d: float) -> None:
        self.p = p
        self.i = i
        self.d = d

    def __repr__(self) -> str:
        return f"PID(p={self.p}, i={self.i}, d={self.d})"
