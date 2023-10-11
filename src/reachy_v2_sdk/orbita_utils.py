import asyncio
from typing import Any, Dict, List

from .register import Register


class OrbitaJoint:
    present_position = Register(readonly=True, label="present_position")
    present_speed = Register(readonly=True, label="present_speed")
    present_load = Register(readonly=True, label="present_load")
    temperature = Register(readonly=True, label="temperature")
    goal_position = Register(readonly=False, label="goal_position")
    speed_limit = Register(readonly=False, label="speed_limit")
    torque_limit = Register(readonly=False, label="torque_limit")

    def __init__(self, initial_state: Dict[str, float], axis_type: str, actuator) -> None:
        self._actuator = actuator
        self.axis_type = axis_type
        self.pid = PID(p=0.0, i=0.0, d=0.0)

        self._state = initial_state

        for field in dir(self):
            value = getattr(self, field)
            if isinstance(value, Register):
                value.label = field

        self._register_needing_sync: List[str] = []

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in ["goal_position", "speed_limit", "torque_limit"]:
            self._state[__name] = __value
            async def set_in_loop():
                self._register_needing_sync.append(__name)
                self._actuator._need_sync.set()
            fut = asyncio.run_coroutine_threadsafe(set_in_loop(), self._actuator._loop)
            fut.result()
        super().__setattr__(__name, __value)


class PID:
    def __init__(self, p: float, i: float, d: float) -> None:
        self.p = p
        self.i = i
        self.d = d

    def __repr__(self) -> str:
        return f"PID(p={self.p}, i={self.i}, d={self.d})"
