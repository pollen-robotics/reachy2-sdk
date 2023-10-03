from typing import Dict

from .register import Register


class OrbitaJoint:
    present_position = Register(readonly=True, label="present_position")
    present_speed = Register(readonly=True, label="present_speed")
    present_load = Register(readonly=True, label="present_load")
    temperature = Register(readonly=True, label="temperature")
    goal_position = Register(readonly=False, label="goal_position")
    speed_limit = Register(readonly=False, label="speed_limit")
    torque_limit = Register(readonly=False, label="torque_limit")

    def __init__(self, initial_state: Dict[str, float], axis_type: str) -> None:
        self.axis_type = axis_type
        self.pid = PID(p=0.0, i=0.0, d=0.0)

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
