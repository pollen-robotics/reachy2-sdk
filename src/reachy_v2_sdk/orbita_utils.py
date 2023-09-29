class OrbitaAxis:
    def __init__(self, axis_type: str) -> None:
        self.axis_type = axis_type
        self.pid = PID(p=0.0, i=0.0, d=0.0)
        self._present_position = 0.0
        self._present_speed = 0.0
        self._present_load = 0.0
        self._temperature = 0.0

        self._goal_position = 0.0
        self._speed_limit = 0.0
        self._torque_limit = 0.0

    @property
    def present_position(self) -> float:
        return self._present_position

    @property
    def present_speed(self) -> float:
        return self._present_speed

    @property
    def present_load(self) -> float:
        return self._present_load

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def goal_position(self) -> float:
        return self._goal_position

    @goal_position.setter
    def goal_position(self, value: float) -> None:
        self._goal_position = value

    @property
    def speed_limit(self) -> float:
        return self._speed_limit

    @speed_limit.setter
    def speed_limit(self, value: float) -> None:
        self._speed_limit = value

    @property
    def torque_limit(self) -> float:
        return self._torque_limit

    @torque_limit.setter
    def torque_limit(self, value: float) -> None:
        self._torque_limit = value


class PID:
    def __init__(self, p: float, i: float, d: float) -> None:
        self.p = p
        self.i = i
        self.d = d

    def __repr__(self) -> str:
        return f"PID(p={self.p}, i={self.i}, d={self.d})"
