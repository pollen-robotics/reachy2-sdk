from abc import ABC
from typing import List

from .actuator import Actuator


class Arm(ABC):
    def __init__(self) -> None:
        self.actuators: List[Actuator] = []


class LeftArm(Arm):
    def __init__(self) -> None:
        super().__init__()


class RightArm(Arm):
    def __init__(self) -> None:
        super().__init__()
