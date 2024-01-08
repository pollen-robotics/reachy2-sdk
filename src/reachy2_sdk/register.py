"""This module defines the Register class.

The Register class is used to define the registers of the motors and joints of the robot
such as the goal position, the present position, the speed limit, etc.
"""
import asyncio
from typing import Any, Callable, Optional, Tuple, Type

from google.protobuf.wrappers_pb2 import BoolValue, FloatValue, UInt32Value
from reachy2_sdk_api.component_pb2 import JointLimits


class Register:
    """Register class.

    This class is used to define the registers of the motors and joints of the robot
    such as the goal position, the present position, the speed limit, etc.
    """

    def __init__(
        self,
        readonly: bool,
        type: Type[Any],
        label: str,
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None,
        conversion: Optional[Tuple[Callable[[Any], Any], Callable[[Any], Any]]] = None,
    ) -> None:
        """Initialize a register with its type and label and if necessary, its conversion functions."""
        self.readonly = readonly
        self.internal_class = type
        self.label = label

        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

        self.cvt_to_internal: Optional[Callable[[Any], Any]] = None
        self.cvt_to_external: Optional[Callable[[Any], Any]] = None
        if conversion is not None:
            self.cvt_to_internal, self.cvt_to_external = conversion

    def __get__(self, instance, owner):  # type: ignore
        """Get the value of the register.

        If the register is for an angle value, convert the value from radians
        (value returned by the server) to degrees.
        """
        if instance is None:
            return self
        value = self.unwrapped_value(instance._state[self.label])
        if self.cvt_to_external is not None:
            if self.internal_class == JointLimits:
                value = (self.cvt_to_external(value[0]), self.cvt_to_external(value[1]))
            elif self.internal_class != BoolValue:
                value = self.cvt_to_external(value)
        return value

    def __set__(self, instance, value):  # type: ignore
        """Set the value of the register.

        If the register is for an angle value, convert the value from degrees
        (value given by the user) to radians.
        When the value is set, it is also added to the list of registers that
        need to be synced with the server. A thread created in ReachySDK will
        then take care of syncing the registers.
        """
        if self.readonly:
            raise AttributeError("can't set attribute")
        if self.cvt_to_internal is not None:
            value = self.cvt_to_internal(value)
        instance._state[self.label] = self.wrapped_value(self.bound(value, instance))

        async def set_in_loop() -> None:
            instance._register_needing_sync.append(self.label)
            instance._actuator._need_sync.set()

        fut = asyncio.run_coroutine_threadsafe(set_in_loop(), instance._actuator._loop)
        fut.result()

    def unwrapped_value(self, value: Any) -> Any:
        """Unwrap the internal value from gRPC protobuf to a simple Python value."""
        if self.internal_class in (BoolValue, FloatValue, UInt32Value):
            return value.value
        elif self.internal_class.__name__ == "PIDGains":
            return (value.p.value, value.i.value, value.d.value)
        elif self.internal_class.__name__ == "JointLimits":
            return (value.min.value, value.max.value)
        return value

    def wrapped_value(self, value: Any) -> Any:
        """Wrap the simple Python value to the corresponding gRPC one."""
        if self.internal_class in (BoolValue, FloatValue, UInt32Value):
            return self.internal_class(value=value)
        elif self.internal_class.__name__ == "PIDGains":
            return self.internal_class(
                p=FloatValue(value=value[0]),
                i=FloatValue(value=value[1]),
                d=FloatValue(value=value[2]),
            )
        elif self.internal_class.__name__ == "JointLimits":
            return self.internal_class(min=FloatValue(value=value[0]), max=FloatValue(value=value[1]))
        return value

    def bound(self, value: float, instance: Any) -> float:
        new_value = value
        if self.label == "goal_position":
            if self.cvt_to_internal is not None:
                self.lower_limit = self.cvt_to_internal(instance.joint_limit[0])
                self.upper_limit = self.cvt_to_internal(instance.joint_limit[1])
        if self.upper_limit is not None and self.lower_limit is not None:
            new_value = max(self.lower_limit, min(self.upper_limit, value))
            if new_value != value:
                if self.cvt_to_external is not None:
                    print(
                        f"""{self.label} should be in \
{self.cvt_to_external(self.lower_limit), self.cvt_to_external(self.upper_limit)}. \
Got {self.cvt_to_external(value)}, set {self.cvt_to_external(new_value)}"""
                                    )
                else:
                    print(
                        f"""{self.label} should be in {self.lower_limit, self.upper_limit}. \
Got {value}, set {new_value}"""
                    )
        return new_value
