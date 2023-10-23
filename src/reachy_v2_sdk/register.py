from typing import Type, Any
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue, UInt32Value


class Register:
    def __init__(self, readonly: bool, type: Type[Any], label: str) -> None:
        self.readonly = readonly
        self.internal_class = type
        self.label = label

    def __get__(self, instance, owner):  # type: ignore
        if instance is None:
            return self
        value = self.unwrapped_value(instance._state[self.label])
        return value

    def __set__(self, instance, value):  # type: ignore
        if self.readonly:
            raise AttributeError("can't set attribute")
        instance._state[self.label] = value

    def unwrapped_value(self, value: Any) -> Any:
        """Unwrap the internal value to a more simple one."""
        if self.internal_class in (BoolValue, FloatValue, UInt32Value):
            return value.value
        if self.internal_class.__name__ == "PIDGains":
            return (value.pid.p, value.pid.i, value.pid.d)
        return value
