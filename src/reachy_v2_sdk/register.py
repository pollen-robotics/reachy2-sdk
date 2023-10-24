from typing import Type, Any, Optional, Tuple, Callable
import asyncio
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue, UInt32Value


class Register:
    def __init__(
        self,
        readonly: bool,
        type: Type[Any],
        label: str,
        conversion: Optional[Tuple[Callable[[Any], Any], Callable[[Any], Any]]] = None,
    ) -> None:
        self.readonly = readonly
        self.internal_class = type
        self.label = label

        self.cvt_to_internal: Optional[Callable[[Any], Any]] = None
        self.cvt_to_external: Optional[Callable[[Any], Any]] = None
        if conversion is not None:
            self.cvt_to_internal, self.cvt_to_external = conversion

    def __get__(self, instance, owner):  # type: ignore
        if instance is None:
            return self
        value = self.unwrapped_value(instance._state[self.label])
        if self.cvt_to_external is not None:
            value = self.cvt_to_external(value)
        return value

    def __set__(self, instance, value):  # type: ignore
        if self.readonly:
            raise AttributeError("can't set attribute")
        if self.cvt_to_internal is not None:
            value = self.cvt_to_internal(value)
        instance._state[self.label] = self.wrapped_value(value)

        async def set_in_loop() -> None:
            instance._register_needing_sync.append(self.label)
            instance._actuator._need_sync.set()

        fut = asyncio.run_coroutine_threadsafe(set_in_loop(), instance._actuator._loop)
        fut.result()

    def unwrapped_value(self, value: Any) -> Any:
        """Unwrap the internal value to a more simple one."""
        if self.internal_class in (BoolValue, FloatValue, UInt32Value):
            return value.value
        elif self.internal_class.__name__ == "PIDGains":
            return (value.p.value, value.i.value, value.d.value)
        return value

    def wrapped_value(self, value: Any) -> Any:
        """Wrap the simple Python value to the corresponding gRPC one."""
        if self.internal_class in (BoolValue, FloatValue, UInt32Value):
            return self.internal_class(value=value)
        elif self.internal_class.__name__ == "PIDGains":
            return self.internal_class(p=FloatValue(value=value[0]), i=FloatValue(value=value[1]), d=FloatValue(value=value[2]))

        return value
