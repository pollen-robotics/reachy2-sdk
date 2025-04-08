import pytest
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue, UInt32Value

from reachy2_sdk.orbita.utils import (
    to_internal_position,
    wrapped_pid_value,
    wrapped_proto_value,
)


@pytest.mark.offline
def test_internal_pos() -> None:
    with pytest.raises(TypeError):
        to_internal_position("2.3")


@pytest.mark.offline
def test_wrapped_proto_value() -> None:
    val = int(5)
    int_val = UInt32Value(value=val)
    assert wrapped_proto_value(val).value == int_val.value

    val = int(5)
    int_val = UInt32Value(value=val)
    assert wrapped_proto_value(int_val).value == val

    val = float(5.0)
    float_val = FloatValue(value=val)
    assert wrapped_proto_value(val).value == float_val.value

    val = bool(True)
    bool_val = BoolValue(value=val)
    assert wrapped_proto_value(val).value == bool_val.value


@pytest.mark.offline
def test_wrapped_pid_value() -> None:
    p = 1.0
    i = 2.0
    d = 3.0
    pid = wrapped_pid_value((p, i, d))
    assert pid.p.value == p
    assert pid.i.value == i
    assert pid.d.value == d
