import grpc
import pytest
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.component_pb2 import PIDGains

from src.reachy2_sdk.orbita2d import (
    Axis,
    Float2d,
    Orbita2d,
    Orbita2dState,
    PID2d,
    Pose2d,
    Vector2d,
)
from src.reachy2_sdk.orbita_utils import _to_position


@pytest.mark.offline
def test_class() -> None:
    grpc_channel = grpc.insecure_channel("dummy:5050")
    compliance = BoolValue(value=True)
    present_position = Pose2d(axis_1=FloatValue(value=1), axis_2=FloatValue(value=2))
    goal_position = Pose2d(axis_1=FloatValue(value=3), axis_2=FloatValue(value=4))
    present_speed = Vector2d(x=FloatValue(value=0), y=FloatValue(value=0))
    present_load = Vector2d(x=FloatValue(value=0), y=FloatValue(value=0))
    temperature = Float2d(motor_1=FloatValue(value=0), motor_2=FloatValue(value=0))
    speed_limit = Float2d(motor_1=FloatValue(value=0), motor_2=FloatValue(value=0))
    torque_limit = Float2d(motor_1=FloatValue(value=0), motor_2=FloatValue(value=0))
    pid = PID2d(
        motor_1=PIDGains(p=FloatValue(value=0), i=FloatValue(value=0), d=FloatValue(value=0)),
        motor_2=PIDGains(p=FloatValue(value=0), i=FloatValue(value=0), d=FloatValue(value=0)),
    )
    orbita2d_state = Orbita2dState(
        present_position=present_position,
        present_speed=present_speed,
        present_load=present_load,
        temperature=temperature,
        compliant=compliance,
        goal_position=goal_position,
        pid=pid,
        speed_limit=speed_limit,
        torque_limit=torque_limit,
    )
    orbita2d = Orbita2d(
        uid=0, name="unit_test", axis1=Axis.PITCH, axis2=Axis.ROLL, initial_state=orbita2d_state, grpc_channel=grpc_channel
    )

    assert orbita2d.compliant
    assert orbita2d.roll.goal_position == _to_position(goal_position.axis_2.value)
    assert orbita2d.roll.present_position == _to_position(present_position.axis_2.value)
    assert orbita2d.pitch.goal_position == _to_position(goal_position.axis_1.value)
    assert orbita2d.pitch.present_position == _to_position(present_position.axis_1.value)

    with pytest.raises(AttributeError):
        orbita2d.yaw
