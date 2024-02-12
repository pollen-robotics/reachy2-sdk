import grpc
import pytest
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.component_pb2 import PIDGains

from reachy2_sdk.orbita.orbita2d import (
    Axis,
    Float2d,
    Orbita2d,
    Orbita2dState,
    PID2d,
    Pose2d,
    Vector2d,
)
from reachy2_sdk.orbita.utils import to_position


@pytest.mark.offline
def test_class() -> None:
    grpc_channel = grpc.insecure_channel("dummy:5050")
    compliance = BoolValue(value=True)
    present_position = Pose2d(axis_1=FloatValue(value=1), axis_2=FloatValue(value=2))
    goal_position = Pose2d(axis_1=FloatValue(value=3), axis_2=FloatValue(value=4))
    present_speed = Vector2d(x=FloatValue(value=5), y=FloatValue(value=6))
    present_load = Vector2d(x=FloatValue(value=7), y=FloatValue(value=8))
    temperature = Float2d(motor_1=FloatValue(value=9), motor_2=FloatValue(value=10))
    speed_limit = Float2d(motor_1=FloatValue(value=11), motor_2=FloatValue(value=12))
    torque_limit = Float2d(motor_1=FloatValue(value=13), motor_2=FloatValue(value=14))
    pid = PID2d(
        motor_1=PIDGains(p=FloatValue(value=15), i=FloatValue(value=16), d=FloatValue(value=17)),
        motor_2=PIDGains(p=FloatValue(value=18), i=FloatValue(value=19), d=FloatValue(value=20)),
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

    assert orbita2d.__repr__() != ""

    assert not orbita2d.is_on()
    # use to_position()  to convert radian to degree
    assert orbita2d.roll.goal_position == to_position(goal_position.axis_2.value)
    assert orbita2d.roll.present_position == to_position(present_position.axis_2.value)
    assert orbita2d.pitch.goal_position == to_position(goal_position.axis_1.value)
    assert orbita2d.pitch.present_position == to_position(present_position.axis_1.value)

    with pytest.raises(AttributeError):
        orbita2d.yaw

    pid_set = orbita2d.get_pid()
    assert pid_set["motor_1"][0] == pid.motor_1.p.value
    assert pid_set["motor_1"][1] == pid.motor_1.i.value
    assert pid_set["motor_1"][2] == pid.motor_1.d.value

    assert pid_set["motor_2"][0] == pid.motor_2.p.value
    assert pid_set["motor_2"][1] == pid.motor_2.i.value
    assert pid_set["motor_2"][2] == pid.motor_2.d.value

    torques_set = orbita2d.get_torque_limit()
    assert torques_set["motor_1"] == torque_limit.motor_1.value
    assert torques_set["motor_2"] == torque_limit.motor_2.value

    speed_set = orbita2d.get_speed_limit()
    assert speed_set["motor_1"] == to_position(speed_limit.motor_1.value)
    assert speed_set["motor_2"] == to_position(speed_limit.motor_2.value)

    orbita2d.temperatures["motor_1"] == temperature.motor_1.value
    orbita2d.temperatures["motor_2"] == temperature.motor_1.value

    # with pytest.raises(ValueError):
    #     orbita2d.set_speed_limit("wrong value")

    # with pytest.raises(ValueError):
    #     orbita2d.set_torque_limit("wrong value")

    # with pytest.raises(ValueError):
    #     orbita2d.set_pid("wrong value")

    # with pytest.raises(ValueError):
    #     orbita2d.set_pid(("1", 2, 3))

    pid_msg = orbita2d._build_grpc_cmd_msg("pid")
    assert isinstance(pid_msg, PID2d)
    assert pid.motor_1.p.value == pid_msg.motor_1.p.value
    assert pid.motor_1.i.value == pid_msg.motor_1.i.value
    assert pid.motor_1.d.value == pid_msg.motor_1.d.value

    assert pid.motor_2.p.value == pid_msg.motor_2.p.value
    assert pid.motor_2.i.value == pid_msg.motor_2.i.value
    assert pid.motor_2.d.value == pid_msg.motor_2.d.value

    float_msg = orbita2d._build_grpc_cmd_msg("speed_limit")
    assert isinstance(float_msg, Float2d)

    assert float_msg.motor_1.value == speed_limit.motor_1.value
    assert float_msg.motor_2.value == speed_limit.motor_2.value

    # simulated update

    compliance = BoolValue(value=False)
    present_position = Pose2d(axis_1=FloatValue(value=21), axis_2=FloatValue(value=22))
    goal_position = Pose2d(axis_1=FloatValue(value=23), axis_2=FloatValue(value=24))
    present_speed = Vector2d(x=FloatValue(value=25), y=FloatValue(value=26))
    present_load = Vector2d(x=FloatValue(value=27), y=FloatValue(value=28))
    temperature = Float2d(motor_1=FloatValue(value=29), motor_2=FloatValue(value=30))
    speed_limit = Float2d(motor_1=FloatValue(value=31), motor_2=FloatValue(value=32))
    torque_limit = Float2d(motor_1=FloatValue(value=33), motor_2=FloatValue(value=34))
    pid_new = PID2d(
        motor_1=PIDGains(p=FloatValue(value=35), i=FloatValue(value=36), d=FloatValue(value=37)),
        motor_2=PIDGains(p=FloatValue(value=38), i=FloatValue(value=39), d=FloatValue(value=40)),
    )
    orbita2d_state = Orbita2dState(
        present_position=present_position,
        present_speed=present_speed,
        present_load=present_load,
        temperature=temperature,
        compliant=compliance,
        goal_position=goal_position,
        pid=pid_new,
        speed_limit=speed_limit,
        torque_limit=torque_limit,
    )

    orbita2d._update_with(orbita2d_state)

    assert orbita2d.is_on()

    assert orbita2d.roll.goal_position == to_position(goal_position.axis_2.value)
    assert orbita2d.roll.present_position == to_position(present_position.axis_2.value)
    assert orbita2d.pitch.goal_position == to_position(goal_position.axis_1.value)
    assert orbita2d.pitch.present_position == to_position(present_position.axis_1.value)

    # pid not changed. testing against old values
    pid_set = orbita2d.get_pid()
    assert pid_set["motor_1"][0] == pid.motor_1.p.value
    assert pid_set["motor_1"][1] == pid.motor_1.i.value
    assert pid_set["motor_1"][2] == pid.motor_1.d.value

    assert pid_set["motor_2"][0] == pid.motor_2.p.value
    assert pid_set["motor_2"][1] == pid.motor_2.i.value
    assert pid_set["motor_2"][2] == pid.motor_2.d.value

    torques_set = orbita2d.get_torque_limit()
    assert torques_set["motor_1"] == torque_limit.motor_1.value
    assert torques_set["motor_2"] == torque_limit.motor_2.value

    speed_set = orbita2d.get_speed_limit()
    assert speed_set["motor_1"] == to_position(speed_limit.motor_1.value)
    assert speed_set["motor_2"] == to_position(speed_limit.motor_2.value)

    orbita2d.temperatures["motor_1"] == temperature.motor_1.value
    orbita2d.temperatures["motor_2"] == temperature.motor_1.value
