import grpc
import pytest
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.component_pb2 import PIDGains
from reachy2_sdk_api.kinematics_pb2 import ExtEulerAngles, Rotation3d
from reachy2_sdk_api.orbita3d_pb2 import Float3d, Orbita3dState, PID3d, Vector3d

from src.reachy2_sdk.orbita3d import Orbita3d
from src.reachy2_sdk.orbita_utils import _to_position


@pytest.mark.offline
def test_class() -> None:
    grpc_channel = grpc.insecure_channel("dummy:5050")
    compliance = BoolValue(value=True)
    pid = PID3d(
        motor_1=PIDGains(p=FloatValue(value=1), i=FloatValue(value=2), d=FloatValue(value=3)),
        motor_2=PIDGains(p=FloatValue(value=4), i=FloatValue(value=5), d=FloatValue(value=6)),
        motor_3=PIDGains(p=FloatValue(value=7), i=FloatValue(value=8), d=FloatValue(value=9)),
    )

    temperature = Float3d(motor_1=FloatValue(value=10), motor_2=FloatValue(value=11), motor_3=FloatValue(value=12))
    speed_limit = Float3d(motor_1=FloatValue(value=13), motor_2=FloatValue(value=14), motor_3=FloatValue(value=15))
    torque_limit = Float3d(motor_1=FloatValue(value=16), motor_2=FloatValue(value=17), motor_3=FloatValue(value=18))
    present_speed = Vector3d(x=FloatValue(value=19), y=FloatValue(value=20), z=FloatValue(value=21))
    present_load = Vector3d(x=FloatValue(value=22), y=FloatValue(value=23), z=FloatValue(value=24))
    present_rot = Rotation3d(rpy=ExtEulerAngles(roll=25, pitch=26, yaw=27))
    goal_rot = Rotation3d(rpy=ExtEulerAngles(roll=28, pitch=29, yaw=30))
    orbita3d_state = Orbita3dState(
        compliant=compliance,
        present_position=present_rot,
        goal_position=goal_rot,
        temperature=temperature,
        pid=pid,
        speed_limit=speed_limit,
        torque_limit=torque_limit,
        present_speed=present_speed,
        present_load=present_load,
    )
    orbita3d = Orbita3d(uid=0, name="unit_test", initial_state=orbita3d_state, grpc_channel=grpc_channel)

    assert orbita3d.compliant

    # use _to_position()  to convert radian to degree
    assert orbita3d.roll.goal_position == _to_position(goal_rot.rpy.roll)
    assert orbita3d.roll.present_position == _to_position(present_rot.rpy.roll)
    assert orbita3d.pitch.goal_position == _to_position(goal_rot.rpy.pitch)
    assert orbita3d.pitch.present_position == _to_position(present_rot.rpy.pitch)
    assert orbita3d.yaw.goal_position == _to_position(goal_rot.rpy.yaw)
    assert orbita3d.yaw.present_position == _to_position(present_rot.rpy.yaw)

    pid_set = orbita3d.get_pid()
    assert pid_set["motor_1"][0] == pid.motor_1.p.value
    assert pid_set["motor_1"][1] == pid.motor_1.i.value
    assert pid_set["motor_1"][2] == pid.motor_1.d.value

    assert pid_set["motor_2"][0] == pid.motor_2.p.value
    assert pid_set["motor_2"][1] == pid.motor_2.i.value
    assert pid_set["motor_2"][2] == pid.motor_2.d.value

    assert pid_set["motor_3"][0] == pid.motor_3.p.value
    assert pid_set["motor_3"][1] == pid.motor_3.i.value
    assert pid_set["motor_3"][2] == pid.motor_3.d.value

    torques_set = orbita3d.get_torque_limit()
    assert torques_set["motor_1"] == torque_limit.motor_1.value
    assert torques_set["motor_2"] == torque_limit.motor_2.value
    assert torques_set["motor_3"] == torque_limit.motor_3.value

    speed_set = orbita3d.get_speed_limit()
    assert speed_set["motor_1"] == _to_position(speed_limit.motor_1.value)
    assert speed_set["motor_2"] == _to_position(speed_limit.motor_2.value)
    assert speed_set["motor_3"] == _to_position(speed_limit.motor_3.value)

    orbita3d.temperatures["motor_1"] == temperature.motor_1.value
    orbita3d.temperatures["motor_2"] == temperature.motor_1.value
    orbita3d.temperatures["motor_3"] == temperature.motor_3.value