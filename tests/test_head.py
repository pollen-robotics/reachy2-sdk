import grpc
import pytest
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.component_pb2 import PIDGains
from reachy2_sdk_api.head_pb2 import Head as Head_proto
from reachy2_sdk_api.head_pb2 import HeadState
from reachy2_sdk_api.kinematics_pb2 import ExtEulerAngles, Rotation3d
from reachy2_sdk_api.orbita3d_pb2 import Float3d, Orbita3dState, PID3d, Vector3d

from src.reachy2_sdk.head import Head
from src.reachy2_sdk.orbita_utils import _to_position


@pytest.mark.offline
def test_class() -> None:
    grpc_channel = grpc.insecure_channel("dummy:5050")

    compliance = BoolValue(value=True)

    pid = PID3d(
        motor_1=PIDGains(p=FloatValue(value=0), i=FloatValue(value=0), d=FloatValue(value=0)),
        motor_2=PIDGains(p=FloatValue(value=0), i=FloatValue(value=0), d=FloatValue(value=0)),
        motor_3=PIDGains(p=FloatValue(value=0), i=FloatValue(value=0), d=FloatValue(value=0)),
    )

    temperature = Float3d(motor_1=FloatValue(value=0), motor_2=FloatValue(value=0), motor_3=FloatValue(value=0))
    speed_limit = Float3d(motor_1=FloatValue(value=0), motor_2=FloatValue(value=0), motor_3=FloatValue(value=0))
    torque_limit = Float3d(motor_1=FloatValue(value=0), motor_2=FloatValue(value=0), motor_3=FloatValue(value=0))
    present_speed = Vector3d(x=FloatValue(value=0), y=FloatValue(value=0), z=FloatValue(value=0))
    present_load = Vector3d(x=FloatValue(value=0), y=FloatValue(value=0), z=FloatValue(value=0))
    present_rot = Rotation3d(rpy=ExtEulerAngles(roll=1, pitch=2, yaw=3))
    goal_rot = Rotation3d(rpy=ExtEulerAngles(roll=4, pitch=5, yaw=6))
    neck_state = Orbita3dState(
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
    head_proto = Head_proto()
    head_state = HeadState(neck_state=neck_state)
    head = Head(head_msg=head_proto, initial_state=head_state, grpc_channel=grpc_channel, goto_stub=None)

    assert head.neck.compliant

    # use _to_position()  to convert radian to degree
    assert head.neck.roll.goal_position == _to_position(goal_rot.rpy.roll)
    assert head.neck.roll.present_position == _to_position(present_rot.rpy.roll)
    assert head.neck.pitch.goal_position == _to_position(goal_rot.rpy.pitch)
    assert head.neck.pitch.present_position == _to_position(present_rot.rpy.pitch)
    assert head.neck.yaw.goal_position == _to_position(goal_rot.rpy.yaw)
    assert head.neck.yaw.present_position == _to_position(present_rot.rpy.yaw)
