import grpc
import numpy as np
import pytest
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from pyquaternion import Quaternion
from reachy2_sdk_api.component_pb2 import PIDGains
from reachy2_sdk_api.head_pb2 import Head as Head_proto
from reachy2_sdk_api.head_pb2 import HeadState
from reachy2_sdk_api.kinematics_pb2 import ExtEulerAngles, Rotation3d
from reachy2_sdk_api.orbita3d_pb2 import Float3d, Orbita3dState, PID3d, Vector3d

from reachy2_sdk.orbita.utils import to_position
from reachy2_sdk.parts.head import Head


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
    present_rot = Rotation3d(rpy=ExtEulerAngles(roll=FloatValue(value=1), pitch=FloatValue(value=2), yaw=FloatValue(value=3)))
    goal_rot = Rotation3d(rpy=ExtEulerAngles(roll=FloatValue(value=4), pitch=FloatValue(value=5), yaw=FloatValue(value=6)))
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

    assert head.__repr__() != ""

    assert not head.neck.is_on()
    assert head.is_off()
    assert not head.is_on()

    assert len(head._actuators) == 1
    assert isinstance(head._actuators, dict)

    # use to_position()  to convert radian to degree
    assert head.neck.roll.goal_position == to_position(goal_rot.rpy.roll.value)
    assert head.neck.roll.present_position == to_position(present_rot.rpy.roll.value)
    assert head.neck.pitch.goal_position == to_position(goal_rot.rpy.pitch.value)
    assert head.neck.pitch.present_position == to_position(present_rot.rpy.pitch.value)
    assert head.neck.yaw.goal_position == to_position(goal_rot.rpy.yaw.value)
    assert head.neck.yaw.present_position == to_position(present_rot.rpy.yaw.value)

    assert head.joints["neck.pitch"]._axis_type == "pitch"
    assert head.joints["neck.pitch"].goal_position == to_position(goal_rot.rpy.pitch.value)
    assert head.joints["neck.pitch"].present_position == to_position(present_rot.rpy.pitch.value)

    assert head.joints["neck.yaw"]._axis_type == "yaw"
    assert head.joints["neck.yaw"].goal_position == to_position(goal_rot.rpy.yaw.value)
    assert head.joints["neck.yaw"].present_position == to_position(present_rot.rpy.yaw.value)

    assert head.joints["neck.roll"]._axis_type == "roll"
    assert head.joints["neck.roll"].goal_position == to_position(goal_rot.rpy.roll.value)
    assert head.joints["neck.roll"].present_position == to_position(present_rot.rpy.roll.value)

    # Head is off
    assert head.look_at(0, 0, 0).id == -1

    assert head.goto([0, 0, 0]).id == -1

    assert head.goto(None).id == -1

    # updating values
    compliance = BoolValue(value=False)

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
    present_rot = Rotation3d(rpy=ExtEulerAngles(roll=FloatValue(value=7), pitch=FloatValue(value=8), yaw=FloatValue(value=9)))
    goal_rot = Rotation3d(rpy=ExtEulerAngles(roll=FloatValue(value=10), pitch=FloatValue(value=11), yaw=FloatValue(value=12)))
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

    head_state = HeadState(neck_state=neck_state)

    head._update_with(head_state)

    assert head.neck.is_on()

    assert len(head._actuators) == 1
    assert isinstance(head._actuators, dict)

    with pytest.raises(ValueError):
        head._check_goto_parameters(duration=0, target=[0, 0, 0])
    with pytest.raises(ValueError):
        head._check_goto_parameters(duration=2, target=[0, 0, 0, 0])
    with pytest.raises(TypeError):
        head._check_goto_parameters(duration=2, target=np.eye(4))

    with pytest.raises(TypeError):
        head.set_speed_limits("wrong value")

    with pytest.raises(ValueError):
        head.set_speed_limits(120)

    with pytest.raises(ValueError):
        head.set_speed_limits(-10)

    with pytest.raises(TypeError):
        head.set_torque_limits("wrong value")

    with pytest.raises(ValueError):
        head.set_torque_limits(120)

    with pytest.raises(ValueError):
        head.set_torque_limits(-10)

    assert head.neck.roll.goal_position == to_position(goal_rot.rpy.roll.value)
    assert head.neck.roll.present_position == to_position(present_rot.rpy.roll.value)
    assert head.neck.pitch.goal_position == to_position(goal_rot.rpy.pitch.value)
    assert head.neck.pitch.present_position == to_position(present_rot.rpy.pitch.value)
    assert head.neck.yaw.goal_position == to_position(goal_rot.rpy.yaw.value)
    assert head.neck.yaw.present_position == to_position(present_rot.rpy.yaw.value)

    assert head.joints["neck.pitch"]._axis_type == "pitch"
    assert head.joints["neck.pitch"].goal_position == to_position(goal_rot.rpy.pitch.value)
    assert head.joints["neck.pitch"].present_position == to_position(present_rot.rpy.pitch.value)

    assert head.joints["neck.yaw"]._axis_type == "yaw"
    assert head.joints["neck.yaw"].goal_position == to_position(goal_rot.rpy.yaw.value)
    assert head.joints["neck.yaw"].present_position == to_position(present_rot.rpy.yaw.value)

    assert head.joints["neck.roll"]._axis_type == "roll"
    assert head.joints["neck.roll"].goal_position == to_position(goal_rot.rpy.roll.value)
    assert head.joints["neck.roll"].present_position == to_position(present_rot.rpy.roll.value)

    with pytest.raises(ValueError):
        head.look_at(1, 0, -0.2, duration=0)

    with pytest.raises(ValueError):
        quat = Quaternion(axis=[1, 0, 0], angle=20.0)
        head.goto(quat, duration=0)

    with pytest.raises(ValueError):
        head.goto([20, 30, 10], duration=0)
