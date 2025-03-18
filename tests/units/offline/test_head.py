import grpc
import numpy as np
import pytest
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from pyquaternion import Quaternion
from reachy2_sdk_api.component_pb2 import ComponentId, PIDGains
from reachy2_sdk_api.dynamixel_motor_pb2 import (
    DynamixelMotor,
    DynamixelMotorState,
    DynamixelMotorStatus,
)
from reachy2_sdk_api.error_pb2 import Error
from reachy2_sdk_api.head_pb2 import Head as Head_proto
from reachy2_sdk_api.head_pb2 import HeadDescription, HeadState, HeadStatus
from reachy2_sdk_api.kinematics_pb2 import ExtEulerAngles, Rotation3d
from reachy2_sdk_api.orbita3d_pb2 import (
    Float3d,
    Orbita3d,
    Orbita3dState,
    Orbita3dStatus,
    PID3d,
    Vector3d,
)
from reachy2_sdk_api.part_pb2 import PartId

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

    head_proto = Head_proto(
        part_id=PartId(id=1, name="head"),
        description=HeadDescription(
            neck=Orbita3d(id=ComponentId(id=10, name="neck")),
        ),
    )

    head_state = HeadState(neck_state=neck_state)
    head = Head(head_msg=head_proto, initial_state=head_state, grpc_channel=grpc_channel, goto_stub=None)

    assert head.__repr__() != ""
    assert (
        str(head) == f"<Head on=False actuators=\n"
        f"\tneck: <Orbita3d on=False joints=\n"
        f'\t<OrbitaJoint axis_type="roll" present_position=57.3 goal_position=229.18 >\n'
        f'\t<OrbitaJoint axis_type="pitch" present_position=114.59 goal_position=286.48 >\n'
        f'\t<OrbitaJoint axis_type="yaw" present_position=171.89 goal_position=343.77 >\n'
        f">\n"
        f">"
    )
    assert not head.neck.is_on()
    assert head.is_off()
    assert not head.is_on()

    head.send_goal_positions()

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

    head._check_goto_parameters(duration=1, target=[0, 0, 0])

    with pytest.raises(ValueError):
        head._check_goto_parameters(duration=0, target=[0, 0, 0])
    with pytest.raises(ValueError):
        head._check_goto_parameters(duration=2, target=[0, 0, 0, 0])
    with pytest.raises(TypeError):
        head._check_goto_parameters(duration=2, target=np.eye(4))

    with pytest.raises(ValueError):
        head._goto_single_joint(neck_joint=0, goal_position=0, duration=0)

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

    with pytest.raises(ValueError):
        head.rotate_by(frame="wrong")

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

    error = Error(details="orbita3d error")
    orbita3d_status = Orbita3dStatus(errors=[error])
    error = Error(details="dynamixel error")
    dynamixel_status = DynamixelMotorStatus(errors=[error])
    head_status = HeadStatus(neck_status=orbita3d_status, l_antenna_status=dynamixel_status, r_antenna_status=dynamixel_status)
    head._update_audit_status(head_status)
    assert head.neck.status == "orbita3d error"


@pytest.mark.offline
def test_class_with_antennas() -> None:
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

    compliance1 = BoolValue(value=True)
    pid1 = PIDGains(p=FloatValue(value=0), i=FloatValue(value=0), d=FloatValue(value=0))
    motor1_id = ComponentId(id=1, name="motor1")
    temperature1 = FloatValue(value=0)
    speed_limit1 = FloatValue(value=0)
    torque_limit1 = FloatValue(value=0)
    present_position1 = FloatValue(value=0.2)
    goal_position1 = FloatValue(value=0.3)
    present_speed1 = FloatValue(value=0)
    present_load1 = FloatValue(value=0)
    motor_state1 = DynamixelMotorState(
        id=motor1_id,
        compliant=compliance1,
        present_position=present_position1,
        goal_position=goal_position1,
        temperature=temperature1,
        pid=pid1,
        speed_limit=speed_limit1,
        torque_limit=torque_limit1,
        present_speed=present_speed1,
        present_load=present_load1,
    )
    motor1 = DynamixelMotor(id=motor1_id)

    compliance2 = BoolValue(value=True)
    pid2 = PIDGains(p=FloatValue(value=0), i=FloatValue(value=0), d=FloatValue(value=0))
    motor2_id = ComponentId(id=2, name="motor2")
    temperature2 = FloatValue(value=0)
    speed_limit2 = FloatValue(value=0)
    torque_limit2 = FloatValue(value=0)
    present_position2 = FloatValue(value=0.2)
    goal_position2 = FloatValue(value=0.3)
    present_speed2 = FloatValue(value=0)
    present_load2 = FloatValue(value=0)
    motor_state2 = DynamixelMotorState(
        id=motor2_id,
        compliant=compliance2,
        present_position=present_position2,
        goal_position=goal_position2,
        temperature=temperature2,
        pid=pid2,
        speed_limit=speed_limit2,
        torque_limit=torque_limit2,
        present_speed=present_speed2,
        present_load=present_load2,
    )
    motor2 = DynamixelMotor(id=motor2_id)

    head_proto = Head_proto(
        part_id=PartId(id=1, name="head"),
        description=HeadDescription(
            neck=Orbita3d(id=ComponentId(id=10, name="neck")),
            l_antenna=motor1,
            r_antenna=motor2,
        ),
    )

    head_state = HeadState(neck_state=neck_state, l_antenna_state=motor_state1, r_antenna_state=motor_state2)
    head = Head(head_msg=head_proto, initial_state=head_state, grpc_channel=grpc_channel, goto_stub=None)

    assert head.__repr__() != ""

    assert not head.neck.is_on()
    assert head.is_off()
    assert not head.is_on()

    assert len(head._actuators) == 3
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

    assert head.joints["l_antenna"].goal_position == to_position(goal_position1.value)
    assert head.joints["l_antenna"].present_position == to_position(present_position1.value)

    assert head.joints["r_antenna"].goal_position == to_position(goal_position2.value)
    assert head.joints["r_antenna"].present_position == to_position(present_position2.value)

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

    compliance = BoolValue(value=False)

    pid = PID3d(
        motor_1=PIDGains(p=FloatValue(value=0), i=FloatValue(value=0), d=FloatValue(value=0)),
        motor_2=PIDGains(p=FloatValue(value=0), i=FloatValue(value=0), d=FloatValue(value=0)),
        motor_3=PIDGains(p=FloatValue(value=0), i=FloatValue(value=0), d=FloatValue(value=0)),
    )

    compliance3 = BoolValue(value=False)
    pid3 = PIDGains(p=FloatValue(value=0), i=FloatValue(value=0), d=FloatValue(value=0))
    motor2_id = ComponentId(id=2, name="motor2")
    temperature3 = FloatValue(value=0)
    speed_limit3 = FloatValue(value=0)
    torque_limit3 = FloatValue(value=0)
    present_position3 = FloatValue(value=0.2)
    goal_position3 = FloatValue(value=0.3)
    present_speed3 = FloatValue(value=0)
    present_load3 = FloatValue(value=0)
    motor_state3 = DynamixelMotorState(
        id=motor2_id,
        compliant=compliance3,
        present_position=present_position3,
        goal_position=goal_position3,
        temperature=temperature3,
        pid=pid3,
        speed_limit=speed_limit3,
        torque_limit=torque_limit3,
        present_speed=present_speed3,
        present_load=present_load3,
    )

    head_state = HeadState(neck_state=neck_state, l_antenna_state=motor_state3, r_antenna_state=motor_state2)

    head._update_with(head_state)

    assert head.neck.is_on()
    assert head.l_antenna.is_on()
    assert not head.r_antenna.is_on()
    assert not head.is_on()
    assert not head.is_off()

    assert head.joints["l_antenna"].goal_position == to_position(goal_position3.value)
    assert head.joints["l_antenna"].present_position == to_position(present_position3.value)

    assert head.joints["r_antenna"].goal_position == to_position(goal_position2.value)
    assert head.joints["r_antenna"].present_position == to_position(present_position2.value)
