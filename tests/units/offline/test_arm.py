import grpc
import numpy as np
import numpy.typing as npt
import pytest
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from online.test_basic_movements import build_pose_matrix
from reachy2_sdk_api.arm_pb2 import Arm as Arm_proto
from reachy2_sdk_api.arm_pb2 import ArmDescription, ArmState
from reachy2_sdk_api.component_pb2 import PIDGains
from reachy2_sdk_api.kinematics_pb2 import ExtEulerAngles, Rotation3d
from reachy2_sdk_api.orbita2d_pb2 import Orbita2d as Orbita2d_proto
from reachy2_sdk_api.orbita3d_pb2 import Float3d, Orbita3dState, PID3d, Vector3d
from reachy2_sdk_api.part_pb2 import PartId

from reachy2_sdk.orbita.orbita2d import (
    Axis,
    Float2d,
    Orbita2dState,
    PID2d,
    Pose2d,
    Vector2d,
)
from reachy2_sdk.orbita.utils import to_position
from reachy2_sdk.parts.arm import Arm


@pytest.mark.offline
def test_class() -> None:
    grpc_channel = grpc.insecure_channel("dummy:5050")

    compliance = BoolValue(value=True)
    shoulder = Orbita2d_proto(axis_1=Axis.PITCH, axis_2=Axis.ROLL, serial_number="tet")
    elbow = Orbita2d_proto(axis_1=Axis.PITCH, axis_2=Axis.YAW, serial_number="tet")

    arm_proto = Arm_proto(
        part_id=PartId(name="l_arm", id=2),
        description=ArmDescription(
            shoulder=shoulder,
            elbow=elbow,
        ),
    )

    present_position = Pose2d(axis_1=FloatValue(value=1), axis_2=FloatValue(value=2))
    goal_position = Pose2d(axis_1=FloatValue(value=3), axis_2=FloatValue(value=4))
    present_speed = Vector2d(x=FloatValue(value=6), y=FloatValue(value=6))
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
    present_rot = Rotation3d(
        rpy=ExtEulerAngles(roll=FloatValue(value=25), pitch=FloatValue(value=26), yaw=FloatValue(value=27))
    )
    goal_rot = Rotation3d(rpy=ExtEulerAngles(roll=FloatValue(value=28), pitch=FloatValue(value=29), yaw=FloatValue(value=30)))

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

    arm_state = ArmState(shoulder_state=orbita2d_state, elbow_state=orbita2d_state, wrist_state=orbita3d_state)
    arm = Arm(arm_msg=arm_proto, initial_state=arm_state, grpc_channel=grpc_channel, goto_stub=None)

    assert not arm.shoulder.is_on()

    assert len(arm._actuators) == 3
    assert isinstance(arm._actuators, dict)

    with pytest.raises(ValueError):
        arm._check_goto_parameters(duration=0, target=[0, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        arm._check_goto_parameters(duration=2, target=[0, 0, 0, 0])
    with pytest.raises(TypeError):
        arm._check_goto_parameters(duration=2, target="default")
    with pytest.raises(ValueError):
        arm._check_goto_parameters(duration=2, target=np.eye(3))
    with pytest.raises(TypeError):
        arm._check_goto_parameters(duration=2, target=np.eye(4), q0=np.eye(4))
    with pytest.raises(ValueError):
        arm._check_goto_parameters(duration=2, target=np.eye(4), q0=[0, 0, 0, 0, 0, 0])

    with pytest.raises(TypeError):
        arm.set_speed_limits("wrong value")

    with pytest.raises(ValueError):
        arm.set_speed_limits(120)

    with pytest.raises(ValueError):
        arm.set_speed_limits(-10)

    with pytest.raises(TypeError):
        arm.set_torque_limits("wrong value")

    with pytest.raises(ValueError):
        arm.set_torque_limits(120)

    with pytest.raises(ValueError):
        arm.set_torque_limits(-10)

    # use to_position()  to convert radian to degree
    assert arm.shoulder.roll.goal_position == to_position(goal_position.axis_2.value)
    assert arm.shoulder.roll.present_position == to_position(present_position.axis_2.value)
    assert arm.shoulder.pitch.goal_position == to_position(goal_position.axis_1.value)
    assert arm.shoulder.pitch.present_position == to_position(present_position.axis_1.value)

    with pytest.raises(AttributeError):
        arm.shoulder.yaw

    assert not arm.elbow.is_on()

    assert arm.elbow.yaw.goal_position == to_position(goal_position.axis_2.value)
    assert arm.elbow.yaw.present_position == to_position(present_position.axis_2.value)
    assert arm.elbow.pitch.goal_position == to_position(goal_position.axis_1.value)
    assert arm.elbow.pitch.present_position == to_position(present_position.axis_1.value)

    with pytest.raises(AttributeError):
        arm.elbow.roll

    assert not arm.wrist.is_on()

    assert arm.wrist.roll.goal_position == to_position(goal_rot.rpy.roll.value)
    assert arm.wrist.roll.present_position == to_position(present_rot.rpy.roll.value)
    assert arm.wrist.pitch.goal_position == to_position(goal_rot.rpy.pitch.value)
    assert arm.wrist.pitch.present_position == to_position(present_rot.rpy.pitch.value)
    assert arm.wrist.yaw.goal_position == to_position(goal_rot.rpy.yaw.value)
    assert arm.wrist.yaw.present_position == to_position(present_rot.rpy.yaw.value)

    assert arm.joints["shoulder.pitch"]._axis_type == "pitch"
    assert arm.joints["shoulder.pitch"].goal_position == to_position(goal_position.axis_1.value)
    assert arm.joints["shoulder.pitch"].present_position == to_position(present_position.axis_1.value)

    assert arm.joints["shoulder.roll"]._axis_type == "roll"
    assert arm.joints["shoulder.roll"].goal_position == to_position(goal_position.axis_2.value)
    assert arm.joints["shoulder.roll"].present_position == to_position(present_position.axis_2.value)

    with pytest.raises(KeyError):
        arm.joints["shoulder.yaw"]

    assert arm.joints["elbow.pitch"]._axis_type == "pitch"
    assert arm.joints["elbow.pitch"].goal_position == to_position(goal_position.axis_1.value)
    assert arm.joints["elbow.pitch"].present_position == to_position(present_position.axis_1.value)

    assert arm.joints["elbow.yaw"]._axis_type == "yaw"
    assert arm.joints["elbow.yaw"].goal_position == to_position(goal_position.axis_2.value)
    assert arm.joints["elbow.yaw"].present_position == to_position(present_position.axis_2.value)

    with pytest.raises(KeyError):
        arm.joints["elbow.roll"]

    assert arm.joints["wrist.pitch"]._axis_type == "pitch"
    assert arm.joints["wrist.pitch"].goal_position == to_position(goal_rot.rpy.pitch.value)
    assert arm.joints["wrist.pitch"].present_position == to_position(present_rot.rpy.pitch.value)

    assert arm.joints["wrist.yaw"]._axis_type == "yaw"
    assert arm.joints["wrist.yaw"].goal_position == to_position(goal_rot.rpy.yaw.value)
    assert arm.joints["wrist.yaw"].present_position == to_position(present_rot.rpy.yaw.value)

    assert arm.joints["wrist.roll"]._axis_type == "roll"
    assert arm.joints["wrist.roll"].goal_position == to_position(goal_rot.rpy.roll.value)
    assert arm.joints["wrist.roll"].present_position == to_position(present_rot.rpy.roll.value)

    with pytest.raises(ValueError):
        arm.inverse_kinematics(target=np.zeros((1, 1)))

    with pytest.raises(ValueError):
        arm.inverse_kinematics(target=np.zeros((4, 4)), q0=[0.0])

    with pytest.raises(ValueError):
        arm.inverse_kinematics(target=np.zeros((4, 4)), q0=np.zeros((4, 4)))

    with pytest.raises(ValueError):
        arm.goto(target=np.zeros((3, 3)))

    with pytest.raises(ValueError):
        arm.goto(target=np.zeros((4, 4)), q0=[0.0])

    # Arm is off
    assert arm.goto(target=np.zeros((4, 4)), q0=[0.0, 0, 0, 0, 0, 0, 0]).id == -1

    with pytest.raises(ValueError):
        arm.goto(target=[0.0])

    # Arm is off
    assert arm.goto(target=[0.0, 0, 0, 0, 0, 0, 0]).id == -1

    with pytest.raises(ValueError):
        arm.goto([0, 0, 0, -90, 0, 0, 0], duration=0)

    with pytest.raises(ValueError):
        arm.goto(build_pose_matrix(0.3, -0.4, -0.3), duration=0)
