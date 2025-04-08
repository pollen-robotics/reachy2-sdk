import grpc
import pytest
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.component_pb2 import ComponentId, PIDGains
from reachy2_sdk_api.dynamixel_motor_pb2 import DynamixelMotorState

from reachy2_sdk.dynamixel.dynamixel_motor import DynamixelMotor
from reachy2_sdk.orbita.utils import to_position


@pytest.mark.offline
def test_class() -> None:
    grpc_channel = grpc.insecure_channel("dummy:5050")

    compliance = BoolValue(value=True)

    pid = PIDGains(p=FloatValue(value=0), i=FloatValue(value=0), d=FloatValue(value=0))

    motor_id = ComponentId(id=1, name="motor")
    temperature = FloatValue(value=0)
    speed_limit = FloatValue(value=0)
    torque_limit = FloatValue(value=0)
    present_position = FloatValue(value=0.2)
    goal_position = FloatValue(value=0.3)
    present_speed = FloatValue(value=0)
    present_load = FloatValue(value=0)
    motor_state = DynamixelMotorState(
        id=motor_id,
        compliant=compliance,
        present_position=present_position,
        goal_position=goal_position,
        temperature=temperature,
        pid=pid,
        speed_limit=speed_limit,
        torque_limit=torque_limit,
        present_speed=present_speed,
        present_load=present_load,
    )
    motor = DynamixelMotor(uid=motor_id.id, name=motor_id.name, initial_state=motor_state, grpc_channel=grpc_channel)

    assert motor.__repr__() != ""

    assert not motor.is_on()

    motor.send_goal_positions()

    # use to_position()  to convert radian to degree
    assert motor.goal_position == to_position(goal_position.value)
    assert motor.present_position == to_position(present_position.value)

    # updating values
    compliance = BoolValue(value=False)

    pid = PIDGains(p=FloatValue(value=0), i=FloatValue(value=0), d=FloatValue(value=0))

    motor_id = ComponentId(id=1, name="motor")
    temperature = FloatValue(value=0)
    speed_limit = FloatValue(value=0)
    torque_limit = FloatValue(value=0)
    present_position = FloatValue(value=0.5)
    goal_position = FloatValue(value=0.4)
    present_speed = FloatValue(value=0)
    present_load = FloatValue(value=0)
    motor_state = DynamixelMotorState(
        id=motor_id,
        compliant=compliance,
        present_position=present_position,
        goal_position=goal_position,
        temperature=temperature,
        pid=pid,
        speed_limit=speed_limit,
        torque_limit=torque_limit,
        present_speed=present_speed,
        present_load=present_load,
    )

    motor._update_with(motor_state)

    assert motor.goal_position == to_position(goal_position.value)
    assert motor.present_position == to_position(present_position.value)

    assert motor.is_on()

    with pytest.raises(TypeError):
        motor.set_speed_limits("wrong value")

    with pytest.raises(ValueError):
        motor.set_speed_limits(120)

    with pytest.raises(ValueError):
        motor.set_speed_limits(-10)
