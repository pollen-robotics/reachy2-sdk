import grpc
import pytest
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.component_pb2 import ComponentId, PIDGains
from reachy2_sdk_api.dynamixel_motor_pb2 import DynamixelMotorState

from reachy2_sdk.components.antenna import Antenna
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

    uid = 2
    name = "antenna"
    antenna = Antenna(uid=uid, name=name, initial_state=motor_state, grpc_channel=grpc_channel, goto_stub=None, part=None)

    assert antenna.__repr__() != ""

    assert antenna.goal_position == to_position(goal_position.value)
    assert antenna.present_position == to_position(present_position.value)

    antenna._check_goto_parameters(duration=1, target=3.0)

    with pytest.raises(ValueError):
        antenna._check_goto_parameters(duration=0, target=2.0)
    with pytest.raises(TypeError):
        antenna._check_goto_parameters(duration=2, target=[2.0])
    with pytest.raises(TypeError):
        antenna._check_goto_parameters(duration=2, target="error")

    compliance = BoolValue(value=False)

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
    antenna._update_with(motor_state)

    with pytest.raises(ValueError):
        antenna.goto(2.0, duration=0)
    with pytest.raises(TypeError):
        antenna.goto([20], duration=0)
