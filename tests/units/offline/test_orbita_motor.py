import numpy as np
import pytest
from google.protobuf.wrappers_pb2 import FloatValue
from reachy2_sdk_api.component_pb2 import PIDGains

from reachy2_sdk.orbita.orbita_motor import OrbitaMotor


@pytest.mark.offline
def test_class() -> None:
    temperature = FloatValue(value=1)
    compliancy = FloatValue(value=2)
    speed = FloatValue(value=3)
    torque = FloatValue(value=4)
    pid = PIDGains(p=FloatValue(value=15), i=FloatValue(value=16), d=FloatValue(value=17))
    state = {"temperature": temperature, "compliant": compliancy, "speed_limit": speed, "torque_limit": torque, "pid": pid}
    motor = OrbitaMotor(initial_state=state, actuator=None)
    assert motor.speed_limit == 300
    assert motor.torque_limit == 400
    assert motor.pid == (pid.p.value, pid.i.value, pid.d.value)
    assert motor.temperature == 1
    assert motor.compliant == 2
